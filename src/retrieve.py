"""Retrieval strategies + retrieval-side metrics (recall@k by construction)."""
import numpy as np
from rank_bm25 import BM25Okapi
from config import PipelineConfig


class Retriever:
    def __init__(self, cfg: PipelineConfig, coll, embed_model, chunks):
        self.cfg = cfg
        self.coll = coll
        self.embed = embed_model
        self.chunks = chunks
        self._bm25 = None
        self._reranker = None
        if cfg.retrieval in ("bm25", "hybrid"):
            self._bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
        if cfg.use_reranker:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(cfg.reranker_model)

    def _dense(self, query: str, k: int):
        q = self.embed.encode([self.cfg.query_prefix + query], normalize_embeddings=True)
        r = self.coll.query(query_embeddings=q.tolist(), n_results=k)
        return [
            {"chunk_id": i, "text": d, "source_pmid": m["source_pmid"], "score": 1 - dist}
            for i, d, m, dist in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]

    def _sparse(self, query: str, k: int):
        scores = self._bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:k]
        return [
            {"chunk_id": self.chunks[i]["chunk_id"], "text": self.chunks[i]["text"],
             "source_pmid": self.chunks[i]["source_pmid"], "score": float(scores[i])}
            for i in idx
        ]

    def retrieve(self, query: str):
        k = self.cfg.top_k
        fetch = k * 4 if self._reranker else k
        if self.cfg.retrieval == "dense":
            got = self._dense(query, fetch)
        elif self.cfg.retrieval == "bm25":
            got = self._sparse(query, fetch)
        else:  # hybrid: reciprocal rank fusion
            dense, sparse = self._dense(query, fetch * 2), self._sparse(query, fetch * 2)
            rrf = {}
            for lst in (dense, sparse):
                for rank, item in enumerate(lst):
                    rrf.setdefault(item["chunk_id"], [0.0, item])
                    rrf[item["chunk_id"]][0] += 1.0 / (60 + rank + 1)
            got = [item for _, item in sorted(rrf.values(), key=lambda x: -x[0])[:fetch]]
        if self._reranker:
            scores = self._reranker.predict([(query, g["text"]) for g in got])
            got = [g for _, g in sorted(zip(scores, got), key=lambda x: -x[0])][:k]
        return got[:k]


def retrieval_report(retriever: Retriever, questions):
    """recall@k: for answerable questions, did any chunk from the question's own
    gold paper reach the top-k? For held-out questions this must be 0 (leak check)."""
    hits, n_ans, leaked = 0, 0, 0
    records = []
    for q in questions:
        got = retriever.retrieve(q["question"])
        gold_hit = any(g["source_pmid"] == q["qid"] for g in got)
        if q["answerable"]:
            n_ans += 1
            hits += gold_hit
        elif gold_hit:
            leaked += 1
        records.append({"qid": q["qid"], "answerable": q["answerable"],
                        "gold_in_topk": gold_hit,
                        "retrieved": [g["chunk_id"] for g in got]})
    return {"recall_at_k": hits / max(1, n_ans), "n_answerable": n_ans,
            "holdout_leaks": leaked, "records": records}
