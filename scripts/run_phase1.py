"""Phase 1 — the ablation grid, retrieval side (local, CPU).

Builds the hardened corpus once, then for each of the 10 configs: build/reuse
the index, run retrieval for the same 150 questions as Phase 0, write
results/phase1_<name>/{config.json, retrieval_report.json, generation_tasks.jsonl}.
Finally bundles all task files into results/phase1_tasks/ for one Colab session.
"""
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA, RESULTS, PipelineConfig, BGE_QUERY_PREFIX
from download_data import download_pubmedqa, download_pubmedqa_unlabeled
from corpus import build_split, build_phase1_corpus
from indexer import build_chunks, build_dense_index
from retrieve import Retriever, retrieval_report
from prompts import GENERATOR_SYSTEM_V1_HARDENED, GENERATOR_SYSTEM_V2, GENERATOR_USER_TEMPLATE

# Fast configs first (reuse MiniLM indexes). BERT-size embedding models last:
# their 52k-passage corpus embedding exceeds this machine's memory, so those
# indexes are built from precomputed Colab embeddings (data/emb_<fingerprint>.npy,
# produced by notebooks/embed_corpus_colab.ipynb) and are DEFERRED if absent.
BASE = PipelineConfig(name="p1_base")
GRID = [
    BASE,
    replace(BASE, name="p1_chunk256", chunk_size_tokens=256),
    replace(BASE, name="p1_chunk1024", chunk_size_tokens=1024),
    replace(BASE, name="p1_bm25", retrieval="bm25"),
    replace(BASE, name="p1_hybrid", retrieval="hybrid"),
    replace(BASE, name="p1_k3", top_k=3),
    replace(BASE, name="p1_k8", top_k=8),
    replace(BASE, name="p1_rerank", use_reranker=True),
    replace(BASE, name="p1_noctx", no_context=True),          # closed-book control (A)
    replace(BASE, name="p1_simgate", abstain_sim_threshold=-1.0),  # similarity gate (B)
    replace(BASE, name="p1_reasonfirst", prompt_version="v1", max_new_tokens=400),  # prompt-order robustness
    replace(BASE, name="p1_bge", embedding_model="BAAI/bge-base-en-v1.5",
            query_prefix=BGE_QUERY_PREFIX),
    replace(BASE, name="p1_pubmedbert",
            embedding_model="NeuML/pubmedbert-base-embeddings"),
]
LOCAL_EMBED_OK = {"sentence-transformers/all-MiniLM-L6-v2"}

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

pubmedqa = download_pubmedqa()
pqau = download_pubmedqa_unlabeled()
qfile = DATA / "phase1_questions.jsonl"
if not (DATA / "phase1_corpus.jsonl").exists() or not qfile.exists():
    build_phase1_corpus(pubmedqa, pqau)
questions = [json.loads(l) for l in qfile.read_text().splitlines()]
print(f"{len(questions)} questions "
      f"({sum(q['answerable'] for q in questions)} answerable)")

tasks_dir = RESULTS / "phase1_tasks"
tasks_dir.mkdir(exist_ok=True)
summary = []
chunks_cache = {}

import gc

from prompts import REFUSAL_STRING

def emit_tasks(cfg, run_dir, question_contexts, pre_responses=None):
    """question_contexts: [(q, context_str)]; pre_responses: {qid: text} for
    system-level abstentions that never reach the LLM."""
    pre_responses = pre_responses or {}
    system = GENERATOR_SYSTEM_V1_HARDENED if cfg.prompt_version == "v1" else GENERATOR_SYSTEM_V2
    tasks = []
    for q, context in question_contexts:
        t = {"qid": q["qid"], "answerable": q["answerable"], "gold_label": q["gold_label"],
             "config": cfg.name, "max_new_tokens": cfg.max_new_tokens,
             "system": system,
             "user": GENERATOR_USER_TEMPLATE.format(context=context, question=q["question"])}
        if q["qid"] in pre_responses:
            t["pre_response"] = pre_responses[q["qid"]]
        tasks.append(t)
    (run_dir / "generation_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    (run_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2))
    shutil.copy(run_dir / "generation_tasks.jsonl",
                tasks_dir / f"generation_tasks_{cfg.name}.jsonl")
    return tasks

for cfg in GRID:
    print(f"\n=== {cfg.name} ===", flush=True)
    run_dir = RESULTS / f"phase1_{cfg.name}"
    run_dir.mkdir(exist_ok=True)

    if cfg.no_context:
        # closed-book control: no retrieval at all — measures parametric leakage
        qc = [(q, "(no passages provided)") for q in questions]
        emit_tasks(cfg, run_dir, qc)
        summary.append({"config": cfg.name, "recall_at_k": None, "k": cfg.top_k,
                        "leaks": None, "control": True})
        print("closed-book control: 150 tasks, no retrieval", flush=True)
        continue

    ck = (cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
    # 8 GB machine: keep only the chunk list we need right now
    for stale in [k for k in chunks_cache if k != ck]:
        del chunks_cache[stale]
    gc.collect()
    if ck not in chunks_cache:
        chunks_cache[ck] = build_chunks(cfg, corpus_file="phase1_corpus.jsonl")
    chunks = chunks_cache[ck]
    coll = embed_model = None
    if cfg.retrieval in ("dense", "hybrid") or cfg.use_reranker:
        if cfg.embedding_model in LOCAL_EMBED_OK:
            coll, embed_model = build_dense_index(cfg, chunks)
        else:
            from indexer import get_complete_collection, build_dense_index_precomputed
            from sentence_transformers import SentenceTransformer
            coll = get_complete_collection(cfg, len(chunks))
            npy = DATA / f"emb_{cfg.index_fingerprint()}.npy"
            if coll is None and npy.exists():
                coll = build_dense_index_precomputed(cfg, chunks, npy)
            if coll is None:
                chunks_file = DATA / f"chunks_c{cfg.chunk_size_tokens}o{cfg.chunk_overlap_tokens}.jsonl"
                if not chunks_file.exists():
                    chunks_file.write_text("\n".join(json.dumps(c) for c in chunks))
                print(f"DEFERRED: {cfg.name} needs {npy.name} from Colab "
                      f"(notebooks/embed_corpus_colab.ipynb; chunks exported to {chunks_file.name})",
                      flush=True)
                summary.append({"config": cfg.name, "recall_at_k": None,
                                "k": cfg.top_k, "leaks": None, "deferred": True})
                continue
            embed_model = SentenceTransformer(cfg.embedding_model)  # queries only (150 texts)
    retriever = Retriever(cfg, coll, embed_model, chunks)
    report = retrieval_report(retriever, questions)
    print(f"recall@{cfg.top_k}: {report['recall_at_k']:.3f} | leaks: {report['holdout_leaks']}",
          flush=True)
    (run_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2))

    retrieved = [(q, retriever.retrieve(q["question"])) for q in questions]
    qc = [(q, "\n\n".join(f"[passage {i+1}] {g['text']}" for i, g in enumerate(got)))
          for q, got in retrieved]

    pre = {}
    if cfg.abstain_sim_threshold:
        import numpy as np
        top1 = {q["qid"]: (got[0]["score"] if got else 0.0) for q, got in retrieved}
        thr = cfg.abstain_sim_threshold
        if thr < 0:  # derive a priori threshold: 5th pct of answerable top-1 scores
            thr = float(np.percentile(
                [top1[q["qid"]] for q in questions if q["answerable"]], 5))
        gated = [q for q in questions if top1[q["qid"]] < thr]
        pre = {q["qid"]: REFUSAL_STRING for q in gated}
        g_ans = sum(1 for q in gated if q["answerable"])
        print(f"similarity gate thr={thr:.4f}: {len(gated)} gated "
              f"({g_ans} answerable, {len(gated)-g_ans} unanswerable)", flush=True)
        (run_dir / "gate_info.json").write_text(json.dumps(
            {"threshold": thr, "gated_qids": [q["qid"] for q in gated],
             "top1_scores": top1}, indent=2))

    emit_tasks(cfg, run_dir, qc, pre_responses=pre)
    summary.append({"config": cfg.name, "recall_at_k": report["recall_at_k"],
                    "k": cfg.top_k, "leaks": report["holdout_leaks"]})

(RESULTS / "phase1_retrieval_summary.json").write_text(json.dumps(summary, indent=2))
print("\n=== RETRIEVAL SUMMARY ===")
for s in summary:
    if s.get("deferred"):
        print(f"{s['config']:<16} DEFERRED (needs Colab embeddings)")
    elif s.get("control"):
        print(f"{s['config']:<16} closed-book control (no retrieval)")
    else:
        print(f"{s['config']:<16} recall@{s['k']}: {s['recall_at_k']:.3f}  leaks: {s['leaks']}")
print(f"\nColab bundle: {tasks_dir} ({len(list(tasks_dir.glob('*.jsonl')))} files)")
