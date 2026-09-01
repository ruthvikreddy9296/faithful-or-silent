"""Chunk corpus passages and build/reuse the retrieval indexes."""
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from config import DATA, PipelineConfig


def word_chunks(text: str, size_tokens: int, overlap_tokens: int):
    """Approximate token chunking by words (4/3 words ~ 1 token). Keeps the
    chunker embedding-model-agnostic; exact tokenizer parity is not required
    for ablation purposes."""
    words = text.split()
    size = max(1, int(size_tokens * 0.75))
    overlap = int(overlap_tokens * 0.75)
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)] or [""]


def build_chunks(cfg: PipelineConfig, corpus_file="pilot_corpus.jsonl"):
    docs = [json.loads(l) for l in (DATA / corpus_file).read_text().splitlines()]
    chunks = []
    for d in docs:
        for j, ch in enumerate(word_chunks(d["text"], cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)):
            if ch.strip():
                chunks.append(
                    {"chunk_id": f'{d["doc_id"]}_c{j}', "text": ch,
                     "source_pmid": d["source_pmid"]}
                )
    return chunks


def get_complete_collection(cfg: PipelineConfig, n_chunks: int):
    """Return the existing collection if complete, else None (no embedding)."""
    fp = cfg.index_fingerprint()
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", fp)[:60]
    try:
        coll = client.get_collection(coll_name)
        if coll.count() == n_chunks:
            return coll
    except Exception:
        pass
    return None


def build_dense_index_precomputed(cfg: PipelineConfig, chunks, npy_path):
    """Assemble the index from Colab-precomputed embeddings — no local encoding
    of the corpus (only queries get encoded locally at retrieval time)."""
    import numpy as np
    emb = np.load(npy_path).astype("float32")
    assert len(emb) == len(chunks), f"{npy_path}: {len(emb)} vs {len(chunks)} chunks"
    fp = cfg.index_fingerprint()
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", fp)[:60]
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
    B = 512
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        coll.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=emb[i : i + len(batch)].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"source_pmid": c["source_pmid"]} for c in batch],
        )
    print(f"  [{fp}] index assembled from {npy_path.name}: {coll.count()} vectors")
    return coll


def build_dense_index(cfg: PipelineConfig, chunks, reuse=True):
    """One persistent Chroma collection per (embedding, chunking) fingerprint.
    If a complete collection already exists, reuse it instead of re-embedding."""
    fp = cfg.index_fingerprint()
    model = SentenceTransformer(cfg.embedding_model)
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", fp)[:60]
    try:
        coll = client.get_collection(coll_name)
        if reuse and coll.count() == len(chunks):
            return coll, model
        client.delete_collection(coll_name)
    except Exception:
        pass
    coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
    B = 256
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        emb = model.encode([c["text"] for c in batch], show_progress_bar=False,
                           normalize_embeddings=True, batch_size=64)
        coll.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=emb.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"source_pmid": c["source_pmid"]} for c in batch],
        )
        if (i // B) % 20 == 0:
            print(f"  [{fp}] embedded {i + len(batch)}/{len(chunks)}", flush=True)
    return coll, model
