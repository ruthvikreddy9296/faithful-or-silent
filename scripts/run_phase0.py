"""Phase 0 pilot — retrieval side, runs locally on CPU.

Steps: download data -> build held-out corpus split -> chunk+index -> retrieve
for all 150 pilot questions -> write (a) retrieval report, (b) generation tasks
JSONL for the Colab generation notebook.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA, RESULTS, PipelineConfig
from download_data import download_pubmedqa
from corpus import build_split
from indexer import build_chunks, build_dense_index
from retrieve import Retriever, retrieval_report
from prompts import GENERATOR_SYSTEM, GENERATOR_USER_TEMPLATE

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

cfg = PipelineConfig()
run_dir = RESULTS / f"phase0_{cfg.name}"
run_dir.mkdir(exist_ok=True)

pubmedqa = download_pubmedqa()
corpus_docs, questions = build_split(pubmedqa)

chunks = build_chunks(cfg)
print(f"{len(chunks)} chunks (size={cfg.chunk_size_tokens}t overlap={cfg.chunk_overlap_tokens}t)")
coll, embed_model = build_dense_index(cfg, chunks)
print(f"dense index: {coll.count()} vectors ({cfg.embedding_model})")

retriever = Retriever(cfg, coll, embed_model, chunks)
report = retrieval_report(retriever, questions)
print(f"recall@{cfg.top_k} on answerable: {report['recall_at_k']:.3f} "
      f"(n={report['n_answerable']}) | holdout leaks: {report['holdout_leaks']} (must be 0)")

# generation tasks for Colab
tasks = []
for q in questions:
    got = retriever.retrieve(q["question"])
    context = "\n\n".join(f"[passage {i+1}] {g['text']}" for i, g in enumerate(got))
    tasks.append({
        "qid": q["qid"], "answerable": q["answerable"], "gold_label": q["gold_label"],
        "system": GENERATOR_SYSTEM,
        "user": GENERATOR_USER_TEMPLATE.format(context=context, question=q["question"]),
    })
(run_dir / "generation_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
(run_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2))
(run_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2))
print(f"wrote {len(tasks)} generation tasks -> {run_dir/'generation_tasks.jsonl'}")
print("next: run notebooks/generate_colab.ipynb on Colab T4, then scripts/score_run.py")
