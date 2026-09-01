"""Memory-minimal finisher for p1_pubmedbert: reuses the assembled index,
never builds the chunk list (dense retrieval doesn't need it)."""
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA, RESULTS, PipelineConfig
from retrieve import Retriever, retrieval_report
from prompts import GENERATOR_SYSTEM_V2, GENERATOR_USER_TEMPLATE

import chromadb
from sentence_transformers import SentenceTransformer

cfg = replace(PipelineConfig(), name="p1_pubmedbert",
              embedding_model="NeuML/pubmedbert-base-embeddings")
questions = [json.loads(l) for l in (DATA / "phase1_questions.jsonl").read_text().splitlines()]

client = chromadb.PersistentClient(path=str(DATA / "chroma"))
coll = client.get_collection("pubmedbert-base-embeddings_c512o16")
print("collection:", coll.count(), "vectors")
model = SentenceTransformer(cfg.embedding_model)
retriever = Retriever(cfg, coll, model, chunks=[])

report = retrieval_report(retriever, questions)
print(f"recall@{cfg.top_k}: {report['recall_at_k']:.3f} | leaks: {report['holdout_leaks']}")

run_dir = RESULTS / f"phase1_{cfg.name}"
run_dir.mkdir(exist_ok=True)
(run_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2))
tasks = []
for q in questions:
    got = retriever.retrieve(q["question"])
    context = "\n\n".join(f"[passage {i+1}] {g['text']}" for i, g in enumerate(got))
    tasks.append({"qid": q["qid"], "answerable": q["answerable"], "gold_label": q["gold_label"],
                  "config": cfg.name, "max_new_tokens": cfg.max_new_tokens,
                  "system": GENERATOR_SYSTEM_V2,
                  "user": GENERATOR_USER_TEMPLATE.format(context=context, question=q["question"])})
(run_dir / "generation_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
(run_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2))
shutil.copy(run_dir / "generation_tasks.jsonl",
            RESULTS / "phase1_tasks" / f"generation_tasks_{cfg.name}.jsonl")
print(f"emitted {len(tasks)} tasks -> bundle complete")
