"""Package the released artifact: certified unanswerable medical QA test set v1.

Assembles per-question: text + gold label (PubMedQA, MIT), construction
provenance, NLI screen scores + certification, and difficulty metadata from the
two completed generator arms. Generator-independent core; difficulty columns
are versioned extras.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA, RESULTS, ROOT, SEED
from metrics import parse_response

ART = ROOT / "artifact"
ART.mkdir(exist_ok=True)

questions = {json.loads(l)["qid"]: json.loads(l)
             for l in (DATA / "phase1_questions.jsonl").read_text().splitlines()}
screen = {o["qid"]: o for o in json.loads((RESULTS / "phase2_screen.json").read_text())}
pubmedqa = json.loads((DATA / "pubmedqa_pqal.json").read_text())

CONFIGS = ["p1_base", "p1_bge", "p1_bm25", "p1_hybrid", "p1_k3", "p1_k8", "p1_rerank",
           "p1_chunk256", "p1_chunk1024", "p1_pubmedbert", "p1_simgate", "p1_reasonfirst"]

def fooled_counts(fname):
    counts = {}
    for c in CONFIGS:
        f = RESULTS / f"phase1_{c}" / fname
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if not r["answerable"]:
                counts.setdefault(r["qid"], 0)
                counts[r["qid"]] += parse_response(r["response_text"])[0] == "answer"
    return counts

qwen = fooled_counts("completions.jsonl")
terra = fooled_counts("completions_openai.jsonl")

records = []
for qid, q in questions.items():
    if q["answerable"]:
        continue
    s = screen[qid]
    records.append({
        "qid": qid,
        "question": q["question"],
        "gold_label": q["gold_label"],
        "gold_conclusion": pubmedqa[qid]["LONG_ANSWER"],
        "certified_true_trap": s["max_entailment"] <= 0.5,
        "screen": {"max_entailment": s["max_entailment"],
                   "mean_entailment": s["mean_entailment"],
                   "nli_model": "cross-encoder/nli-deberta-v3-base"},
        "difficulty": {"qwen2.5-7b_configs_fooled_of_12": qwen.get(qid, 0),
                       "gpt-5.6-terra_configs_fooled_of_12": terra.get(qid, 0)},
    })

meta = {
    "name": "certified-unanswerable-medical-qa",
    "version": "1.0",
    "n_questions": len(records),
    "n_certified": sum(r["certified_true_trap"] for r in records),
    "construction": ("Held-out-context method on PubMedQA PQA-L (seed 42 split): each "
                     "question's gold evidence passages are verifiably EXCLUDED from the "
                     "52,101-passage retrieval corpus (built from 400 answerable questions' "
                     "passages + 350 PQA-L + 15,000 PQA-U distractor papers). An NLI screen "
                     "then certifies that no RELATED indexed passage entails the gold "
                     "conclusion (max entailment <= 0.5)."),
    "license": "MIT (derived from PubMedQA, MIT; Jin et al., 2019)",
    "intended_use": ("Measuring false-answer/abstention behavior of medical RAG systems "
                     "under retrieval failure. Pair with the companion corpus-build script "
                     "to reproduce the exact corpus."),
}
out = {"metadata": meta, "questions": records}
(ART / "unanswerable_medical_qa_v1.json").write_text(json.dumps(out, indent=1))
print(f"artifact: {meta['n_questions']} questions, {meta['n_certified']} certified true traps")
print("wrote", ART / "unanswerable_medical_qa_v1.json")
