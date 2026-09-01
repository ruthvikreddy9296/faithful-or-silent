"""Phase 2 semantic screen: adjudicate held-out questions as TRUE-unanswerable
vs answerable-from-related-evidence.

For each held-out question: top-5 passages retrieved by the base config are
scored with an NLI cross-encoder against the question's gold conclusion
(PubMedQA LONG_ANSWER). max P(entailment) over passages = evidence score.
Memory-minimal: no chunk list, no corpus in RAM.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from config import DATA, RESULTS

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # Apache 2.0

questions = [json.loads(l) for l in (DATA / "phase1_questions.jsonl").read_text().splitlines()]
heldout = [q for q in questions if not q["answerable"]]
pubmedqa = json.loads((DATA / "pubmedqa_pqal.json").read_text())

client = chromadb.PersistentClient(path=str(DATA / "chroma"))
coll = client.get_collection("all-MiniLM-L6-v2_c512o16")
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nli = CrossEncoder(NLI_MODEL)
# label order for this model: contradiction, entailment, neutral
ENT_IDX = 1

out = []
for i, q in enumerate(heldout):
    emb = embed.encode([q["question"]], normalize_embeddings=True)
    r = coll.query(query_embeddings=emb.tolist(), n_results=5)
    passages = r["documents"][0]
    gold = pubmedqa[q["qid"]]["LONG_ANSWER"]
    import numpy as np
    logits = nli.predict([(p, gold) for p in passages])          # premise, hypothesis
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ent = probs[:, ENT_IDX]
    out.append({"qid": q["qid"], "question": q["question"],
                "gold_label": q["gold_label"],
                "max_entailment": float(ent.max()),
                "mean_entailment": float(ent.mean()),
                "per_passage": [float(x) for x in ent]})
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{len(heldout)}", flush=True)

(RESULTS / "phase2_screen.json").write_text(json.dumps(out, indent=1))
import numpy as np
scores = np.array([o["max_entailment"] for o in out])
for thr in (0.5, 0.7, 0.9):
    n = int((scores > thr).sum())
    print(f"evidence score > {thr}: {n}/{len(scores)} held-out questions "
          f"({n/len(scores):.1%}) have related corpus evidence")
print("wrote", RESULTS / "phase2_screen.json")
