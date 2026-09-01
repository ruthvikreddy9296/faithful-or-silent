"""Phase 1 statistical analysis: paired McNemar tests on false-answer outcomes.

Same 250 unanswerable questions run through every config, so config pairs are
compared with McNemar's exact test on discordant pairs (answered-vs-refused).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scipy.stats import binomtest
from config import RESULTS
from metrics import parse_response

CONFIGS = ["p1_base", "p1_bge", "p1_bm25", "p1_hybrid", "p1_k3", "p1_k8",
           "p1_rerank", "p1_chunk256", "p1_chunk1024", "p1_pubmedbert",
           "p1_simgate", "p1_reasonfirst"]

# qid -> {config: answered_bool} over unanswerable questions
answered = {}
for c in CONFIGS:
    fname = sys.argv[1] if len(sys.argv) > 1 else "completions.jsonl"
    rows = [json.loads(l) for l in
            (RESULTS / f"phase1_{c}" / fname).read_text().splitlines()]
    for r in rows:
        if not r["answerable"]:
            answered.setdefault(r["qid"], {})[c] = (
                parse_response(r["response_text"])[0] == "answer")

KEY_PAIRS = [("p1_bge", "p1_bm25"), ("p1_bge", "p1_base"), ("p1_base", "p1_bm25"),
             ("p1_base", "p1_reasonfirst"), ("p1_base", "p1_simgate"),
             ("p1_base", "p1_k8"), ("p1_bge", "p1_reasonfirst"),
             ("p1_base", "p1_hybrid"), ("p1_base", "p1_rerank"),
             ("p1_base", "p1_chunk256"), ("p1_base", "p1_chunk1024")]

out = []
for a, b in KEY_PAIRS:
    n01 = sum(1 for q in answered if not answered[q][a] and answered[q][b])
    n10 = sum(1 for q in answered if answered[q][a] and not answered[q][b])
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0
    fa = sum(answered[q][a] for q in answered) / len(answered)
    fb = sum(answered[q][b] for q in answered) / len(answered)
    out.append({"pair": f"{a} vs {b}", "fa_a": round(fa, 3), "fa_b": round(fb, 3),
                "n01": n01, "n10": n10, "discordant": n,
                "mcnemar_p": float(f"{p:.2e}")})
    print(f"{a:<15} FA={fa:.3f}  vs  {b:<15} FA={fb:.3f}   "
          f"discordant={n:>3}  p={p:.2e}{'  ***' if p < 0.001 else '  *' if p < 0.05 else ''}")

# Spearman correlation: recall vs false-answer rate across retrieval configs
from scipy.stats import spearmanr
arm = ("_" + sys.argv[1].replace("completions_","").replace(".jsonl","")) if len(sys.argv) > 1 else ""
front = json.loads((RESULTS / f"phase1_frontier{arm}.json").read_text())
pts = [(m["recall_at_k"], m["false_answer_rate"]) for m in front
       if m["recall_at_k"] is not None and m["config"] not in ("p1_reasonfirst", "p1_simgate")]
rho, pv = spearmanr([p[0] for p in pts], [p[1] for p in pts])
print(f"\nSpearman recall vs false-answer rate (retrieval configs, n={len(pts)}): "
      f"rho={rho:.3f}, p={pv:.4f}")

# Sensitivity: k=5-only, so recall is measured at a single fixed k (the full
# set mixes recall@3/@5/@8, which a reviewer could contest).
pts5 = [(m["recall_at_k"], m["false_answer_rate"]) for m in front
        if m["recall_at_k"] is not None
        and m["config"] not in ("p1_reasonfirst", "p1_simgate", "p1_k3", "p1_k8")]
rho5, pv5 = spearmanr([p[0] for p in pts5], [p[1] for p in pts5])
print(f"Spearman k=5-only sensitivity (n={len(pts5)}): rho={rho5:.3f}, p={pv5:.4f}")

json.dump({"mcnemar": out, "spearman_recall_fa": {"rho": rho, "p": pv, "n": len(pts)},
           "spearman_recall_fa_k5": {"rho": rho5, "p": pv5, "n": len(pts5)}},
          open(RESULTS / f"phase1_stats{arm}.json", "w"), indent=2)
print("wrote", RESULTS / f"phase1_stats{arm}.json")
