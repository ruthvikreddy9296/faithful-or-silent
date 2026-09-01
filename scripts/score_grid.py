"""Aggregate all Phase 1 config scores into the safe-failure frontier table.

Usage: python scripts/score_grid.py
Expects results/phase1_<name>/completions.jsonl for each config
(use scripts/place_completions.py to unpack the Colab bundle first).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS
from metrics import score_run, bootstrap_ci, parse_response

rows_out = []
for run_dir in sorted(RESULTS.glob("phase1_*")):
    comp = run_dir / (sys.argv[1] if len(sys.argv) > 1 else "completions.jsonl")
    if not comp.exists() or run_dir.name == "phase1_tasks":
        continue
    rows = [json.loads(l) for l in comp.read_text().splitlines()]
    m = score_run(rows)
    fa = [1.0 if parse_response(r["response_text"])[0] == "answer" else 0.0
          for r in rows if not r["answerable"]]
    m["false_answer_rate_ci95"] = bootstrap_ci(fa)
    rep_file = run_dir / "retrieval_report.json"
    m["recall_at_k"] = (json.loads(rep_file.read_text())["recall_at_k"]
                        if rep_file.exists() else None)
    m["config"] = run_dir.name.replace("phase1_", "")
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    rows_out.append(m)

if not rows_out:
    sys.exit("no phase1_*/completions.jsonl found yet")

rows_out.sort(key=lambda r: r["false_answer_rate"])
hdr = f'{"config":<14} {"recall@k":>8} {"acc(ans)":>9} {"falseRef":>9} {"falseAns":>9} {"FA CI95":>14} {"unparse":>8}'
print(hdr)
print("-" * len(hdr))
for m in rows_out:
    lo, hi = m["false_answer_rate_ci95"]
    print(f'{m["config"]:<14} {(f"{m["recall_at_k"]:.3f}" if m["recall_at_k"] is not None else "—"):>8} {m["accuracy_when_answering"]:>9.3f} '
          f'{m["false_refusal_rate"]:>9.3f} {m["false_answer_rate"]:>9.3f} '
          f'{f"[{lo:.2f},{hi:.2f}]":>14} {m["unparseable_rate"]:>8.3f}')

(RESULTS / (f"phase1_frontier_{sys.argv[1].replace(chr(39)+chr(39),chr(39)+chr(39)).replace('completions_','').replace('.jsonl','')}.json" if len(sys.argv) > 1 else "phase1_frontier.json")).write_text(json.dumps(rows_out, indent=2))
print(f"\nwrote {RESULTS/'phase1_frontier.json'}")
