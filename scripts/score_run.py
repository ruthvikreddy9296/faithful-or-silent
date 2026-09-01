"""Score a completed generation run: python scripts/score_run.py results/phase0_base"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import score_run, bootstrap_ci, parse_response

run_dir = Path(sys.argv[1])
rows = [json.loads(l) for l in (run_dir / "completions.jsonl").read_text().splitlines()]
m = score_run(rows)

fa = [1.0 if parse_response(r["response_text"])[0] == "answer" else 0.0
      for r in rows if not r["answerable"]]
m["false_answer_rate_ci95"] = bootstrap_ci(fa)

(run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
