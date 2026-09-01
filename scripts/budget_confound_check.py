"""Bound the reason-first token-budget confound (150 -> 400 max_new_tokens).

Claim to verify: for the greedy open-weight arms, raising the generation budget
of the VERDICT-FIRST prompt cannot change any headline metric, because
(a) greedy decoding means a larger budget only appends tokens to the identical
    emitted prefix;
(b) every verdict-first response already contains its parse target (verdict
    line or refusal string) in that prefix (unparseable == 0); and
(c) an appended continuation could flip an outcome only by emitting the exact
    refusal string AFTER a stated verdict -- a pattern checked here across the
    entire corpus.

Writes results/budget_confound.json. Terra is excluded from (a)-(b): the API
enforces default sampling (non-deterministic), but the reason-first effect is
null there, so there is nothing to deconfound.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS, REFUSAL_STRING
from metrics import parse_response

import re

CONFIGS = ["p1_base", "p1_bge", "p1_bm25", "p1_hybrid", "p1_k3", "p1_k8",
           "p1_rerank", "p1_chunk256", "p1_chunk1024", "p1_pubmedbert",
           "p1_simgate", "p1_reasonfirst", "p1_noctx"]
ARMS = {"qwen": "", "terra": "_openai", "llama": "_llama"}

out = {"note": __doc__.strip().splitlines()[0],
       "verdict_first_base": {}, "verdict_then_refusal_anywhere": {}}

# (b): verdict-first base responses -- unparseable count and prefix stability.
for arm in ("qwen", "llama"):
    rows = [json.loads(l) for l in
            (RESULTS / "phase1_p1_base" / f"completions{ARMS[arm]}.jsonl")
            .read_text().splitlines()]
    unparseable = sum(1 for r in rows
                      if parse_response(r["response_text"])[0] == "unparseable")
    # prefix stability: parse class fixed within the first 400 characters
    unstable = sum(1 for r in rows
                   if parse_response(r["response_text"])[0]
                   != parse_response(r["response_text"][:400])[0])
    out["verdict_first_base"][arm] = {
        "n": len(rows), "unparseable": unparseable,
        "parse_class_not_fixed_in_first_400_chars": unstable,
        "max_response_chars": max(len(r["response_text"]) for r in rows),
    }

# (c): anywhere in the corpus, does a response state a verdict and THEN emit
# the refusal string? (The only pattern by which appended tokens could flip
# an "answer" outcome to "refusal".)
ref = REFUSAL_STRING.lower().rstrip(".")
flips = 0
total = 0
for arm, suf in ARMS.items():
    for c in CONFIGS:
        p = RESULTS / f"phase1_{c}" / f"completions{suf}.jsonl"
        for line in p.read_text().splitlines():
            total += 1
            t = json.loads(line)["response_text"]
            m = re.search(r"VERDICT:\s*(yes|no|maybe)", t, re.IGNORECASE)
            if m and ref in t.lower()[m.end():]:
                flips += 1
out["verdict_then_refusal_anywhere"] = {"responses_checked": total,
                                        "verdict_then_refusal": flips}

json.dump(out, open(RESULTS / "budget_confound.json", "w"), indent=2)
print(json.dumps(out, indent=1))
