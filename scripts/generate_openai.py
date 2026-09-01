"""Commercial reference arm: run grid task files through a version-pinned OpenAI model.

This is the paper's SECONDARY arm — primary claims rest on open-weight models.
Runs locally (the API does the compute). Requires OPENAI_API_KEY in the env;
the key is never written to disk or logs.

Usage:
  export OPENAI_API_KEY=sk-...
  .venv/bin/python scripts/generate_openai.py --model <exact-model-string> \
      [--tasks results/phase1_tasks] [--limit N]

Pin the exact model string (check platform.openai.com/docs/models) — the paper
must cite it verbatim plus the access date.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS

API_URL = "https://api.openai.com/v1/chat/completions"


def call_openai(model, system, user, max_tokens, key, retries=5):
    # NOTE: GPT-5.6 models reject temperature overrides (default=1 only) —
    # documented in the paper's methodology as a generator-arm difference.
    body = json.dumps({
        "model": model, "max_completion_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
                # resolved model string from the API (may carry a dated snapshot
                # if the provider exposes one) — recorded for reproducibility
                return resp["choices"][0]["message"]["content"], resp.get("model", model)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            # network blips (timeouts, resets, DNS) — back off and retry
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"network error after {retries} attempts: {e}")
    raise RuntimeError("retries exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="exact OpenAI model string to pin (e.g. a GPT-5.6 tier)")
    ap.add_argument("--tasks", default=str(RESULTS / "phase1_tasks"))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap tasks per file")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set. export it first (never commit it).")

    task_files = sorted(Path(args.tasks).glob("generation_tasks*.jsonl"))
    if not task_files:
        sys.exit(f"no task files in {args.tasks}")

    for tf in task_files:
        config = tf.stem.replace("generation_tasks_", "")
        run_dir = RESULTS / f"phase1_{config}"
        run_dir.mkdir(exist_ok=True)
        out = run_dir / "completions_openai.jsonl"
        done = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        tasks = [json.loads(l) for l in tf.read_text().splitlines()]
        if args.limit:
            tasks = tasks[: args.limit]
        if len(done) >= len(tasks):
            print(f"{config}: already complete ({len(done)})")
            continue
        print(f"== {config}: {len(done)}/{len(tasks)} done, resuming")
        for i, t in enumerate(tasks[len(done):], start=len(done)):
            if t.get("pre_response"):
                text, served = t["pre_response"], None  # similarity gate: no API call
            else:
                text, served = call_openai(args.model, t["system"], t["user"],
                                           t.get("max_new_tokens", 150), key)
                time.sleep(0.2)  # gentle pacing
            done.append({"qid": t["qid"], "answerable": t["answerable"],
                         "gold_label": t["gold_label"], "config": t.get("config", config),
                         "response_text": text, "generator": args.model,
                         "served_model": served})
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                out.write_text("\n".join(json.dumps(d) for d in done))
                print(f"  {config}: {i+1}/{len(tasks)}")
        out.write_text("\n".join(json.dumps(d) for d in done))
    print("openai arm complete")


if __name__ == "__main__":
    main()
