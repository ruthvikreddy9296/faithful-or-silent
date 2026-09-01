"""Unpack the Colab completions bundle into per-config result folders.

Usage: python scripts/place_completions.py ~/Downloads/completions_bundle.zip
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS

zip_path = Path(sys.argv[1]).expanduser()
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        name = Path(info.filename).name
        if not name.startswith("completions_") or not name.endswith(".jsonl"):
            continue
        stem = name[len("completions_"):-len(".jsonl")]
        # arm-suffixed outputs (e.g. p1_base__llama) land as completions_<arm>.jsonl
        config, _, arm = stem.partition("__")
        out_name = f"completions_{arm}.jsonl" if arm else "completions.jsonl"
        dest_dir = RESULTS / (config if config.startswith("phase") else f"phase1_{config}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / out_name).write_bytes(z.read(info))
        print(f"{name} -> {dest_dir/out_name}")
