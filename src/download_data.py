"""Download open datasets: PubMedQA (MIT) and MedMCQA (Apache 2.0).

PubMedQA expert-labeled set (PQA-L, 1000 questions) comes straight from the
official GitHub repo as JSON. MedMCQA validation split comes from the Hugging
Face parquet endpoint (no `datasets` library needed).
"""
import json
import urllib.request
from config import DATA

PUBMEDQA_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)
MEDMCQA_PARQUET_API = (
    "https://huggingface.co/api/datasets/openlifescienceai/medmcqa/parquet/default/validation"
)


def download_pubmedqa() -> dict:
    out = DATA / "pubmedqa_pqal.json"
    if not out.exists():
        print("downloading PubMedQA PQA-L ...")
        urllib.request.urlretrieve(PUBMEDQA_URL, out)
    data = json.loads(out.read_text())
    print(f"PubMedQA: {len(data)} expert-labeled questions")
    return data


def download_medmcqa():
    out = DATA / "medmcqa_validation.parquet"
    if not out.exists():
        print("downloading MedMCQA validation split ...")
        with urllib.request.urlopen(MEDMCQA_PARQUET_API, timeout=30) as r:
            files = json.loads(r.read())
        urllib.request.urlretrieve(files[0], out)
    import pandas as pd
    df = pd.read_parquet(out)
    print(f"MedMCQA validation: {len(df)} questions")
    return df


PQAU_PARQUET_API = (
    "https://huggingface.co/api/datasets/qiaojin/PubMedQA/parquet/pqa_unlabeled/train"
)


def download_pubmedqa_unlabeled():
    """PQA-U: 61k real PubMed questions+contexts — Phase 1 distractor pool."""
    import pandas as pd
    out = DATA / "pubmedqa_pqau.parquet"
    if not out.exists():
        print("downloading PubMedQA PQA-U (distractor pool) ...")
        with urllib.request.urlopen(PQAU_PARQUET_API, timeout=30) as r:
            files = json.loads(r.read())
        frames = []
        for i, url in enumerate(files):
            part = DATA / f"pqau_part{i}.parquet"
            if not part.exists():
                urllib.request.urlretrieve(url, part)
            frames.append(pd.read_parquet(part))
        pd.concat(frames, ignore_index=True).to_parquet(out)
        for i in range(len(files)):
            (DATA / f"pqau_part{i}.parquet").unlink(missing_ok=True)
    df = pd.read_parquet(out)
    print(f"PQA-U: {len(df)} questions")
    return df


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    download_pubmedqa()
    download_medmcqa()
    download_pubmedqa_unlabeled()
