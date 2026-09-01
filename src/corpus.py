"""Build the pilot corpus via the held-out-context method.

PubMedQA pairs each question with its gold evidence passages. We:
  - sample N_ANSWERABLE questions -> their passages go INTO the corpus
  - sample N_UNANSWERABLE questions -> their passages are EXCLUDED (held out)
  - all remaining questions' passages are added as distractors

A held-out question is therefore *verifiably* unanswerable w.r.t. this corpus:
its evidence provably is not indexed. Answerability is corpus-relative and
here it is controlled by construction, not by annotator judgment.
"""
import json
import random
from config import DATA, SEED, PILOT


def build_split(pubmedqa: dict):
    rng = random.Random(SEED)
    pmids = sorted(pubmedqa.keys())
    rng.shuffle(pmids)

    n_a, n_u = PILOT["n_answerable"], PILOT["n_unanswerable"]
    answerable = pmids[:n_a]
    unanswerable = pmids[n_a : n_a + n_u]
    distractors = pmids[n_a + n_u :]

    corpus_docs = []  # (doc_id, text, source_pmid)
    for pmid in answerable + distractors:
        for i, ctx in enumerate(pubmedqa[pmid]["CONTEXTS"]):
            corpus_docs.append(
                {"doc_id": f"{pmid}_{i}", "text": ctx, "source_pmid": pmid}
            )

    def qrec(pmid, flag):
        e = pubmedqa[pmid]
        return {
            "qid": pmid,
            "question": e["QUESTION"],
            "gold_label": e["final_decision"],  # yes | no | maybe
            "answerable": flag,
        }

    questions = [qrec(p, True) for p in answerable] + [
        qrec(p, False) for p in unanswerable
    ]

    (DATA / "pilot_corpus.jsonl").write_text(
        "\n".join(json.dumps(d) for d in corpus_docs)
    )
    (DATA / "pilot_questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions)
    )
    print(
        f"corpus: {len(corpus_docs)} passages from {len(answerable) + len(distractors)} "
        f"papers | questions: {n_a} answerable + {n_u} held-out unanswerable"
    )
    return corpus_docs, questions


def phase1_split_ids(pubmedqa: dict):
    """Nested-superset split: the pilot's 100 answerable (pmids[0:100]) and 50
    held-out (pmids[100:150]) keep their roles; new answerable and held-out
    blocks extend them. Same SEED shuffle as the pilot."""
    from config import PHASE1
    rng = random.Random(SEED)
    pmids = sorted(pubmedqa.keys())
    rng.shuffle(pmids)
    pa, pu = PILOT["n_answerable"], PILOT["n_unanswerable"]  # 100, 50
    na, nu = PHASE1["n_answerable"], PHASE1["n_unanswerable"]  # 400, 250
    answerable = pmids[:pa] + pmids[pa + pu : pa + pu + (na - pa)]
    heldout = pmids[pa : pa + pu] + pmids[pa + pu + (na - pa) : pa + pu + (na - pa) + (nu - pu)]
    distractors = pmids[pa + pu + (na - pa) + (nu - pu) :]
    assert len(answerable) == na and len(heldout) == nu
    assert not set(answerable) & set(heldout)
    return answerable, heldout, distractors


def build_phase1_corpus(pubmedqa: dict, pqau_df):
    """Scaled split + large PQA-U distractor pool. Held-out questions' papers
    are excluded from the corpus (leak check enforced downstream; PQA-U rows
    sharing a held-out pubid are dropped defensively)."""
    from config import PHASE1
    answerable, heldout_list, pqal_distractors = phase1_split_ids(pubmedqa)
    heldout = set(heldout_list)

    corpus_docs = []
    for pmid in answerable + pqal_distractors:
        for i, ctx in enumerate(pubmedqa[pmid]["CONTEXTS"]):
            corpus_docs.append({"doc_id": f"{pmid}_{i}", "text": ctx, "source_pmid": pmid})

    n_added = 0
    for _, row in pqau_df.iterrows():
        if n_added >= PHASE1["n_distractor_unlabeled"]:
            break
        pmid = str(row["pubid"])
        if pmid in heldout or pmid in pubmedqa:
            continue
        for i, ctx in enumerate(row["context"]["contexts"]):
            corpus_docs.append({"doc_id": f"u{pmid}_{i}", "text": str(ctx), "source_pmid": f"u{pmid}"})
        n_added += 1

    (DATA / "phase1_corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus_docs))

    def qrec(pmid, flag):
        e = pubmedqa[pmid]
        return {"qid": pmid, "question": e["QUESTION"],
                "gold_label": e["final_decision"], "answerable": flag}

    questions = [qrec(p, True) for p in answerable] + [qrec(p, False) for p in heldout_list]
    (DATA / "phase1_questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions))
    print(f"phase1 corpus: {len(corpus_docs)} passages "
          f"({n_added} PQA-U distractor papers + {len(answerable) + len(pqal_distractors)} PQA-L) | "
          f"questions: {len(answerable)} answerable + {len(heldout_list)} held-out")
    return corpus_docs, questions


if __name__ == "__main__":
    pubmedqa = json.loads((DATA / "pubmedqa_pqal.json").read_text())
    build_split(pubmedqa)
