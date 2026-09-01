# Faithful or Silent? Evaluating Retrieval Design and Abstention Behavior in Medical Retrieval-Augmented Generation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22238849.svg)](https://doi.org/10.5281/zenodo.22238849)

Code, results, and the certified unanswerable-question test artifact for the
paper *"Faithful or Silent? Evaluating Retrieval Design and Abstention
Behavior in Medical Retrieval-Augmented Generation"*
(Sai Ruthvik Reddy Avuku, [ORCID 0009-0006-6282-6213](https://orcid.org/0009-0006-6282-6213)).
Archived snapshot: [DOI 10.5281/zenodo.22238849](https://doi.org/10.5281/zenodo.22238849).
arXiv link: **coming with the preprint**.

## What this measures

Medical RAG systems are evaluated almost exclusively on accuracy. This study
measures how retrieval design changes **how a system fails** when the corpus
verifiably lacks the answer — does it abstain, or fabricate?

- **Held-out-context construction:** on PubMedQA, the gold evidence passages
  of 250 held-out questions are excluded from a 52,101-passage corpus, making
  those questions unanswerable *by construction*, with an automated leak
  check (zero leaks in every configuration).
- **The grid:** 13 retrieval configurations (embeddings, sparse/dense/hybrid,
  chunking, retrieval depth, reranking, similarity gating, prompt order,
  closed-book control) × 3 model families (Qwen2.5-7B-Instruct,
  Llama-3.1-8B-Instruct, GPT-5.6-terra) = 25,350 completions, all archived
  in `results/`.
- **Headline findings:** the safe-failure frontier is not flat (false-answer
  rate spans 12.4%–32.8% within one family across configs that barely differ
  in accuracy); the same reason-first prompt halves one family's false-answer
  rate, does nothing in a second, and doubles it in a third; a closed-book
  control refuses everything (zero parametric leakage); an NLI evidence
  screen shows hallucinations track topical similarity, not evidential
  support.

## The artifact

`artifact/unanswerable_medical_qa_v1.json` — 250 held-out questions, of which
**244 are certified evidence-free traps** by an NLI screen (DeBERTa-v3-base
cross-encoder, max entailment ≤ 0.5 over every retrieved passage), with
per-question screen scores and per-generator difficulty metadata. Reusable as
a drop-in abstention test set; the held-out-context method reproduces it on
any corpus with question–evidence pairing.

## Repository layout

- `src/` — pipeline: config, corpus construction, indexing, retrieval
  (dense/BM25/RRF/rerank/gate), prompts, metrics
- `scripts/` — grid runner, scoring, statistics (McNemar, Spearman,
  bootstrap), NLI screen, artifact packaging, figure and table generation
- `notebooks/` — Colab generation and corpus-embedding notebooks (resumable)
- `results/` — per-configuration outputs: raw completions for all three arms,
  retrieval reports, metrics, statistics, provenance
  (`arm_provenance.json`, `corpus_stats.json`)
- `artifact/` — the released test set

## Reproducing

```
pip install -r requirements.txt
python src/download_data.py            # PubMedQA (PQA-L + PQA-U)
python scripts/run_phase1.py           # corpus, indexes, retrieval, task files
# generation: notebooks/generate_colab.ipynb (GPU; arm-suffixed outputs)
python scripts/place_completions.py && python scripts/score_grid.py
python scripts/analyze_phase1.py       # paired tests + correlations
python scripts/make_tables.py          # regenerates every number in the paper
```

All randomness is seeded (seed 42); every run logs its full configuration.
Every number in the paper is generated from files in `results/` — none are
hand-typed.

## Licenses

- Code: [Apache License 2.0](LICENSE)
- Data (`artifact/`, `results/`): [CC BY 4.0](artifact/LICENSE)

## Citation

Citation entry will be added when the preprint is posted (arXiv, cs.CL).
