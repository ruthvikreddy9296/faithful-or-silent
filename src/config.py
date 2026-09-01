"""Central configuration for all experiments. Every run logs this."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

SEED = 42

# The exact refusal contract the generator is instructed to use.
REFUSAL_STRING = "I don't know based on the provided context."


@dataclass
class PipelineConfig:
    """One point in the ablation grid."""
    name: str = "base"
    # chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 16
    # embeddings (sentence-transformers model id)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_prefix: str = ""            # e.g. BGE retrieval instruction
    # retrieval
    retrieval: str = "dense"          # dense | bm25 | hybrid
    search_type: str = "similarity"   # similarity | mmr
    top_k: int = 5
    use_reranker: bool = False        # cross-encoder rerank of top-20 -> top_k
    reranker_model: str = "BAAI/bge-reranker-base"
    # abstention architecture knobs
    abstain_sim_threshold: float = 0.0  # >0: system-abstain if top-1 score below it;
                                        # -1.0: derive as 5th pct of answerable top-1 scores
    no_context: bool = False            # closed-book control (parametric-leakage probe)
    # generation (consumed by the Colab harness)
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 150         # verdict-first prompt => short outputs
    temperature: float = 0.0
    prompt_version: str = "v2"        # v1: reason-then-verdict (Phase 0); v2: verdict-first

    def index_fingerprint(self) -> str:
        """Configs sharing chunking+embedding share one index."""
        slug = self.embedding_model.split("/")[-1].replace(".", "-")
        return f"{slug}_c{self.chunk_size_tokens}o{self.chunk_overlap_tokens}"

    def as_dict(self):
        return asdict(self)


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Phase 0 pilot layout (kept for reproducibility of results/phase0_base)
PILOT = {
    "n_answerable": 100,
    "n_unanswerable": 50,
    "n_distractor_questions": 850,
}

# Phase 1: scaled split, built as a SUPERSET of the pilot split (same SEED
# shuffle; the pilot's 100 answerable / 50 held-out keep their roles) so
# Phase 0 results stay comparable. 400 answerable + 250 verifiably
# unanswerable gives the statistical power the false-answer-rate claims need.
PHASE1 = {
    "n_answerable": 400,
    "n_unanswerable": 250,
    "n_distractor_unlabeled": 15_000,  # PQA-U papers padding the corpus
}
