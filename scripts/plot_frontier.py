"""Safe-failure frontier figure (draft). Colors: validated 3-slot categorical
palette (all-pairs safe for scatter); text in ink tokens, not series colors."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

front = {m["config"]: m for m in json.loads((RESULTS / "phase1_frontier.json").read_text())}
LABELS = {"p1_base": "base", "p1_bge": "BGE", "p1_bm25": "BM25", "p1_hybrid": "hybrid",
          "p1_k3": "k=3", "p1_k8": "k=8", "p1_rerank": "rerank", "p1_chunk256": "chunk 256",
          "p1_chunk1024": "chunk 1024", "p1_pubmedbert": "PubMedBERT",
          "p1_simgate": "similarity gate", "p1_reasonfirst": "reason-first"}
FAMILY = {c: ("gate" if c == "p1_simgate" else
              "prompt" if c == "p1_reasonfirst" else "retrieval") for c in LABELS}
COLORS = {"retrieval": BLUE, "gate": ORANGE, "prompt": AQUA}
NAMES = {"retrieval": "Retrieval variants", "gate": "System gate", "prompt": "Reason-first prompt"}

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

seen = set()
for c, lab in LABELS.items():
    m = front[c]
    fam = FAMILY[c]
    ax.scatter(m["false_answer_rate"], m["accuracy_when_answering"],
               s=70, color=COLORS[fam], zorder=3,
               label=NAMES[fam] if fam not in seen else None)
    seen.add(fam)
    # manual offsets to avoid collisions in the mid-cluster
    OFF = {"k=3": (-0.012, 0.008), "chunk 256": (0.004, -0.011),
           "chunk 1024": (0.002, 0.014), "PubMedBERT": (0.006, -0.004),
           "base": (-0.020, 0.007), "BGE": (-0.006, 0.008), "k=8": (0.005, 0.006),
           "rerank": (0.005, -0.008)}
    dx, dy = OFF.get(lab, (0.006, 0.004))
    ax.annotate(lab, (m["false_answer_rate"] + dx, m["accuracy_when_answering"] + dy),
                fontsize=8, color=INK2)

ax.set_xlabel("False-answer rate on unanswerable questions (lower = safer)",
              fontsize=10, color=INK)
ax.set_ylabel("Accuracy when answering (answerable questions)", fontsize=10, color=INK)
ax.set_title("The safe-failure frontier: 12 RAG configurations, Qwen2.5-7B, n=650",
             fontsize=11, color=INK, pad=12)
ax.grid(True, linewidth=0.4, color="#e6e5e1", zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c2")
ax.tick_params(colors=INK2, labelsize=9)
leg = ax.legend(loc="lower right", fontsize=9, frameon=False)
for t in leg.get_texts():
    t.set_color(INK)

fig.tight_layout()
out = RESULTS / "frontier_draft.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
