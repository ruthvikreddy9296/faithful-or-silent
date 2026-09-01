"""Three-family safe-failure frontier (Figure 1 candidate). Shared axes,
selective direct labels, validated 3-slot palette, ink-token text."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 (IEEE-friendly)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RESULTS

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

ARMS = [("phase1_frontier.json", "Qwen2.5-7B (open)"),
        ("phase1_frontier_openai.json", "GPT-5.6 Terra (commercial)"),
        ("phase1_frontier_llama.json", "Llama-3.1-8B (open)")]
FAMILY = lambda c: ("gate" if c == "p1_simgate" else
                    "prompt" if c == "p1_reasonfirst" else "retrieval")
COLORS = {"retrieval": BLUE, "gate": ORANGE, "prompt": AQUA}
# distinct marker shapes so panels stay legible in grayscale / for CVD readers
MARKERS = {"retrieval": "o", "gate": "s", "prompt": "^"}
NAMES = {"retrieval": "Retrieval variants", "gate": "Similarity gate", "prompt": "Reason-first prompt"}
# selective labels only — full values live in the tables
LABEL = {"p1_bm25": "BM25", "p1_bge": "BGE", "p1_k8": "k=8", "p1_base": "base",
         "p1_simgate": "gate", "p1_reasonfirst": "reason-first"}

# Label offsets in POINTS (resolution-independent), (dx, dy, ha).
# Defaults suit spread-out panels; overrides de-collide tight clusters.
DEFAULT_OFF = (5, 4, "left")
OFFSETS = {
    # Llama panel: gate/base/k=8 sit in one tight cluster under BGE
    ("phase1_frontier_llama.json", "p1_bge"): (2, 7, "left"),
    ("phase1_frontier_llama.json", "p1_simgate"): (-6, 3, "right"),
    ("phase1_frontier_llama.json", "p1_base"): (0, -16, "center"),
    ("phase1_frontier_llama.json", "p1_k8"): (7, -3, "left"),
    # Terra panel: bge and k=8 adjacent; reason-first label runs into BGE's
    ("phase1_frontier_openai.json", "p1_bge"): (2, 7, "left"),
    ("phase1_frontier_openai.json", "p1_k8"): (7, -3, "left"),
    ("phase1_frontier_openai.json", "p1_reasonfirst"): (-6, 2, "right"),
}

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=300, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, (fname, title) in zip(axes, ARMS):
    ax.set_facecolor(SURFACE)
    front = {m["config"]: m for m in json.loads((RESULTS / fname).read_text())}
    seen = set()
    for c, m in front.items():
        if c == "p1_noctx":
            continue
        fam = FAMILY(c)
        ax.scatter(m["false_answer_rate"], m["accuracy_when_answering"],
                   s=55, color=COLORS[fam], marker=MARKERS[fam], zorder=3,
                   label=NAMES[fam] if fam not in seen else None)
        seen.add(fam)
        if c in LABEL:
            dx, dy, ha = OFFSETS.get((fname, c), DEFAULT_OFF)
            ax.annotate(LABEL[c],
                        (m["false_answer_rate"], m["accuracy_when_answering"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, fontsize=7.5, color=INK2)
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(True, linewidth=0.4, color="#e6e5e1", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8.5)

axes[0].set_ylabel("Accuracy when answering", fontsize=9.5, color=INK)
axes[1].set_xlabel("False-answer rate on unanswerable questions (lower = safer)",
                   fontsize=9.5, color=INK)
leg = axes[2].legend(loc="lower right", fontsize=8, frameon=False)
for t in leg.get_texts():
    t.set_color(INK)
fig.suptitle("The safe-failure frontier across three model families "
             "(13 configs, n=650 per config)", fontsize=11, color=INK, y=1.0)
fig.tight_layout()
out = RESULTS / "frontier_3arm.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
out_pdf = RESULTS / "frontier_3arm.pdf"
fig.savefig(out_pdf, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out_pdf, "(vector — copy to paper/figures/fig1_frontier.pdf)")
