import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from utils import (FIG_DIR, load_label_embeddings,
                   reduce_tsne, SEMANTIC_GROUPS, GROUP_COLORS)

matplotlib.use("Agg")


def label_to_group(label: str) -> str:
    for group, members in SEMANTIC_GROUPS.items():
        if label in members:
            return group
    return "other"


def plot_labels(coords, labels: list[str], title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, (x, y) in zip(labels, coords):
        color = GROUP_COLORS.get(label_to_group(label), "#333333")
        ax.scatter(x, y, color=color, s=200, zorder=3)
        ann = ax.annotate(label, (x, y), fontsize=13, ha="center", va="bottom",
                          xytext=(0, 9), textcoords="offset points", zorder=5)
        ann.set_path_effects([
            pe.withStroke(linewidth=2, foreground="white")
        ])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=12, label=g)
        for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=handles, title="Semantic Groups",
              loc="upper right", fontsize=13, title_fontsize=14,
              frameon=True, framealpha=0.9)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "datalim")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run() -> tuple:
    print("\n[2A] Label embedding visualization...")
    label_emb, labels = load_label_embeddings()

    coords_tsne = reduce_tsne(label_emb, perplexity=5)
    plot_labels(coords_tsne, labels,
                "",
                os.path.join(FIG_DIR, "2a_labels_tsne.png"))

    return coords_tsne


if __name__ == "__main__":
    run()
