import os
from colorsys import rgb_to_hls, hls_to_rgb

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from utils import (FIG_DIR, load_column_embeddings, load_unified_df,
                   reduce_tsne, SEMANTIC_GROUPS, GROUP_COLORS)


def _semantic_color_map(unique_labels: list[str]) -> dict[str, tuple]:
    """Assign each label a shade of its semantic group color.

    Labels within the same group share the same hue but differ in lightness,
    giving a clear visual hierarchy between groups while keeping intra-group
    labels distinguishable.
    """
    def make_shades(hex_color: str, n: int) -> list[tuple]:
        r, g, b = mcolors.to_rgb(hex_color)
        h, _, s = rgb_to_hls(r, g, b)
        if n == 1:
            return [(r, g, b)]
        lightnesses = [0.28 + i * 0.42 / (n - 1) for i in range(n)]
        return [hls_to_rgb(h, li, min(s, 0.85)) for li in lightnesses]

    color_map: dict[str, tuple] = {}
    for group, members in SEMANTIC_GROUPS.items():
        present = [l for l in members if l in unique_labels]
        if not present:
            continue
        for label, color in zip(present, make_shades(GROUP_COLORS[group], len(present))):
            color_map[label] = color

    # labels not in any semantic group → grey shades
    unlabeled = sorted(l for l in unique_labels if l not in color_map)
    for label, color in zip(unlabeled, make_shades("#95a5a6", max(len(unlabeled), 1))):
        color_map[label] = color

    return color_map


def _sorted_labels(unique_labels: list[str]) -> list[str]:
    """Return labels sorted by semantic group order, then alphabetically within group."""
    group_index = {l: i for i, (_, members) in enumerate(SEMANTIC_GROUPS.items())
                   for l in members}
    return sorted(unique_labels,
                  key=lambda l: (group_index.get(l, len(SEMANTIC_GROUPS)), l))


def plot_columns(coords: np.ndarray, unified_df: pd.DataFrame,
                 title: str, path: str) -> None:
    gt_labels     = unified_df["ground_truth"].values
    all_wrong     = unified_df["all_wrong"].values
    unique_labels = sorted(set(gt_labels))

    label_to_color = _semantic_color_map(unique_labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    normal_mask = ~all_wrong
    colors = [label_to_color[l] for l in gt_labels[normal_mask]]
    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
               c=colors, s=46, alpha=0.78, zorder=2)

    hard_idx = np.where(all_wrong)[0]
    hard_colors = [label_to_color[gt_labels[i]] for i in hard_idx]
    ax.scatter(coords[hard_idx, 0], coords[hard_idx, 1],
               c=hard_colors, marker="X", s=125, alpha=0.9, zorder=4,
               edgecolors="black", linewidths=1.2)

    sorted_lbls = _sorted_labels(unique_labels)
    gt_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=label_to_color[l], markersize=12, label=l)
        for l in sorted_lbls
    ]
    hard_handle = plt.Line2D([0], [0], marker="X", color="w",
                              markerfacecolor="white", markersize=12,
                              markeredgecolor="black", markeredgewidth=1.2,
                              label="Universally\nMissed")
    ax.legend(handles=gt_handles + [hard_handle], title="Ground Truth Labels",
              loc="upper right", fontsize=10, title_fontsize=11, ncol=2,
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


def run() -> tuple[np.ndarray, np.ndarray]:
    print("\n[2B] Column embedding visualization...")
    col_emb    = load_column_embeddings()
    unified_df = load_unified_df()

    coords_tsne = reduce_tsne(col_emb, perplexity=30, init="random")
    plot_columns(coords_tsne, unified_df,
                 "",
                 os.path.join(FIG_DIR, "2b_columns_tsne.png"))

    return coords_tsne


if __name__ == "__main__":
    run()
