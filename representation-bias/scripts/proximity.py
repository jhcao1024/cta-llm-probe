import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from utils import load_column_embeddings, load_unified_df

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(EXP_DIR)
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts", "representation-bias")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

METHODS   = ["sc_zs", "sc_fs", "sc_rag", "mc_zs", "mc_fs", "mc_rag", "mc_cot"]
PRED_COLS = [f"{m}_pred" for m in METHODS]


def _leave_one_out_centroid(
    embeddings: np.ndarray,
    label_mask: np.ndarray,
    current_idx: int,
) -> np.ndarray | None:
    """Compute a label centroid excluding the current point when applicable."""
    candidate_idx = np.where(label_mask)[0]
    candidate_idx = candidate_idx[candidate_idx != current_idx]
    if len(candidate_idx) == 0:
        return None
    return embeddings[candidate_idx].mean(axis=0)


def run() -> pd.DataFrame:
    print("\n[2C] Hard case cluster proximity analysis...")

    unified_df = load_unified_df()
    col_emb    = load_column_embeddings()
    gt_labels  = unified_df["ground_truth"].values
    all_wrong  = unified_df["all_wrong"].values

    results = []
    for idx in np.where(all_wrong)[0]:
        row         = unified_df.iloc[idx]
        gt          = row["ground_truth"]
        wrong_label = pd.Series([row[c] for c in PRED_COLS]).value_counts().index[0]

        pos = col_emb[idx]

        gt_centroid = _leave_one_out_centroid(col_emb, gt_labels == gt, idx)
        wrong_centroid = _leave_one_out_centroid(col_emb, gt_labels == wrong_label, idx)

        dist_gt = np.linalg.norm(pos - gt_centroid) if gt_centroid is not None else np.nan
        dist_wrong = (
            np.linalg.norm(pos - wrong_centroid) if wrong_centroid is not None else np.nan
        )
        closer_to_wrong = (
            dist_wrong < dist_gt if not (np.isnan(dist_gt) or np.isnan(dist_wrong)) else None
        )

        results.append({
            "table_id":         row["table_id"],
            "col_idx":          row["col_idx"],
            "ground_truth":    gt,
            "wrong_label":     wrong_label,
            "dist_gt":         round(dist_gt, 4),
            "dist_wrong":      round(dist_wrong, 4),
            "closer_to_wrong": closer_to_wrong,
            "excluded":        closer_to_wrong is None,
        })

    result_df      = pd.DataFrame(results)
    excluded       = result_df["excluded"]
    valid          = ~excluded
    n_excluded     = int(excluded.sum())
    n_closer_wrong = int(result_df.loc[valid, "closer_to_wrong"].astype(bool).sum())
    n_total        = int(valid.sum())

    if n_excluded > 0:
        print(f"\n  Excluded (missing gt/wrong centroid after leave-one-out): {n_excluded} case(s)")
        print(
            result_df.loc[excluded, ["table_id", "col_idx", "ground_truth", "wrong_label"]]
            .to_string(index=False)
        )

    print(f"\n  Analysable hard cases: {n_total}")
    print(f"  Closer to wrong label cluster : {n_closer_wrong}/{n_total} "
          f"({n_closer_wrong / n_total * 100:.1f}%)")
    print("\n  Per-case breakdown:")
    print(
        result_df.loc[
            valid,
            ["table_id", "col_idx", "ground_truth", "wrong_label",
             "dist_gt", "dist_wrong", "closer_to_wrong"],
        ].to_string(index=False)
    )

    csv_path = os.path.join(ARTIFACT_DIR, "2c_hard_case_cluster_proximity.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return result_df


if __name__ == "__main__":
    run()
