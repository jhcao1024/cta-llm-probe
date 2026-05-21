import json
import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EXP_DIR)
FAILURE_PATTERNS_ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "failure-patterns")
RETRIEVAL_DIAGNOSTICS_ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "retrieval-diagnostics")
OUT_DIR = os.path.join(ROOT, "artifacts", "binary-probing")
os.makedirs(OUT_DIR, exist_ok=True)


def get_most_freq_wrong(row, pred_cols, pred_priority):
    wrong_by_col = {
        c: row[c]
        for c in pred_cols
        if row[c] != row["ground_truth"]
    }
    wrong_preds = list(wrong_by_col.values())
    if not wrong_preds:
        return None

    counts = pd.Series(wrong_preds).value_counts()
    top_freq = counts.iloc[0]
    tied_labels = set(counts[counts == top_freq].index)
    if len(tied_labels) == 1:
        return counts.index[0]

    for pred_col in pred_priority:
        pred = wrong_by_col.get(pred_col)
        if pred in tied_labels:
            return pred

    return counts.index[0]


def build_query_table_string(group):
    lines = ["Classify these columns:\n"]
    for _, row in group.sort_values("col_idx").iterrows():
        lines.append(f"Column {int(row['col_idx'])}: {str(row['column_values'])[:500]}")
    return "\n".join(lines)


def build_support_label_lines(group, target_col_idx):
    lines = []
    for _, row in group.sort_values("col_idx").iterrows():
        col_idx = int(row["col_idx"])
        if col_idx == target_col_idx:
            continue
        lines.append(f"Column {col_idx} -> {row['ground_truth']}")
    return lines


def main():
    print("=" * 60)
    print("BINARY-PROBING: PREPARING PROBING DATA")
    print("=" * 60)

    unified_df = pd.read_csv(os.path.join(FAILURE_PATTERNS_ARTIFACT_DIR, "unified_df.csv"))
    with open(os.path.join(RETRIEVAL_DIAGNOSTICS_ARTIFACT_DIR, "sc_retrieval_artifacts.json")) as f:
        sc_artifacts = json.load(f)
    with open(os.path.join(RETRIEVAL_DIAGNOSTICS_ARTIFACT_DIR, "mc_retrieval_artifacts.json")) as f:
        mc_artifacts = json.load(f)

    pred_cols = [c for c in unified_df.columns if c.endswith("_pred")]
    pred_priority = [
        "mc_rag_pred",
        "mc_fs_pred",
        "mc_cot_pred",
        "sc_rag_pred",
        "sc_fs_pred",
        "mc_zs_pred",
        "sc_zs_pred",
    ]
    unified_df["hard_distractor"] = unified_df.apply(
        get_most_freq_wrong,
        axis=1,
        pred_cols=pred_cols,
        pred_priority=pred_priority,
    )

    error_mask = (unified_df[pred_cols].ne(unified_df["ground_truth"], axis=0)).any(axis=1)
    targets = unified_df[error_mask].copy()

    sc_map = {(item["table_id"], int(item["col_idx"])): item for item in sc_artifacts}
    mc_map = {item["table_id"]: item for item in mc_artifacts}
    table_map = {
        tid: group.copy()
        for tid, group in unified_df.groupby("table_id", sort=False)
    }

    cases = []
    for _, row in targets.iterrows():
        tid = row["table_id"]
        col_idx = int(row["col_idx"])
        sc_item = sc_map[(tid, col_idx)]
        mc_item = mc_map[tid]
        table_group = table_map[tid]

        support_lines = build_support_label_lines(table_group, col_idx)
        case = {
            "column_id": row["column_id"],
            "table_id": tid,
            "col_idx": col_idx,
            "column_values": row["column_values"],
            "ground_truth": row["ground_truth"],
            "distractor": row["hard_distractor"],
            "all_wrong": bool(row["all_wrong"]),
            "empty_support": len(support_lines) == 0,
            "query_table_string": build_query_table_string(table_group),
            "table_support_label_lines": support_lines,
            "sc_retrieved_examples": [
                {
                    "rank": int(ex["rank"]),
                    "label": ex["label"],
                    "data": ex["data"],
                    "sim": float(ex["sim"]),
                }
                for ex in sc_item["top_k"]
            ],
            "mc_retrieved_examples": [
                {
                    "rank": int(ex["rank"]),
                    "table_id": ex["table_id"],
                    "labels": list(ex["labels"]),
                    "formatted_string": ex["formatted_string"],
                    "sim": float(ex["sim"]),
                }
                for ex in mc_item["top_k"]
            ],
        }
        cases.append(case)

    out_path = os.path.join(OUT_DIR, "probing_dataset.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Prepared {len(cases)} probing cases.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
