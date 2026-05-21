import os
import json
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import ARTIFACT_DIR, EMB_DIR, DATA_DIR, EVAL_DIR, RESULTS_DIR, get_cosine_sim, ensure_dir, safe_literal_eval

OUT_DIR = ensure_dir(os.path.join(EVAL_DIR, "mc"))
EXP3_ARTIFACT_DIR = ensure_dir(ARTIFACT_DIR)

K = 5  # number of retrieved tables


def aggregate_label_scores(top_label_sets, top_sims, strategy):
    scores = {}
    for rank, (label_set, sim) in enumerate(zip(top_label_sets, top_sims)):
        w = 1.0 / (rank + 1) if strategy == "rr" else float(sim)
        for label in label_set:
            scores[label] = scores.get(label, 0) + w
    return scores


def load_mc_rag_table_metrics() -> pd.DataFrame:
    mc_rag_path = os.path.join(RESULTS_DIR, "multi_col_rag.csv")
    if os.path.exists(mc_rag_path):
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))
        mc_rag = pd.read_csv(mc_rag_path)

        preds = []
        for pred_list in mc_rag["pred"]:
            preds.extend(safe_literal_eval(pred_list))

        rebuilt = test_df[["table_id", "col_idx", "class"]].rename(
            columns={"class": "ground_truth"}
        ).copy()
        rebuilt["mc_rag_pred"] = preds
        rebuilt["mc_rag_correct"] = rebuilt["mc_rag_pred"] == rebuilt["ground_truth"]

        return (rebuilt.groupby("table_id")
                      .agg(
                          mc_rag_col_accuracy=("mc_rag_correct", "mean"),
                          mc_rag_n_correct=("mc_rag_correct", "sum"),
                          mc_rag_n_cols=("mc_rag_correct", "size"),
                      )
                      .reset_index())

    unified_path = os.path.join(ROOT, "artifacts", "failure-patterns", "unified_df.csv")
    if os.path.exists(unified_path):
        df = pd.read_csv(unified_path)
        return (df.groupby("table_id")
                  .agg(
                      mc_rag_col_accuracy=("mc_rag_correct", "mean"),
                      mc_rag_n_correct=("mc_rag_correct", "sum"),
                      mc_rag_n_cols=("mc_rag_correct", "size"),
                  )
                  .reset_index())

    raise FileNotFoundError(
        "Could not compute MC RAG table metrics: neither data/released/cta-predictions/multi_col_rag.csv "
        "nor artifacts/failure-patterns/unified_df.csv is available."
    )


def run():
    print("\n[3B] Multi-Column Retrieval Coverage Analysis...")
    test_emb = np.load(os.path.join(EMB_DIR, "multi_col_table_embeddings.npy"))
    with open(os.path.join(EMB_DIR, "multi_col_table_metadata.json")) as f:
        test_meta = json.load(f)
    train_emb_df = pd.read_csv(os.path.join(DATA_DIR, "trainset_for_multi_rag_embeds.csv"))
    train_emb    = np.array([ast.literal_eval(e) for e in train_emb_df["embedding"].values])
    train_raw    = pd.read_csv(os.path.join(DATA_DIR, "trainset_for_multi_rag.csv"))
    train_label_map = train_raw.groupby("table_id")["class"].apply(set).to_dict()


    def format_example(group):
        lines = ["Classify these columns:\n"]
        labels = []
        for idx in sorted(group["col_idx"].unique()):
            data = group[group["col_idx"] == idx]["data"].values[0]
            lines.append(f"Column {idx}: {data}")
            labels.append(group[group["col_idx"] == idx]["class"].values[0])
        return "\n".join(lines), labels

    train_example_map = {}
    for tid, grp in train_raw.groupby("table_id"):
        train_example_map[tid] = format_example(grp)

    strategies = ["rr", "sim"]
    stats = {s: [] for s in strategies}
    artifacts = []  # for Exp4 MC-2
    mc_rag_table = load_mc_rag_table_metrics()

    for i, q_vec in enumerate(tqdm(test_emb, desc="  MC Retrieval", leave=False)):
        gt_labels = set(test_meta[i]["ground_truth_labels"])
        m         = test_meta[i]["column_count"]
        u         = len(gt_labels)

        sims    = get_cosine_sim(q_vec, train_emb)
        top_idx = np.argsort(sims)[-K:][::-1]
        top_tids = train_emb_df.iloc[top_idx]["table_id"].values
        top_sims = sims[top_idx]

        top_label_sets = [train_label_map.get(tid, set()) for tid in top_tids]

        top_k_artifact = []
        for r, (tid, sim_val) in enumerate(zip(top_tids, top_sims)):
            formatted_str, labels = train_example_map.get(tid, ("", []))
            top_k_artifact.append({
                "rank": r,
                "table_id": str(tid),
                "labels": labels,
                "formatted_string": formatted_str,
                "sim": float(sim_val),
            })
        artifacts.append({
            "table_id": test_meta[i]["table_id"],
            "column_count": test_meta[i]["column_count"],
            "ground_truth_labels": test_meta[i]["ground_truth_labels"],
            "top_k": top_k_artifact,
        })

        for s in strategies:
            scores       = aggregate_label_scores(top_label_sets, top_sims, s)
            top_m_labels = set(sorted(scores, key=lambda l: (-scores[l], l))[:m])

            inter = gt_labels & top_m_labels
            union = gt_labels | top_m_labels
            coverage_rate = len(inter) / len(gt_labels) if gt_labels else 0.0
            stats[s].append({
                "table_id":      test_meta[i]["table_id"],
                "m":             m,
                "u":             u,
                "m_eq_u":        int(m == u),
                "recall_at_m":   coverage_rate,
                "coverage_rate": coverage_rate,
                "jaccard":       len(inter) / len(union) if union else 0.0,
                "exact_match":   int(top_m_labels == gt_labels),
            })

    rows = []
    for s in strategies:
        df = pd.DataFrame(stats[s])
        rows.append({
            "Strategy":         f"coverage_{s}",
            "Recall@m":         df["recall_at_m"].mean(),
            "Coverage_Rate":    df["coverage_rate"].mean(),
            "Jaccard":          df["jaccard"].mean(),
            "Exact_Match_Rate": df["exact_match"].mean(),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "mc_summary.csv"), index=False)
    print(summary_df.to_string(index=False))


    alignment_summary_rows = []
    for s in strategies:
        per_table_df = pd.DataFrame(stats[s])
        per_table_df.to_csv(os.path.join(OUT_DIR, f"mc_per_table_{s}.csv"), index=False)

        aligned = per_table_df.merge(mc_rag_table, on="table_id", how="left")
        aligned["mc_rag_all_columns_correct"] = (
            aligned["mc_rag_n_correct"] == aligned["mc_rag_n_cols"]
        ).astype(int)
        aligned["retrieval_full_coverage"] = (aligned["recall_at_m"] == 1.0).astype(int)
        aligned["strategy"] = s
        aligned.to_csv(os.path.join(OUT_DIR, f"mc_alignment_{s}.csv"), index=False)

        exact_groups = aligned.groupby("exact_match")["mc_rag_col_accuracy"].mean().to_dict()
        full_groups = aligned.groupby("retrieval_full_coverage")["mc_rag_col_accuracy"].mean().to_dict()
        alignment_summary_rows.append({
            "strategy": s,
            "tables": len(aligned),
            "m_eq_u_all": bool(aligned["m_eq_u"].all()),
            "exact_match_1_tables": int((aligned["exact_match"] == 1).sum()),
            "exact_match_0_tables": int((aligned["exact_match"] == 0).sum()),
            "mc_rag_acc_when_exact_1": float(exact_groups.get(1, np.nan)),
            "mc_rag_acc_when_exact_0": float(exact_groups.get(0, np.nan)),
            "retrieval_full_coverage_1_tables": int((aligned["retrieval_full_coverage"] == 1).sum()),
            "retrieval_full_coverage_0_tables": int((aligned["retrieval_full_coverage"] == 0).sum()),
            "mc_rag_acc_when_full_coverage_1": float(full_groups.get(1, np.nan)),
            "mc_rag_acc_when_full_coverage_0": float(full_groups.get(0, np.nan)),
        })

    pd.DataFrame(alignment_summary_rows).to_csv(
        os.path.join(OUT_DIR, "mc_alignment_summary.csv"), index=False
    )


    artifact_path = os.path.join(EXP3_ARTIFACT_DIR, "mc_retrieval_artifacts.json")
    with open(artifact_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"  Retrieval artifacts saved -> {artifact_path} ({len(artifacts)} tables)")


if __name__ == "__main__":
    run()
