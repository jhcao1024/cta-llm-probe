import json
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from utils import ARTIFACT_DIR, EMB_DIR, DATA_DIR, EVAL_DIR, ROOT, get_cosine_sim, ensure_dir

OUT_DIR = ensure_dir(os.path.join(EVAL_DIR, "sc"))
EXP3_ARTIFACT_DIR = ensure_dir(ARTIFACT_DIR)


def majority_vote(labels, scores, strategy="rr"):
    """
    labels: ordered best-first (rank 0 = most similar)
    scores: cosine similarities, parallel to labels
    strategy: "plain" | "rr" | "sim"
    """
    if strategy == "plain":
        # Count occurrences; break ties by earliest (highest-rank) appearance
        counts, first_rank = {}, {}
        for i, l in enumerate(labels):
            if l not in counts:
                counts[l] = 0
                first_rank[l] = i
            counts[l] += 1
        return max(counts, key=lambda l: (counts[l], -first_rank[l]))

    weights = [1.0 / (i + 1) for i in range(len(labels))] if strategy == "rr" else list(scores)
    weighted = {}
    for l, w in zip(labels, weights):
        weighted[l] = weighted.get(l, 0) + w
    return max(weighted, key=weighted.get)


def _quadrant(a_correct: np.ndarray, b_correct: np.ndarray,
              label_a: str, label_b: str, total: int) -> dict:
    q = {
        "both_correct":      int(( a_correct &  b_correct).sum()),
        f"{label_a}_only":   int(( a_correct & ~b_correct).sum()),
        f"{label_b}_only":   int((~a_correct &  b_correct).sum()),
        "both_wrong":        int((~a_correct & ~b_correct).sum()),
    }
    print(f"\n  Quadrant Analysis ({label_a} vs {label_b}):")
    for k, v in q.items():
        print(f"    {k}: {v} ({v/total*100:.1f}%)")
    return q


def run():
    print("\n[3A] Single-Column Retrieval Analysis...")
    test_emb  = np.load(os.path.join(EMB_DIR, "column_embeddings.npy"))
    test_meta = pd.read_csv(os.path.join(EMB_DIR, "column_metadata.csv"))
    train_df  = pd.read_csv(os.path.join(DATA_DIR, "RAG_2000_sample.csv"))
    train_emb = np.array([ast.literal_eval(e) for e in train_df["embedding"].values])
    train_labels = train_df["class"].values
    train_data   = train_df["data"].values

    strategies = ["plain", "rr", "sim"]
    preds = {s: [] for s in strategies}
    artifacts = []  # for Exp4 SC-2

    for qi, q_vec in enumerate(tqdm(test_emb, desc="  SC Retrieval", leave=False)):
        sims       = get_cosine_sim(q_vec, train_emb)
        idx        = np.argsort(sims)[-5:][::-1]
        top_labels = train_labels[idx]
        top_sims   = sims[idx]
        for s in strategies:
            preds[s].append(majority_vote(top_labels, top_sims, s))

        artifacts.append({
            "table_id": test_meta["table_id"].iloc[qi],
            "col_idx":  int(test_meta["col_idx"].iloc[qi]),
            "top_k": [
                {"rank": r, "label": str(train_labels[idx[r]]),
                 "data": str(train_data[idx[r]]), "sim": float(top_sims[r])}
                for r in range(len(idx))
            ],
        })

    gt = test_meta["class"].values


    def _load_pred(fname):
        path = os.path.join(ROOT, "data", "released", "cta-predictions", fname)
        return pd.read_csv(path)["pred"].values if os.path.exists(path) else None

    zs_pred  = _load_pred("single_col_zero_shot.csv")
    fs_pred  = _load_pred("single_col_few_shot.csv")
    llm_pred = _load_pred("single_col_rag.csv")


    rows = []
    # Ordered progression: ZS → MV_plain → FS → MV_rr → MV_sim → RAG
    if zs_pred is not None:
        rows.append({"Method": "SC_ZS",    "Accuracy": accuracy_score(gt, zs_pred),
                     "Micro_F1": f1_score(gt, zs_pred, average="micro"),
                     "Macro_F1": f1_score(gt, zs_pred, average="macro")})
    for s in strategies:
        rows.append({"Method": f"MV_{s}",  "Accuracy": accuracy_score(gt, preds[s]),
                     "Micro_F1": f1_score(gt, preds[s], average="micro"),
                     "Macro_F1": f1_score(gt, preds[s], average="macro")})
    if fs_pred is not None:
        rows.append({"Method": "SC_FS",    "Accuracy": accuracy_score(gt, fs_pred),
                     "Micro_F1": f1_score(gt, fs_pred, average="micro"),
                     "Macro_F1": f1_score(gt, fs_pred, average="macro")})
    if llm_pred is not None:
        rows.append({"Method": "SC_RAG",   "Accuracy": accuracy_score(gt, llm_pred),
                     "Micro_F1": f1_score(gt, llm_pred, average="micro"),
                     "Macro_F1": f1_score(gt, llm_pred, average="macro")})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "sc_summary.csv"), index=False)
    print(summary_df.to_string(index=False))


    artifact_path = os.path.join(EXP3_ARTIFACT_DIR, "sc_retrieval_artifacts.json")
    with open(artifact_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"  Retrieval artifacts saved -> {artifact_path} ({len(artifacts)} columns)")


    per_col = test_meta[["table_id", "col_idx", "class"]].copy()
    if zs_pred  is not None: per_col["pred_zs"]  = zs_pred
    for s in strategies:     per_col[f"pred_{s}"] = preds[s]
    if fs_pred  is not None: per_col["pred_fs"]  = fs_pred
    if llm_pred is not None: per_col["pred_sc_rag"] = llm_pred
    per_col.to_csv(os.path.join(OUT_DIR, "sc_per_column.csv"), index=False)

    total = len(gt)
    mv = np.array(preds["plain"])
    mv_correct = mv == gt


    if llm_pred is not None:
        llm_correct = llm_pred == gt
        q_mv_rag = _quadrant(mv_correct, llm_correct, "MV_plain", "SC_RAG", total)
        pd.DataFrame([q_mv_rag]).to_csv(
            os.path.join(OUT_DIR, "sc_quadrant_mv_vs_rag.csv"), index=False)


    if fs_pred is not None:
        fs_correct = fs_pred == gt
        q_mv_fs = _quadrant(mv_correct, fs_correct, "MV_plain", "SC_FS", total)
        pd.DataFrame([q_mv_fs]).to_csv(
            os.path.join(OUT_DIR, "sc_quadrant_mv_vs_fs.csv"), index=False)

    # Save per-type accuracy whenever at least one non-retrieval baseline exists.
    if zs_pred is not None or fs_pred is not None or llm_pred is not None:
        per_type = []
        for label in sorted(set(gt)):
            mask = gt == label
            entry = {"label": label, "n": int(mask.sum())}
            if zs_pred  is not None: entry["SC_ZS"]    = float((zs_pred[mask]  == gt[mask]).mean())
            entry["MV_plain"] = float((mv[mask] == gt[mask]).mean())
            for s in ["rr", "sim"]:
                mv_s = np.array(preds[s])
                entry[f"MV_{s}"] = float((mv_s[mask] == gt[mask]).mean())
            if fs_pred  is not None: entry["SC_FS"]    = float((fs_pred[mask]  == gt[mask]).mean())
            if llm_pred is not None: entry["SC_RAG"]   = float((llm_pred[mask] == gt[mask]).mean())
            per_type.append(entry)
        per_type_df = pd.DataFrame(per_type).sort_values("MV_plain")
        per_type_df.to_csv(os.path.join(OUT_DIR, "sc_per_type.csv"), index=False)
        print(f"  Per-type accuracy saved → sc_per_type.csv")


if __name__ == "__main__":
    run()
