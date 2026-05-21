import ast
import os
import matplotlib
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EXP_DIR)
DATA_DIR = os.path.join(ROOT, "data", "raw")
RESULTS_DIR = os.path.join(ROOT, "data", "released", "cta-predictions")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "failure-patterns")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# method keys and display names (ordered by complexity)
METHODS = ["sc_zs", "sc_fs", "sc_rag", "mc_zs", "mc_fs", "mc_rag", "mc_cot"]
METHOD_LABELS = {
    "sc_zs":  "SC Zero-Shot",
    "sc_fs":  "SC Few-Shot",
    "sc_rag": "SC RAG",
    "mc_zs":  "MC Zero-Shot",
    "mc_fs":  "MC Few-Shot",
    "mc_rag": "MC RAG",
    "mc_cot": "MC CoT",
}


def flatten_mc(mc_series: pd.Series) -> list:
    """Flatten table-level list predictions to column-level."""
    result = []
    for val in mc_series:
        result.extend(ast.literal_eval(val))
    return result


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load all results and merge into a single column-level dataframe."""
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))

    # single-col: 300 rows, already column-level
    sc_zs  = pd.read_csv(os.path.join(RESULTS_DIR, "single_col_zero_shot.csv"))
    sc_fs  = pd.read_csv(os.path.join(RESULTS_DIR, "single_col_few_shot.csv"))
    sc_rag = pd.read_csv(os.path.join(RESULTS_DIR, "single_col_rag.csv"))

    # multi-col: 90 rows (table-level), flatten to column-level
    mc_zs  = pd.read_csv(os.path.join(RESULTS_DIR, "multi_col_zero_shot.csv"))
    mc_fs  = pd.read_csv(os.path.join(RESULTS_DIR, "multi_col_few_shot.csv"))
    mc_rag = pd.read_csv(os.path.join(RESULTS_DIR, "multi_col_rag.csv"))
    mc_cot = pd.read_csv(os.path.join(RESULTS_DIR, "multi_col_cot.csv"))

    df = test_df.rename(columns={"class": "ground_truth", "data": "column_values"}).copy()
    df = df.reset_index(drop=True)
    df["column_id"] = df["table_id"].astype(str) + "::" + df["col_idx"].astype(str)

    df["sc_zs_pred"]  = sc_zs["pred"].values
    df["sc_fs_pred"]  = sc_fs["pred"].values
    df["sc_rag_pred"] = sc_rag["pred"].values
    df["mc_zs_pred"]  = flatten_mc(mc_zs["pred"])
    df["mc_fs_pred"]  = flatten_mc(mc_fs["pred"])
    df["mc_rag_pred"] = flatten_mc(mc_rag["pred"])
    df["mc_cot_pred"] = flatten_mc(mc_cot["pred"])

    for m in METHODS:
        df[f"{m}_correct"] = df[f"{m}_pred"] == df["ground_truth"]

    df["all_wrong"] = ~df[[f"{m}_correct" for m in METHODS]].any(axis=1)

    return df, METHODS


def analyze_hard_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    hard = df[df["all_wrong"]]
    total = len(df)

    print(f"\n=== 1B: Hard Cases ===")
    print(f"  Total columns    : {total}")
    print(f"  Hard cases       : {len(hard)} ({len(hard) / total * 100:.1f}%)")

    type_dist  = hard["ground_truth"].value_counts()
    type_total = df["ground_truth"].value_counts()
    hard_rate  = (type_dist / type_total).dropna().sort_values(ascending=False)

    print("\n  Hard-case rate by type (all types with any hard case):")
    print(hard_rate.to_string())

    always_wrong = hard_rate[hard_rate == 1.0]
    print(f"\n  Types with 100% hard-case rate ({len(always_wrong)}): {always_wrong.index.tolist()}")

    return hard, hard_rate


if __name__ == "__main__":
    print("=" * 60)
    print("Prediction error analysis")
    print("=" * 60)

    print("\n[1A] Loading data...")
    df, methods = load_data()
    print(f"  Unified dataframe: {df.shape} (columns x fields)")

    hard, hard_rate = analyze_hard_cases(df)

    out_df   = os.path.join(ARTIFACT_DIR, "unified_df.csv")
    df.to_csv(out_df, index=False)
    print(f"\n[Saved] {out_df}")


    print("\n" + "=" * 60)
    print("Prediction error analysis complete.")
    print("=" * 60)
