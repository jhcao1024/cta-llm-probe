import os
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EXP_DIR)
INPUT_DIR = os.path.join(ROOT, "data", "released", "binary-probing")
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITIONS = ["SC1", "SC2RAG", "MC1", "MC2RAG"]


def load_dfs():
    return {
        condition: pd.read_csv(os.path.join(INPUT_DIR, f"probing_results_{condition}.csv"))
        for condition in CONDITIONS
    }


def table_overall(dfs):
    rows = []
    for condition in CONDITIONS:
        df = dfs[condition]
        rows.append(
            {
                "condition": condition,
                "n": len(df),
                "binary_accuracy": round(df["correct"].mean(), 4),
                "n_correct": int(df["correct"].sum()),
                "out_of_given_label_rate": round(df["out_of_given_label"].mean(), 4),
                "n_out_of_given_label": int(df["out_of_given_label"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "4e_overall_accuracy.csv"), index=False)
    print("\n[Overall]")
    print(out.to_string(index=False))
    return out


def table_out_of_given_summary(dfs):
    rows = []
    for condition in CONDITIONS:
        df = dfs[condition]
        grouped = df.groupby("choice_status").size().to_dict()
        rows.append(
            {
                "condition": condition,
                "gt": int(grouped.get("gt", 0)),
                "distractor": int(grouped.get("distractor", 0)),
                "out_of_given_label": int(grouped.get("out_of_given_label", 0)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "4e_out_of_given_summary.csv"), index=False)
    print("\n[Choice Status Counts]")
    print(out.to_string(index=False))
    return out


def table_out_of_given_cases(dfs):
    rows = []
    for condition in CONDITIONS:
        df = dfs[condition]
        subset = df[df["out_of_given_label"]].copy()
        subset["condition"] = condition
        rows.append(
            subset[
                [
                    "condition",
                    "column_id",
                    "table_id",
                    "col_idx",
                    "ground_truth",
                    "distractor",
                    "prediction",
                    "raw_prediction",
                    "all_wrong",
                ]
            ]
        )
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(os.path.join(OUTPUT_DIR, "4e_out_of_given_cases.csv"), index=False)
    print(f"\n[Out-of-given-label cases] {len(out)} rows")
    return out


def table_per_pair(dfs):
    pairs = sorted(
        set((row["ground_truth"], row["distractor"]) for _, row in dfs["SC1"].iterrows())
    )
    rows = []
    for gt, dist in pairs:
        row = {"gt": gt, "distractor": dist}
        for condition in CONDITIONS:
            grp = dfs[condition][
                (dfs[condition]["ground_truth"] == gt)
                & (dfs[condition]["distractor"] == dist)
            ]
            row[f"{condition}_n"] = len(grp)
            row[f"{condition}_acc"] = round(grp["correct"].mean(), 4) if len(grp) else None
            row[f"{condition}_oogl"] = (
                round(grp["out_of_given_label"].mean(), 4) if len(grp) else None
            )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("SC1_n", ascending=False)
    out.to_csv(os.path.join(OUTPUT_DIR, "4e_per_pair_accuracy.csv"), index=False)
    return out


def table_per_gt(dfs):
    gts = sorted(dfs["SC1"]["ground_truth"].unique())
    rows = []
    for gt in gts:
        row = {"ground_truth": gt}
        for condition in CONDITIONS:
            grp = dfs[condition][dfs[condition]["ground_truth"] == gt]
            row[f"{condition}_n"] = len(grp)
            row[f"{condition}_acc"] = round(grp["correct"].mean(), 4) if len(grp) else None
            row[f"{condition}_oogl"] = (
                round(grp["out_of_given_label"].mean(), 4) if len(grp) else None
            )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("SC1_acc")
    out.to_csv(os.path.join(OUTPUT_DIR, "4e_per_gt_accuracy.csv"), index=False)
    return out


def table_per_case(dfs):
    base = dfs["SC1"][
        [
            "column_id",
            "table_id",
            "col_idx",
            "ground_truth",
            "distractor",
            "all_wrong",
            "empty_support",
        ]
    ].copy()
    for condition in CONDITIONS:
        tier_df = dfs[condition][
            [
                "column_id",
                "table_id",
                "prediction",
                "raw_prediction",
                "choice_status",
                "correct",
                "out_of_given_label",
            ]
        ].rename(
            columns={
                "prediction": f"pred_{condition}",
                "raw_prediction": f"raw_{condition}",
                "choice_status": f"status_{condition}",
                "correct": f"correct_{condition}",
                "out_of_given_label": f"oogl_{condition}",
            }
        )
        base = base.merge(tier_df, on=["column_id", "table_id"], how="left", validate="one_to_one")

    base["n_conditions_correct"] = sum(base[f"correct_{condition}"] for condition in CONDITIONS)
    base["n_conditions_oogl"] = sum(base[f"oogl_{condition}"] for condition in CONDITIONS)
    base.to_csv(os.path.join(OUTPUT_DIR, "4e_per_case_detail.csv"), index=False)
    return base


def main():
    print("=" * 60)
    print("BINARY-PROBING: ANALYSIS")
    print("=" * 60)

    dfs = load_dfs()
    table_overall(dfs)
    table_out_of_given_summary(dfs)
    table_out_of_given_cases(dfs)
    table_per_pair(dfs)
    table_per_gt(dfs)
    table_per_case(dfs)

    print(f"\n[Done] All tables saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
