import json
import os
from collections import Counter

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EXP_DIR)
ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "binary-probing")
OUT_DIR = os.path.join(ROOT, "data", "released", "binary-probing")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "gpt-5.4-mini-2026-03-17"
client = OpenAI()


SYSTEM_PROMPT = (
    "Your task is to classify the columns of a given table with only one of the provided column semantic types. "
    "Your instructions are: 1. Look at the column and the types given to you. 2. Examine the values of the column. "
    "3. Select a type that best represents the meaning of the column. 4. Answer with the selected type only, and print the type only once. "
    "You may be given labeled reference examples as guidance. "
    "The format of the answer should be like this: type Print 'I don't know' if you are not able to find the semantic type."
)


def choice_line(gt, dist):
    return f"Choose one for the target column: '{gt}' or '{dist}'"


def support_block(case):
    if not case["table_support_label_lines"]:
        return "Known labels for other columns: none"
    return "Known labels for other columns:\n" + "\n".join(case["table_support_label_lines"])


def build_sc1(case):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Target column values:\n{str(case['column_values'])[:500]}\n\n"
                + choice_line(case["ground_truth"], case["distractor"])
            ),
        },
    ]


def build_sc2rag(case):
    examples = []
    for idx, ex in enumerate(case["sc_retrieved_examples"], 1):
        examples.append(
            f"Reference example {idx}\n"
            f"Column values: {str(ex['data'])[:500]}\n"
            f"Annotated label: {ex['label']}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Retrieved labeled reference examples:\n\n"
                + "\n\n".join(examples)
                + "\n\nTarget column values:\n"
                + f"{str(case['column_values'])[:500]}\n\n"
                + choice_line(case["ground_truth"], case["distractor"])
            ),
        },
    ]


def build_mc1(case):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{case['query_table_string']}\n\n"
                f"Target column: Column {case['col_idx']}\n\n"
                f"{support_block(case)}\n\n"
                + choice_line(case["ground_truth"], case["distractor"])
            ),
        },
    ]


def build_mc2rag(case):
    examples = []
    for idx, ex in enumerate(case["mc_retrieved_examples"], 1):
        examples.append(
            f"Reference table {idx}\n"
            f"{ex['formatted_string']}\n"
            f"Annotated labels: {ex['labels']}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Retrieved labeled reference tables:\n\n"
                + "\n\n".join(examples)
                + "\n\nQuery table:\n"
                + f"{case['query_table_string']}\n\n"
                + f"Target column: Column {case['col_idx']}\n\n"
                + f"{support_block(case)}\n\n"
                + choice_line(case["ground_truth"], case["distractor"])
            ),
        },
    ]


CONDITIONS = [
    ("SC1", build_sc1),
    ("SC2RAG", build_sc2rag),
    ("MC1", build_mc1),
    ("MC2RAG", build_mc2rag),
]


def parse_binary_choice(raw_text, gt, distractor):
    raw = (raw_text or "").strip()
    low = raw.lower()
    gt_low = gt.lower()
    dist_low = distractor.lower()

    if low == gt_low:
        return gt, "gt"
    if low == dist_low:
        return distractor, "distractor"
    if gt_low in low and dist_low not in low:
        return gt, "gt"
    if dist_low in low and gt_low not in low:
        return distractor, "distractor"
    return raw, "out_of_given_label"


def run_condition(cases, condition_name, build_fn):
    rows = []
    for case in tqdm(cases, desc=f"  {condition_name}", leave=False):
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_fn(case),
            temperature=0,
        )
        raw_prediction = response.choices[0].message.content.strip()
        parsed_prediction, choice_status = parse_binary_choice(
            raw_prediction,
            case["ground_truth"],
            case["distractor"],
        )

        rows.append(
            {
                "column_id": case["column_id"],
                "table_id": case["table_id"],
                "col_idx": case["col_idx"],
                "ground_truth": case["ground_truth"],
                "distractor": case["distractor"],
                "prediction": parsed_prediction,
                "raw_prediction": raw_prediction,
                "choice_status": choice_status,
                "out_of_given_label": choice_status == "out_of_given_label",
                "correct": choice_status == "gt",
                "all_wrong": case["all_wrong"],
                "empty_support": case["empty_support"],
            }
        )

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"probing_results_{condition_name}.csv")
    df.to_csv(out_path, index=False)

    print(
        f"  {condition_name}: {df['correct'].mean():.1%} binary accuracy  "
        f"({int(df['correct'].sum())}/{len(df)}) | "
        f"out_of_given_label={int(df['out_of_given_label'].sum())}"
    )
    return df


def main():
    print("=" * 60)
    print("BINARY-PROBING: API RUNNER")
    print("=" * 60)

    with open(os.path.join(ARTIFACT_DIR, "probing_dataset.json")) as f:
        cases = json.load(f)

    print(f"\nTotal cases: {len(cases)}")
    print(f"Permanent hard cases: {sum(case['all_wrong'] for case in cases)}")
    print(f"Empty support cases: {sum(case['empty_support'] for case in cases)}")

    pair_counts = Counter((case["ground_truth"], case["distractor"]) for case in cases)
    print("\nTop 10 (GT → distractor) pairs:")
    for (gt, dist), n in pair_counts.most_common(10):
        print(f"  {gt:12s} → {dist:12s}  n={n}")

    total_calls = len(CONDITIONS) * len(cases)
    print(f"\nTotal API calls: {len(CONDITIONS)} conditions × {len(cases)} cases = {total_calls}\n")

    for condition_name, build_fn in CONDITIONS:
        print(f"[{condition_name}]")
        run_condition(cases, condition_name, build_fn)

    print(f"\n[Done] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
