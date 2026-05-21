import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT       = os.path.dirname(EXP_DIR)
DATA_DIR   = os.path.join(ROOT, "data", "raw")
OUT_DIR    = os.path.join(ROOT, "artifacts", "representation-bias", "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI()

LABELS = [
    "sex", "category", "album", "status", "origin", "format", "day", "location",
    "notes", "duration", "nationality", "region", "club", "address", "rank", "name",
    "position", "description", "country", "state", "city", "code", "symbol", "isbn",
    "age", "type", "gender", "team", "year", "company", "result", "artist",
]


def get_embeddings_batch(texts: list[str], batch_size: int = 100,
                         model: str = "text-embedding-ada-002") -> np.ndarray:
    """Call OpenAI embeddings API in batches, return stacked array."""
    texts = [str(t).replace("\n", " ") for t in texts]
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  batches", leave=False):
        response = client.embeddings.create(input=texts[i:i + batch_size], model=model)
        all_embeddings.extend([d.embedding for d in response.data])
    return np.array(all_embeddings)


def format_table_for_embedding(group: pd.DataFrame) -> str:
    """
    Serialize a table group into a single string for table-level embedding.
    Must match the format_table logic used in cta.py RAG retrieval.
    """
    columns = []
    for idx, sub_group in group.groupby("col_idx"):
        vals = sub_group["data"].iloc[:200].astype(str).tolist()
        columns.append(f"Column {idx}: " + " ".join(vals))
    return " | ".join(columns)


def _skip_or_run(path: str, label: str, generate_fn):
    """Run generate_fn and save result only if path doesn't already exist."""
    if os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} already exists.")
        return
    print(f"  Generating {label}...")
    generate_fn(path)


def _gen_labels(_):
    emb = get_embeddings_batch(LABELS)
    np.save(os.path.join(OUT_DIR, "label_embeddings.npy"), emb)
    with open(os.path.join(OUT_DIR, "label_metadata.json"), "w") as f:
        json.dump(LABELS, f)


def _gen_columns(test_df: pd.DataFrame, _):
    emb = get_embeddings_batch(test_df["data"].tolist())
    np.save(os.path.join(OUT_DIR, "column_embeddings.npy"), emb)
    test_df[["table_id", "col_idx", "class"]].to_csv(
        os.path.join(OUT_DIR, "column_metadata.csv"), index=False
    )


def _gen_tables(test_df: pd.DataFrame, _):
    table_ids = list(test_df["table_id"].unique())
    table_strings, table_metadata = [], []
    for tid in table_ids:
        group = test_df[test_df["table_id"] == tid]
        table_strings.append(format_table_for_embedding(group))
        table_metadata.append({
            "table_id":            tid,
            "ground_truth_labels": list(group["class"].values),
            "column_count":        len(group),
        })
    emb = get_embeddings_batch(table_strings, batch_size=15)
    np.save(os.path.join(OUT_DIR, "multi_col_table_embeddings.npy"), emb)
    with open(os.path.join(OUT_DIR, "multi_col_table_metadata.json"), "w") as f:
        json.dump(table_metadata, f, indent=2)


def main():
    print("=" * 60)
    print("Generating embeddings for representation-bias")
    print("=" * 60)

    # 1. Label embeddings
    print("\n[1] Label embeddings (32 labels)")
    _skip_or_run(os.path.join(OUT_DIR, "label_embeddings.npy"),
                 "label embeddings", _gen_labels)

    # 2. Column embeddings
    print("\n[2] Column embeddings (300 test columns)")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))
    _skip_or_run(os.path.join(OUT_DIR, "column_embeddings.npy"),
                 "column embeddings", lambda p: _gen_columns(test_df, p))

    # 3. Multi-column whole table embeddings
    print("\n[3] Multi-column whole table embeddings (90 tables, multi-column format)")
    _skip_or_run(os.path.join(OUT_DIR, "multi_col_table_embeddings.npy"),
                 "multi-column whole table embeddings", lambda p: _gen_tables(test_df, p))

    print("\nDone! Files saved in 'artifacts/representation-bias/embeddings/':")
    for fname in [
        "label_embeddings.npy", "label_metadata.json",
        "column_embeddings.npy", "column_metadata.csv",
        "multi_col_table_embeddings.npy", "multi_col_table_metadata.json",
    ]:
        status = "✓" if os.path.exists(os.path.join(OUT_DIR, fname)) else "missing"
        print(f"  [{status}] {fname}")


if __name__ == "__main__":
    main()
