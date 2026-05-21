import json
import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(EXP_DIR)
FAILURE_PATTERNS_ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts", "failure-patterns")
REPRESENTATION_BIAS_ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts", "representation-bias")
EMB_DIR = os.path.join(REPRESENTATION_BIAS_ARTIFACT_DIR, "embeddings")
FIG_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)


def load_label_embeddings() -> tuple[np.ndarray, list[str]]:
    emb = np.load(os.path.join(EMB_DIR, "label_embeddings.npy"))
    with open(os.path.join(EMB_DIR, "label_metadata.json")) as f:
        labels = json.load(f)
    return emb, labels


def load_column_embeddings() -> np.ndarray:
    return np.load(os.path.join(EMB_DIR, "column_embeddings.npy"))


def load_unified_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FAILURE_PATTERNS_ARTIFACT_DIR, "unified_df.csv")).reset_index(drop=True)


def reduce_tsne(embeddings: np.ndarray, perplexity: int, max_iter: int = 2000,
                init: str = "pca") -> np.ndarray:
    return TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                init=init, random_state=42).fit_transform(embeddings)

'''
def reduce_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float = 0.1,
                densmap: bool = False) -> np.ndarray:
    from umap import UMAP

    return UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                densmap=densmap, random_state=42).fit_transform(embeddings)
'''


# Shared semantic grouping used by both label_viz and column_viz
SEMANTIC_GROUPS: dict[str, list[str]] = {
    "geographic":   ["country", "city", "state", "region", "location", "address", "origin"],
    "person":       ["name", "age", "sex", "gender", "nationality"],
    "temporal":     ["year", "day", "duration"],
    "organization": ["company", "club", "team"],
    "categorical":  ["category", "type", "status", "format", "result"],
    "identifier":   ["code", "symbol", "isbn", "rank", "position"],
    "creative":     ["album", "artist"],
    "descriptive":  ["notes", "description"],
}

GROUP_COLORS: dict[str, str] = {
    "geographic":   "#2ecc71",
    "person":       "#3498db",
    "temporal":     "#e67e22",
    "organization": "#9b59b6",
    "categorical":  "#e74c3c",
    "identifier":   "#1abc9c",
    "creative":     "#f39c12",
    "descriptive":  "#95a5a6",
}
