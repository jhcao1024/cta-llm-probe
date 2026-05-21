import ast
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EXP_DIR)
DATA_DIR = os.path.join(ROOT, "data", "raw")
EMB_DIR = os.path.join(ROOT, "artifacts", "representation-bias", "embeddings")
EVAL_DIR = os.path.join(EXP_DIR, "outputs")
RESULTS_DIR = os.path.join(ROOT, "data", "released", "cta-predictions")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "retrieval-diagnostics")

def get_cosine_sim(query_vec, candidate_matrix):
    """Efficiently compute cosine similarities for a query against a matrix."""
    dot = np.dot(candidate_matrix, query_vec)
    norm_q = np.linalg.norm(query_vec)
    norm_c = np.linalg.norm(candidate_matrix, axis=1)
    return dot / (norm_q * norm_c + 1e-9)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def safe_literal_eval(entry):
    try:
        val = ast.literal_eval(entry)
        if isinstance(val, list):
            return val
        else:
            return [str(val)]
    except (ValueError, SyntaxError):
        return [entry]
