# Retrieval Diagnostics

## Expected Inputs

- `../data/raw/*`
- `../data/released/cta-predictions/*`
- `../artifacts/failure-patterns/unified_df.csv`
- `../artifacts/representation-bias/embeddings/*`


## Run

From the root directory:

```bash
python3 retrieval-diagnostics/scripts/retrieval_diagnostics.py
```

## Outputs

- `../artifacts/retrieval-diagnostics/sc_retrieval_artifacts.json`
- `../artifacts/retrieval-diagnostics/mc_retrieval_artifacts.json`
- `outputs/**/*.csv`

## Notes
- `../artifacts/retrieval-diagnostics/*.json` will be used at a later step.
