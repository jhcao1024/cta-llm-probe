# Representation Bias

## Expected Inputs

- `../data/raw/test_set.csv`
- `../artifacts/failure-patterns/unified_df.csv`

## Run

Prerequisite:

- `OPENAI_API_KEY` must be set for embedding generation

From the root directory:

```bash
python3 representation-bias/scripts/main.py
```

## Outputs
- `../artifacts/representation-bias/embeddings/*`
- `../artifacts/representation-bias/2c_hard_case_cluster_proximity.csv`
- `./figures/*`

## Notes
- `../artifacts/representation-bias/embeddings/` will be used at a later step.