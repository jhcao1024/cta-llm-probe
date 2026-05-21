# Failure Patterns

## Expected Inputs

- `../data/raw/test_set.csv`
- `../data/released/cta-predictions/*`

## Run

From the root directory:

```bash
python3 failure-patterns/scripts/error_analysis.py
```

## Outputs
- `../artifacts/failure-patterns/unified_df.csv`

## Notes
- `../artifacts/failure-patterns/unified_df.csv` will be used at a later step.
