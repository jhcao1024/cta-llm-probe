# Binary Probing

## Expected Inputs

- `../artifacts/failure-patterns/unified_df.csv`
- `../artifacts/retrieval-diagnostics/sc_retrieval_artifacts.json`
- `../artifacts/retrieval-diagnostics/mc_retrieval_artifacts.json`

## Run

From the root directory, run API-based probing:

```bash
# OPENAI_API_KEY must be set
python3 binary-probing/scripts/prepare_probing_data.py
python3 binary-probing/scripts/run_probing.py
```

Or skip the API calls, as the probing results are released under `../data/released/binary-probing/`.


Run the analysis script to save the derived summary tables to `./outputs/*.csv`
```bash
python3 binary-probing/scripts/analysis.py
```

## Notes
- `run_probing.py` writes probing outputs to `../data/released/binary-probing/`.