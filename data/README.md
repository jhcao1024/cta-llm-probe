# Data Folder

## Layout

```text
data/
├── README.md
├── raw/
└── released/
    ├── cta-predictions/
    └── binary-probing/
```

## Data

### `raw/`

Please refer to [drafiei/llms-for-cta](https://github.com/drafiei/llms-for-cta) for the raw dataset and code for generating CTA predictions. For this diagnostic project, we release the predictions we obtained by running their code under `data/released`, and our error analysis was performed on these outputs.

### `released/cta-predictions/`

- `single_col_zero_shot.csv`
- `single_col_few_shot.csv`
- `single_col_rag.csv`
- `multi_col_zero_shot.csv`
- `multi_col_few_shot.csv`
- `multi_col_rag.csv`
- `multi_col_cot.csv`

These files are the outputs of the seven LLM-based methods from [drafiei/llms-for-cta](https://github.com/drafiei/llms-for-cta), which we used for our error analytics. For all the few-shot and RAG methods, we provided exactly five examples.

### `released/binary-probing/`

- `probing_results_SC1.csv`
- `probing_results_SC2RAG.csv`
- `probing_results_MC1.csv`
- `probing_results_MC2RAG.csv`

These files are the outputs of the binary-choice probing experiment, for which we provide separate code under `binary-probing`. We release these files because they are used for further analysis.