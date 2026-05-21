# LLM-Based CTA Diagnostics

This repository contains code and artifacts for a diagnostic study of failures of LLM-based Column Type Annotation.

The repository is organized as follows:

- `data/`
  - raw upstream data & released LLM/API-produced outputs
- `artifacts/`
  - placeholder for intermediate files between stages; empty by default in the released repo
- `failure-patterns/`
  - preliminary error analysis
- `representation-bias/`
  - embedding-space diagnostics
- `retrieval-diagnostics/`
  - retrieval analysis
- `binary-probing/`
  - binary-choice probing experiments

## Environment Setup

From the root directory:

```bash
conda env create -f environment.yml
conda activate cta-llm-probe
export OPENAI_API_KEY=<Your API KEY>
```

## Data

See: [data/README.md](https://github.com/jhcao1024/cta-llm-probe/blob/main/data/README.md)

## Pipeline

Run commands from the root directory after activating the conda environment. For some experiments, you may need to set `OPENAI_API_KEY`.

### 1. Failure Patterns

```bash
python3 failure-patterns/scripts/error_analysis.py
```

### 2. Representation Bias

```bash
python3 representation-bias/scripts/main.py
```

### 3. Retrieval Diagnostics

```bash
python3 retrieval-diagnostics/scripts/retrieval_diagnostics.py
```

### 4. Binary-Choice Probing

```bash
python3 binary-probing/scripts/run_probing.py
```