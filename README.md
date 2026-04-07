# Alignment Evaluation Suite

This repository contains evaluation code and scripts for multilingual alignment and preference evaluation across several tasks, including language-model evaluation, multilingual question answering, summarization, and Alpaca-style instruction evaluation.

<!-- ## Key components

- `lm-eval-way.py`: Runs language-model evaluation using `lm-eval`.
- `light-eval-way.py`: Runs multilingual question answering evaluation using `lighteval` / MKQA.
- `xlsum-eval.py`: Runs multilingual summarization evaluation.
- `convert-score.py`: Converts JSON evaluation outputs into a pandas DataFrame for analysis.
- `run-all.sh`: Submits all four evaluation jobs (LM, MKQA, XLSum, Alpaca) via SLURM.
- `run-lm-eval.sh`, `run-mkqa.sh`, `run-xlsum.sh`, `run-alpaca.sh`: SLURM job wrappers for individual evaluation tasks.
- `setup.sh`: Creates the `alignment-eval` conda environment and installs required dependencies.
- `xalpaca/`: Contains Alpaca evaluation support and related configs.
- `multilingual_rouge_scoring/`: Provides local ROUGE scoring utilities used by the summarization pipeline.
- `checkpoints/`: Stores model checkpoint and reference directories used for evaluation.
- `results/` and `results-org-model/`: Default output directories for evaluation results. -->

## Installation

Recommended setup uses the provided Conda environment specification. **Remember to change the conda path in bash files**.

*This is a Slurm file*

```bash
bash setup.sh
```

## Usage
**Remember to change the conda path in bash file of each task.**

### Run all evaluations

`run-all.sh` submits all four evaluation jobs for a model to Slurm.

```bash
./run-all.sh <model_path_or_name> [output_dir]
```

Example:

```bash
./run-all.sh /path/to/model results
```

### Run a single evaluation task

*These are Slurm files*

- LM evaluation:
  ```bash
  ./run-lm-eval.sh <model_path_or_name> [output_dir]
  ```
- MKQA evaluation:
  ```bash
  ./run-mkqa.sh <model_path_or_name> [output_dir]
  ```
- XLSum evaluation:
  ```bash
  ./run-xlsum.sh <model_path_or_name> [output_dir]
  ```
- Alpaca generation:
  ```bash
  ./run-xalpaca-gen.sh <model_path_or_name>
  ```
  the generate texts are saved to checkpoints/x-alpacaeval/model_name

### Convert results for analysis

```bash
python convert-score.py [input_dir]
```
`input_dir` is the above `output_dir`

## Repository structure

- `checkpoints/`: Checkpoint and reference data directories used by evaluation scripts.
- `legacy/`: code that might works but I found better ways.
- `multilingual_rouge_scoring/`: ROUGE score proposed by xlsum.
- `xalpaca/`: Alpaca multilingual generation.
- `results/`: Default output directorie for evaluation runs.

## Missing 
- Alpaca score from generated text