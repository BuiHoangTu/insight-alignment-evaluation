#!/bin/bash
#SBATCH --partition=insight
#SBATCH --nodes=1
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1000:00:00
#SBATCH --job-name=lm_eval
#SBATCH --output=lm_eval.log

evaluate_model=$1
output_dir=$2

~/.micromamba/bin/micromamba run -n alignment-eval python lm-eval-way.py \
  --model_name "$evaluate_model" \
  --output_path "$output_dir"