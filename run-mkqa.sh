#!/bin/bash
#SBATCH --partition=insight
#SBATCH --nodes=1
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1000:00:00
#SBATCH --job-name=mkqa_eval
#SBATCH --output=mkqa_eval.log

evaluate_model=$1
output_dir=$2

source ~/micromamba/etc/profile.d/conda.sh

# export PYTHONFAULTHANDLER=1
# export TRANSFORMERS_VERBOSITY=info
# export LIGTHEVAL_LOG_LEVEL=DEBUG

conda run -n alignment-eval python light-eval-way.py \
  --model_name "$evaluate_model" \
  --output_path "$output_dir"