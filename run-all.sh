#!/bin/bash

evaluate_model=$1
output_dir=${2:-results}

mkdir -p "$output_dir"

echo "Evaluate $evaluate_model to $output_dir"

# if model is a path, get the last part as the model alias
model_alias=$(basename "$evaluate_model")

jid1=$(sbatch --output=lm_${model_alias}.log -- run-lm-eval.sh "$evaluate_model" "$output_dir" | awk '{print $4}')
jid2=$(sbatch --output=mkqa_${model_alias}.log -- run-mkqa.sh "$evaluate_model" "$output_dir" | awk '{print $4}')
jid3=$(sbatch --output=xlsum_${model_alias}.log -- run-xlsum.sh "$evaluate_model" "$output_dir" | awk '{print $4}')
jid4=$(sbatch --output=alpaca_${model_alias}_gen.log -- run-xalpaca-gen.sh "$evaluate_model"| awk '{print $4}')

echo "Submitted jobs:"
echo "LM:     $jid1"
echo "MKQA:   $jid2"
echo "XLSum:  $jid3"
echo "XAlpaca generate: $jid4"