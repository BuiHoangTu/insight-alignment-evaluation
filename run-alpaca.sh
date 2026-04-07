#!/bin/bash
#SBATCH --partition=insight
#SBATCH --nodes=1
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=3
#SBATCH --time=1600:00:00

#SBATCH --job-name=eval
#SBATCH --output=run-alpaca.log

# read args
evaluate_model=$1
if [ -z "$evaluate_model" ]; then
  echo "Usage: $0 <evaluate_model>"
  exit 1
fi

# Read $2 default to "results"
output_dir=${2:-results}

# Create the directory if it does not exist
mkdir -p "$output_dir"
echo "Evaluating model: $evaluate_model and saving results to: $output_dir"

# setup for xalpaca evaluation
link libcuda.so.1 to libcuda.so
if [ ! -f "$HOME/libcuda_shim/libcuda.so" ]; then
    mkdir -p $HOME/libcuda_shim
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $HOME/libcuda_shim/libcuda.so
fi
export LD_LIBRARY_PATH=$HOME/libcuda_shim:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/libcuda_shim:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

# create yaml file for the model
SOURCE_CONFIG=$(pwd)/xalpaca-configs/lacomsa.yaml
CONFIG_FILE="$(mktemp)"
# replace the checkpoint path in the yaml file
sed "s|/insight-fast/hbui/projects/llama3_8b_lacomsa/checkpoint-94|${evaluate_model}|g" "$SOURCE_CONFIG" > "$CONFIG_FILE"

# (Optional) activate conda if needed
source ~/micromamba/etc/profile.d/conda.sh

## run alpaca evaluation
conda run -n alignment-eval alpaca_eval evaluate_from_model $CONFIG_FILE \
  --annotators_config=$(pwd)/xalpaca-configs/local-annotators.yaml \
  --output_path=$output_dir

rm $CONFIG_FILE

echo "Evaluation is completed."