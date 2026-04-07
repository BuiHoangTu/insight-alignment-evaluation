#!/bin/bash
#SBATCH --partition=insight
#SBATCH --nodes=1
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1000:00:00
#SBATCH --job-name=xalpaca-gen
#SBATCH --output=xalpaca_gen.log

evaluate_model=$1

source ~/micromamba/etc/profile.d/conda.sh

resolved_path="$(realpath "$evaluate_model" 2>/dev/null)" || true
if [[ -n "$resolved_path" ]]; then
  evaluate_model="$resolved_path"
fi

# setup for xalpaca evaluation
# link libcuda.so.1 to libcuda.so
if [ ! -f "$HOME/libcuda_shim/libcuda.so" ]; then
    mkdir -p $HOME/libcuda_shim
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $HOME/libcuda_shim/libcuda.so
fi
export LD_LIBRARY_PATH=$HOME/libcuda_shim:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/libcuda_shim:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH


### MAIN RUN ###
cd xalpaca
output_dir="../checkpoints/x-alpacaeval/${evaluate_model##*/}"

conda activate alignment-eval
bash scripts/batch_inference_for_xalpacaeval.sh \
  dummy_device \
  "bn de en es fr ru sw th zh" \
  dummy_dataset \
  "M0" \
  $evaluate_model \
  "$output_dir"