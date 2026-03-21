#!/bin/bash
#SBATCH --partition=insight
#SBATCH --nodes=1
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=1600:00:00

#SBATCH --job-name=eval
#SBATCH --output=run-debug-xalpaca.log

# (Optional) activate conda if needed
source ~/micromamba/etc/profile.d/conda.sh
conda activate alignment-eval

export CUDA_VISIBLE_DEVICES=1,2
# python lm-eval-way.py --langs en th --debug
# python mkqa-eval.py --langs en th --debug
# python xlsum-eval.py --langs english thai --debug

## setup for xalpaca evaluation
# link libcuda.so.1 to libcuda.so
if [ ! -f "$HOME/libcuda_shim/libcuda.so" ]; then
    mkdir -p $HOME/libcuda_shim
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $HOME/libcuda_shim/libcuda.so
fi
export LD_LIBRARY_PATH=$HOME/libcuda_shim:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/libcuda_shim:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

alpaca_eval evaluate_from_model $(pwd)/xalpaca-configs/lacomsa.yaml \
  --annotators_config=$(pwd)/xalpaca-configs/local-annotators.yaml \
  --max_instances=10


echo "Evaluation is completed."