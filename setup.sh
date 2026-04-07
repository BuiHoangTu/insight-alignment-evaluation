#!/bin/bash
#SBATCH --partition=insight
#SBATCH --nodes=1
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --time=1600:00:00

#SBATCH --job-name=setup
#SBATCH --output=setup.log

source ~/micromamba/etc/profile.d/conda.sh

# create if not exists
if ! conda env list | grep -q "alignment-eval"; then
    conda env create -f env.yml
fi

conda activate alignment-eval

# install m-rouge
cd multilingual_rouge_scoring
pip3 install -r requirements.txt
python3 -m unidic download # for japanese segmentation
pip3 install --upgrade ./
cd ..

# install xalpaca
cd xalpaca
pip3 install -r requirements.txt
cd ..


echo "Setup is completed."
