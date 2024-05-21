#!/bin/bash
# Job names: nas_search(voxceleb) nas_search_vc(vietnam-celeb), nas_search_cc(cn-celeb)
#SBATCH --job-name=nas_search_cc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --exclude=eris,apollon,helios
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainNASSEARCHModel.py --config config_nas_search.yml
python3 trainNASSEARCHModel.py --config config_nas_search_cn_celeb.yml
# python3 trainNASSEARCHModel.py --config config_nas_search_vietnam_celeb.yml
# python3 trainNASSEARCHModel.py --config config_nas_search_finetuner_vietnam_celeb.yml
# python3 -m pdb trainNASSEARCHModel.py

conda deactivate