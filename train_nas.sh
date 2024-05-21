#!/bin/bash
# Job names: nas(voxceleb) nas_vc(vietnam-celeb), nas_cc(cn-celeb)
#SBATCH --job-name=nas_vc_15_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
##SBATCH --exclude=eris,apollon,helios
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainNASModel.py --config config_nas.yml
# python3 trainNASModel.py --config config_nas_cn_celeb.yml
python3 trainNASModel.py --config config_nas_vietnam_celeb.yml
# python3 trainNASModel.py --config config_nas_finetuner_vietnam_celeb.yml
# python3 -m pdb trainNASModel.py

conda deactivate