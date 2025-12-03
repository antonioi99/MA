#!/bin/bash

#SBATCH --job-name="shap_explanations"
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=64G
# #SBATCH --nodelist=vader
#SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0-7:59:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
# #SBATCH --partition=p_csunivie_gres
# #SBATCH --partition=p_csunivie
#SBATCH --requeue

# #SBATCH --begin=now+5hours

python3 --version
df -h
python main_explanations.py --exp formatter --subset_size 10 --start 0 --set dev