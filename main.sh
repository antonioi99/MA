#!/bin/bash

#SBATCH --job-name="shap_explanations"
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
#SBATCH --time=0-7:59:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
# #SBATCH --partition=p_csunivie_gres
#SBATCH --partition=p_csunivie
#SBATCH --requeue

python3 --version
df -h
python main_similarity.py --type group