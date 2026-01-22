#!/bin/bash

#SBATCH --job-name="ma_antonio"
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=40G
# #SBATCH --nodelist=vader
#SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
# #SBATCH --gres=gpu:h100:1
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
# #SBATCH --nodes=1
# #SBATCH --partition=p_csunivie_gres
# #SBATCH --partition=p_csunivie
#SBATCH --requeue
#SBATCH --container-writable

# #SBATCH --begin=now+8hours

export $(grep -v '^#' .env | xargs)

python3 --version
df -h
# python -m spacy download en_core_web_sm

python main_analysis.py
