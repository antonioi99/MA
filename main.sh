#!/bin/bash

#SBATCH --job-name="ma_antonio"
#SBATCH --container-image=/srv/home/users/a12225670cs/MA/loris3+antonio+latest.sqsh
# #SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=dgx1
# #SBATCH --exclude=dgx1
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
# #SBATCH --partition=p_csunivie_gres
# #SBATCH --partition=p_low
#SBATCH --requeue
#SBATCH --container-writable

# #SBATCH --begin=now+1hours

export $(grep -v '^#' .env | xargs)



python3 --version
df -h
# python -m spacy download en_core_web_sm


pip install lime accelerate --break-system-packages

python main_graphs.py