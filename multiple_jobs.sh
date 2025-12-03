#!/bin/bash

#SBATCH --job-name="shap_explanations"
#SBATCH --array=31-36%3
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
# #SBATCH --nodelist=dgx1
#SBATCH --container-mount-home 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

#SBATCH --time=0-7:59:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
#SBATCH --partition=p_csunivie
#SBATCH --requeue

# Calculate actual task ID: 225 + (array_id - 1) * 25
ACTUAL_TASK_ID=$((SLURM_ARRAY_TASK_ID * 200))

python3 --version
df -h
python main_explanations.py --exp formatter --start ${ACTUAL_TASK_ID} --subset_size 400 --set dev