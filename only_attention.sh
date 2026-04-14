#!/bin/bash

#SBATCH --job-name="ma_innocenti_formatter"
#SBATCH --array=0-99
#SBATCH --container-image=/srv/home/users/a12225670cs/MA/loris3+antonio+latest.sqsh
#SBATCH --container-mount-home 
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8
# #SBATCH --exclude=dgx1
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
#SBATCH --partition=p_low
#SBATCH --container-writable
#SBATCH --requeue

export $(grep -v '^#' .env | xargs)

python -m spacy download en_core_web_sm
pip install lime accelerate --break-system-packages

SUBSET_SIZE=150
START=$(( SLURM_ARRAY_TASK_ID * SUBSET_SIZE ))

echo "Running formatter job $SLURM_ARRAY_TASK_ID: start=$START, subset_size=$SUBSET_SIZE"

python main_explanations.py \
    --type formatter \
    --start $START \
    --subset_size $SUBSET_SIZE