#!/bin/bash

#SBATCH --job-name="ma_innocenti"
#SBATCH --array=0-5%2
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx1
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
# #SBATCH --partition=p_csunivie_gres
#SBATCH --container-writable
#SBATCH --requeue

export $(grep -v '^#' .env | xargs)

python3 --version
df -h
python -m spacy download en_core_web_sm
pip install lime accelerate --break-system-packages

PRED_ORDERS=("pos_neg" "neg_pos")
LLMS=("prometheus")
EXPLANATIONS=("shap" "lime" "attention")

N_PRED_ORDERS=${#PRED_ORDERS[@]}   # 2
N_LLMS=${#LLMS[@]}                 # 1
N_EXPLANATIONS=${#EXPLANATIONS[@]} # 3

TASK_ID=$SLURM_ARRAY_TASK_ID

llm_idx=$((TASK_ID % N_LLMS))
TASK_ID=$((TASK_ID / N_LLMS))

explanation_idx=$((TASK_ID % N_EXPLANATIONS))
TASK_ID=$((TASK_ID / N_EXPLANATIONS))

pred_order_idx=$((TASK_ID % N_PRED_ORDERS))

pred_order="${PRED_ORDERS[$pred_order_idx]}"
llm="${LLMS[$llm_idx]}"
explanation="${EXPLANATIONS[$explanation_idx]}"

echo "Running job $SLURM_ARRAY_TASK_ID: pred_order=$pred_order, llm=$llm, explanation=$explanation"

python main_llm.py \
    --explanation_format baseline \
    --data_size 10000 \
    --start 0 \
    --pred_order "$pred_order" \
    --max_new_tokens 128 \
    --prompter pairwise \
    --llm "$llm" \
    --explanation "$explanation"



