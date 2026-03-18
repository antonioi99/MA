#!/bin/bash

#SBATCH --job-name="ma_innocenti"
#SBATCH --array=0-161%6
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx1
#SBATCH --time=0-20:00:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
#SBATCH --container-writable
#SBATCH --requeue


export $(grep -v '^#' .env | xargs)

python3 --version
df -h
python -m spacy download en_core_web_sm
pip install lime accelerate --break-system-packages

EXPLANATION_FORMATS=("baseline" "text_scores" "text_labels" "structured_text_scores" "structured_text_labels" "top_words_scores" "top_words_labels" "natural_words" "part_of_speech")
PRED_ORDERS=("pos_neg" "neg_pos")
EXPLANATIONS=("shap" "lime" "attention")

# Precompute all valid combinations as (llm, prompter) pairs
LLM_PROMPTER_PAIRS=(
    "prometheus pairwise"
    "qwen single"
    "llama single"
)

# Build the full list of combinations
combinations=()
for pair in "${LLM_PROMPTER_PAIRS[@]}"; do
    llm=$(echo $pair | cut -d' ' -f1)
    prompter=$(echo $pair | cut -d' ' -f2)
    for explanation_format in "${EXPLANATION_FORMATS[@]}"; do
        for pred_order in "${PRED_ORDERS[@]}"; do
            for explanation in "${EXPLANATIONS[@]}"; do
                combinations+=("$llm $prompter $explanation_format $pred_order $explanation")
            done
        done
    done
done

# Pick the combination for this task
combo="${combinations[$SLURM_ARRAY_TASK_ID]}"
llm=$(echo $combo | cut -d' ' -f1)
prompter=$(echo $combo | cut -d' ' -f2)
explanation_format=$(echo $combo | cut -d' ' -f3)
pred_order=$(echo $combo | cut -d' ' -f4)
explanation=$(echo $combo | cut -d' ' -f5)

echo "Running job $SLURM_ARRAY_TASK_ID: llm=$llm, prompter=$prompter, explanation_format=$explanation_format, pred_order=$pred_order, explanation=$explanation"

python main_llm.py \
    --explanation_format "$explanation_format" \
    --data_size 10000 \
    --start 0 \
    --pred_order "$pred_order" \
    --max_new_tokens 128 \
    --prompter "$prompter" \
    --llm "$llm" \
    --explanation "$explanation"