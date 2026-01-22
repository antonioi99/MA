#!/bin/bash

#SBATCH --job-name="shap_explanations"
#SBATCH --array=0-8%2
#SBATCH --container-image="ghcr.io#loris3/antonio:latest"
#SBATCH --container-mount-home 
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
# #SBATCH --gres=gpu:1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0-10:59:00
#SBATCH --container-workdir=/srv/home/users/a12225670cs/MA
#SBATCH --nodes=1
#SBATCH --partition=p_csunivie_gres
#SBATCH --container-writable
#SBATCH --requeue

# Calculate actual task ID: 225 + (array_id - 1) * 25
# ACTUAL_TASK_ID=$((SLURM_ARRAY_TASK_ID * 200))


FORMATS=("baseline" "text_scores" "text_labels" "structured_text_scores" "structured_text_labels" "top_words_scores" "top_words_labels" "natural_words" "part_of_speech")
FORMAT="${FORMATS[$SLURM_ARRAY_TASK_ID]}"

export $(grep -v '^#' .env | xargs)

python3 --version
df -h
# python -m spacy download en_core_web_sm
# python main_explanations.py --exp formatter --start ${ACTUAL_TASK_ID} --subset_size 400 --set dev
# python main_llm.py --explanation_format "$FORMAT" --data_size 2000 --start 0 --pred_order pos_neg --max_new_tokens 128 --chain_of_thought --llm prometheus --prompter single
# python main_llm.py --explanation_format "$FORMAT" --data_size 8000 --start 2000 --pred_order pos_neg --max_new_tokens 128 --chain_of_thought --llm prometheus --prompter single
# python main_llm.py --explanation_format "$FORMAT" --data_size 10000 --start 0 --pred_order pos_neg --max_new_tokens 50 --prompter single --llm llama

# python main_llm.py --explanation_format "$FORMAT" --data_size 10000 --start 0 --pred_order pos_neg --max_new_tokens 256 --prompter single --llm qwen 


# noch fehlend
# python main_llm.py --explanation_format "$FORMAT" --data_size 10000 --start 0 --pred_order neg_pos --max_new_tokens 256 --prompter pairwise --llm prometheus --chain_of_thought
# python main_llm.py --explanation_format "$FORMAT" --data_size 10000 --start 0 --pred_order neg_pos --max_new_tokens 256 --prompter pairwise --llm prometheus --chain_of_thought