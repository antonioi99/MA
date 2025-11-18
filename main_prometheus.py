import json
import random
import torch
import re
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from datasets import load_dataset
import helper_functions


def main():

    # ===== WORK WITH A SUBSET =====
    # Take 128 samples (64 positive + 64 negative for balance)
    subset_size = 128

    dataset_train = load_dataset("imdb", split="train")
    dataset_test = load_dataset("imdb", split="test")

    positive_samples = [i for i, label in enumerate(dataset_test['label']) if label == 1][:subset_size//2]
    negative_samples = [i for i, label in enumerate(dataset_test['label']) if label == 0][:subset_size//2]

    subset_indices = positive_samples + negative_samples
    subset_texts = [dataset_test['text'][i] for i in subset_indices]
    subset_labels = [dataset_test['label'][i] for i in subset_indices]

    print(f"Working with {len(subset_texts)} samples")
    print(f"Positive: {sum(subset_labels)}")
    print(f"Negative: {len(subset_labels) - sum(subset_labels)}")

    # Create groups on the subset
    groups = helper_functions.create_similarity_groups(subset_texts, subset_labels, max_features=1000)

    print(f"\nCreated {len(groups)} groups of 4")
    print(f"Total instances grouped: {len(groups) * 4}")

    model_name = "Unbabel/M-Prometheus-3B"
    JSON_FILE_PATH = os.path.join(data_folder, 'datadata.json')

    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    subset_predicted_labels= []
    for n in sorted(subset_indices):
        subset_predicted_labels.append(data[f"{n}"]["prediction"])


    system_template = (
        "You are a helpful assistant. "
        "Your task is to understand and predict what another model will assign as a sentiment score—either 0 (negative) or 1 (positive)—to movie reviews. "
        "The experiment is divided into four phases: "
        "1. Learning phase: You will see the inputs (movie reviews) and the outputs (predicted labels) for 4 samples. "
        "2. Pre-prediction phase: Based on what you have learned, you will try to guess which label the model predicted for the next 4 samples. In this phase, you will see only the movie reviews as text. Based on what you have learned from the learning phase, you need to guess what score was assigned by the classification model. "
        "3. Learning phase with explanations: In this phase, you will again have access to the inputs, outputs, and explanations for 4 samples. The explanations will be provided in different forms and should help you understand how the model works. "
        "4. Post-prediction phase: In this phase, you will guess the labels predicted by the model for 4 instances. You will see only the movie reviews as text. Based on what you have learned from the learning phases, you need to guess what score was assigned by the classification model."
        )
    
    # Setup experiment with similarity groups
    experiment = helper_functions.GetPredictions(JSON_FILE_PATH, system_template, model_name)
    experiment.setup_experiment(
        texts=subset_texts,          # Pass your 128 texts
        labels=subset_predicted_labels,        # Pass your 128 model predictions
        similarity_groups=groups,    # Pass the 32 groups
        num_train_groups=16,         # 16 groups for training
        num_test_groups=16,          # 16 groups for testing
        seed=42
    )

    # Run complete experiment
    results = experiment.run_complete_experiment()


    # Save results
    experiment.save_results()

if __name__ == "__main__":
    main()

    