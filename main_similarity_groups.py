from datasets import load_dataset, concatenate_datasets
from helper_functions import create_similarity_groups_from_data
import os
import json

def main():


    folder_merged = "explanations/NLP_format/merged_data"
    merged_data = os.path.join(folder_merged, 'merged_data.json')


    # Load test set
    print("Loading test set...")
    dataset_test = load_dataset("antonio4210/imdb-dev-test-split", split="test")


    test_texts = dataset_test['text']

    test_texts = list(test_texts)
    test_ids = list(range(len(test_texts))) 

    folder_similarity_groups = 'similarity_groups'
    os.makedirs(folder_similarity_groups, exist_ok=True)
    output_path = os.path.join(folder_similarity_groups, "similarity_groups.json")

    predictions_json = 'classification_model_predictions/dev_set/predictions.json'

    groups = create_similarity_groups_from_data(json_file_path=merged_data,
                                                predictions_json=predictions_json,
                                                output_json_path=output_path, 
                                                test_texts=test_texts, 
                                                test_ids=test_ids,
                                                max_features=5000)


if __name__ == '__main__':
    main()