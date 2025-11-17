from datasets import load_dataset, concatenate_datasets
from helper_functions import create_similarity_groups_from_data
import os

def main():

    # Load test set from IMDB
    print("Loading test set from IMDB...")
    dataset_test = load_dataset("imdb", split="test")
    dataset_test = concatenate_datasets([
        dataset_test.select(range(7500, 7510)),
        dataset_test.select(range(12500, 12510))
    ])


    test_texts = dataset_test['text']

    json_path = "explanations4NLP/explanations_0_150.json"
    folder_similarity_groups = 'similarity_groups'
    os.makedirs(folder_similarity_groups, exist_ok=True)
    output_path = os.path.join(folder_similarity_groups, "similarity_groups.json")

    groups = create_similarity_groups_from_data(json_file_path=json_path, 
                                                output_json_path=output_path, 
                                                test_texts=test_texts, 
                                                max_features=5000)


if __name__ == '__main__':
    main()