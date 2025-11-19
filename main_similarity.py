from datasets import load_dataset, concatenate_datasets
from helper_functions import create_similarity_groups_from_data
import argparse
from helper_functions_exp import merge_json_files_from_folder
import os
import json

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", 
                        type=str, 
                        choices=["merge", "group"], 
                        required=True)

    args = parser.parse_args()

    if args.type == 'merge':

        # Merge all JSON files from a folder
        merged = merge_json_files_from_folder("explanations4NLP")
        
        with open("explanations4NLP/merged_data.json", 'w') as f:
            json.dump(merged, f, indent=2)
        
        print(f"Merged {len(merged)} samples")


    if args.type == 'group':

        # Load test set from IMDB
        print("Loading test set from IMDB...")
        dataset_test = load_dataset("imdb", split="test")
        dataset_test = dataset_test.select(range(7500, 17500))


        test_texts = dataset_test['text']

        test_texts = list(test_texts)
        test_ids = list(range(len(test_texts))) 

        json_path = "explanations4NLP/merged_data.json"
        folder_similarity_groups = 'similarity_groups'
        os.makedirs(folder_similarity_groups, exist_ok=True)
        output_path = os.path.join(folder_similarity_groups, "similarity_groups.json")

        groups = create_similarity_groups_from_data(json_file_path=json_path, 
                                                    output_json_path=output_path, 
                                                    test_texts=test_texts, 
                                                    test_ids=test_ids,
                                                    max_features=5000)


if __name__ == '__main__':
    main()