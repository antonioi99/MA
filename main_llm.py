import helper_llm
import argparse
import os


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_format", 
                        type=str,
                        choices=["baseline", "text_scores", "text_labels", "structured_text_scores", "structured_text_labels", "top_words_scores", "top_words_labels"],  
                        required=True)
    parser.add_argument("--data_size",
                        type=int,
                        required=True)
    args = parser.parse_args()

    
    groups_file = "similarity_groups/similarity_groups.json"
    dev_data_file = "explanations4NLP/merged_data/merged_data.json"
    output_folder = 'test_results'
    os.makedirs(output_folder, exist_ok=True)

    if args.explanation_format == 'baseline':
        
        results = helper_llm.test_experiment(
            groups_file=groups_file,
            dev_data_file=dev_data_file,
            num_test_instances=args.data_size,
            use_explanations=False,
            output_file=f"{output_folder}/baseline.json"
        )
    else:
        results_with_exp = helper_llm.test_experiment(
            groups_file=groups_file,
            dev_data_file=dev_data_file,
            num_test_instances=args.data_size,
            use_explanations=True,
            explanation_format=f"{args.explanation_format}",
            output_file=f"{output_folder}/{args.explanation_format}.json"
        )


if __name__ == "__main__":
    main()