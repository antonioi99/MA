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
    parser.add_argument("--prediction_type",
                        type=str,
                        choices=['POSITIVE_NEGATIVE', '0_1'],
                        required=True)
    parser.add_argument("--chain_of_thought",
                        action='store_true',  # Present = True, Absent = False
                        help="Include chain of thought reasoning")
    parser.add_argument("--start",
                        type=int,
                        required=False)
    args = parser.parse_args()

    
    groups_file = "similarity_groups/similarity_groups.json"
    dev_data_file = "explanations4NLP/merged_data/merged_data.json"
        
    main_folder = 'test_results'
    prediction_type_folder = f'{args.prediction_type}'
    chain_of_thought_folder = f'chain_of_thought_True' if args.chain_of_thought else 'no_chain_of_thought'

    dir_results = f'{main_folder}/{prediction_type_folder}/{chain_of_thought_folder}'
    os.makedirs(dir_results, exist_ok=True)

    if args.start:
        end = args.start + args.data_size
        output_file = f'{dir_results}/{args.explanation_format}_{args.start}_{end}.json'  
    else:
        output_file = f'{dir_results}/{args.explanation_format}.json'


    results = helper_llm.test_experiment(
        groups_file=groups_file,
        dev_data_file=dev_data_file,
        num_test_instances=args.data_size,
        start=args.start,
        prediction_type=args.prediction_type,
        chain_of_thought=args.chain_of_thought,
        use_explanations=False if args.explanation_format == 'baseline' else True,
        explanation_format=f"{args.explanation_format}",
        output_file=output_file
    )


if __name__ == "__main__":
    main()