import helper_llm
import argparse
import os


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_format", 
                        type=str,
                        choices=["baseline", "text_scores", "text_labels", "structured_text_scores", "structured_text_labels", "top_words_scores", "top_words_labels", "natural_words", "part_of_speech"],  
                        required=True)
    parser.add_argument("--data_size",
                        type=int,
                        required=True)
    parser.add_argument("--chain_of_thought",
                        action='store_true',  # Present = True, Absent = False
                        help="Include chain of thought reasoning")
    parser.add_argument("--start",
                        type=int,
                        required=True)
    parser.add_argument("--pred_order", 
                        type=str,
                        choices=["pos_neg", "neg_pos"],  
                        required=True)
    parser.add_argument("--max_new_tokens",
                        type=int,
                        required=True)
    parser.add_argument("--prompter",
                        type=str,
                        choices=["single", "pairwise"],
                        required=True)
    parser.add_argument("--llm",
                        type=str,
                        required=True)
    args = parser.parse_args()

    if args.llm == 'prometheus':
        model_name = "Unbabel/M-Prometheus-3B"
        prompt_folder = f'{args.prompter}'
    
    groups_file = "similarity_groups/similarity_groups.json"
    dev_data_file = "explanations4NLP/merged_data/merged_data.json"
    dev_data_predictions = "classification_model_predictions/dev_set/predictions.json"
        
    main_folder = 'test_results'
    model_folder = f'{args.llm}'
    chain_of_thought_folder = f'chain_of_thought_True' if args.chain_of_thought else 'no_chain_of_thought'
    pred_order_folder = args.pred_order

    dir_results = f'{main_folder}/{model_folder}/{prompt_folder}/{chain_of_thought_folder}/{pred_order_folder}'
    os.makedirs(dir_results, exist_ok=True)

    end = args.start + args.data_size
    output_file = f'{dir_results}/{args.explanation_format}_{args.start}_{end}.json'  


    results = helper_llm.test_experiment(
        groups_file=groups_file,
        dev_data_file=dev_data_file,
        dev_data_predictions=dev_data_predictions,
        num_test_instances=args.data_size,
        pred_order=args.pred_order,
        start=args.start,
        chain_of_thought=args.chain_of_thought,
        use_explanations=False if args.explanation_format == 'baseline' else True,
        explanation_format=f"{args.explanation_format}",
        output_file=output_file,
        model_name=model_name,
        max_new_tokens=args.max_new_tokens,
        llm_prompter=args.prompter
    )


if __name__ == "__main__":
    main()