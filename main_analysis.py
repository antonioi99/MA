import helper_analysis
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--chain_of_thought",
			action ="store_true")
    parser.add_argument("--pred_order",
			type=str,
			choices=["pos_neg", "neg_pos"],
			required=True)

    args = parser.parse_args()

    analysis_folder = 'analysis'
    chain_of_thought_folder = f'chain_of_thought_True' if args.chain_of_thought else 'no_chain_of_thought'
    subdirectory = f"{chain_of_thought_folder}/{args.pred_order}"
    os.makedirs(f"{analysis_folder}/{subdirectory}", exist_ok=True)

    for element in ["baseline", "text_scores", "text_labels", "structured_text_scores", "structured_text_labels", "top_words_scores", "top_words_labels", "natural_words", "part_of_speech"]:
    
        results = helper_analysis.quick_analyze(f'test_results/{subdirectory}/{element}_0_10000.json', 
                                                'classification_model_predictions/test_set/predictions.json',
                                                print_results=True,
                                                save_to=f'{analysis_folder}/{subdirectory}/results_{element}.json')

if __name__ == '__main__':
    main()
