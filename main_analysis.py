from itertools import product
import os
from helper_analysis import McNemarAnalyzer, quick_analyze, AggregatedAnalyzer

def main():

    analysis_main_folder = 'analysis/temperature_0.1'

    # CoT = ['chain_of_thought_True', 'no_chain_of_thought']
    llm = ['prometheus', 'qwen', 'llama']
    explanation_type = ["shap", "attention", "lime"]
    prompter = ['single', 'pairwise']
    pred_order = ['pos_neg', 'neg_pos']
    explanation_formats = ["baseline", "text_scores", "text_labels", "structured_text_scores", 
                           "structured_text_labels", "top_words_scores", "top_words_labels", 
                           "natural_words", "part_of_speech"]

    for model, explanation, prompt, order, exp_format  in product(llm, explanation_type, prompter, pred_order, explanation_formats):
        subdirectory = f"temperature_0.1/{model}/{explanation}/{prompt}/no_chain_of_thought/{order}"
        os.makedirs(f"{analysis_main_folder}/{subdirectory}", exist_ok=True)

        try:
            quick_analyze(f'test_results/{subdirectory}/{exp_format}.json', 
                                          'classification_model_predictions/test_set/predictions.json',
                                          save_to=f'{analysis_main_folder}/{subdirectory}/results_{exp_format}.json')
        except FileNotFoundError:
            expected_missing = (prompt == 'single' and model == 'prometheus') or \
                            (prompt == 'pairwise' and model != 'prometheus')
            if not expected_missing:
                print(f"Skipping: {subdirectory}/{exp_format}.json (file not found)")
            continue


    analyzer = McNemarAnalyzer(base_path="analysis")

    analyzer.analyze_and_save_all(output_dir="tables/temperature_0.1/per_model")
    analyzer.analyze_and_save_all_aggregated(output_dir="tables/temperature_0.1/per_model_aggregated")
    analyzer.analyze_and_save_all_conservative(output_dir="tables/temperature_0.1/per_model_conservative")

    aggregated_analyzer = AggregatedAnalyzer(analyzer)

    aggregated_analyzer.analyze_and_save_all(output_dir="tables/temperature_0.1/cross_model")
    aggregated_analyzer.analyze_and_save_all_conservative(output_dir="tables/temperature_0.1/cross_model_conservative")


if __name__ == '__main__':
    main()