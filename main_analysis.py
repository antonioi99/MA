import helper_analysis
import os

def main():
    
    analysis_folder = 'analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    for element in ["baseline", "text_scores", "text_labels", "structured_text_scores", "structured_text_labels", "top_words_scores", "top_words_labels"]:
    
        results = helper_analysis.quick_analyze(f'test_results/{element}.json', 
                                                'classification_model_predictions/test_set/predictions.json',
                                                print_results=True,
                                                save_to=f'{analysis_folder}/results_{element}.json')

if __name__ == '__main__':
    main()