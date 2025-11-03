from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import helper_functions, helper_functions_exp
import numpy as np
import json
import argparse
import pickle
from tqdm import tqdm

def main():

    import shap

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", 
                        type=str, 
                        choices=["lime", "shap", "formatter"], 
                        required=True)
    args = parser.parse_args()

    HF_TOKEN = "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"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_TOKEN"] = HF_TOKEN

    FINETUNED_CLASSIFICATION_MODEL = "yash3056/Llama-3.2-1B-imdb"

    # model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_CLASSIFICATION_MODEL, num_labels=2, device_map="auto")
    model = AutoModelForSequenceClassification.from_pretrained(
    FINETUNED_CLASSIFICATION_MODEL,
    num_labels=2,
    device_map=None,   # Disable automatic sharding
    dtype="float32"  # Avoid mixed precision on CPU
)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_CLASSIFICATION_MODEL)

    dataset_train = load_dataset("imdb", split="train")
    dataset_test = load_dataset("imdb", split="test")


    # ===== WORK WITH A SUBSET =====

    subset_size = 10

    
    negative_samples = [i for i, label in enumerate(dataset_train['label']) if label == 0][:subset_size//2]
    positive_samples = [i for i, label in enumerate(dataset_train['label']) if label == 1][:subset_size//2]

    subset_indices = negative_samples + positive_samples
    subset_texts = [dataset_train['text'][i] for i in subset_indices]
    subset_labels = [dataset_train['label'][i] for i in subset_indices]

    print(f"Working with {len(subset_texts)} samples")
    print(f"Positive: {sum(subset_labels)}")
    print(f"Negative: {len(subset_labels) - sum(subset_labels)}")

    # # Create groups on the subset
    # groups = helper_functions.create_similarity_groups(subset_texts, subset_labels, max_features=1000)

    # print(f"\nCreated {len(groups)} groups of 4")
    # print(f"Total instances grouped: {len(groups) * 4}")

    # # Examine a sample group
    # print("\n" + "="*80)
    # print("SAMPLE GROUP:")
    # print("="*80)
    # sample_group = groups[0]
    # print(f"Average similarity: {sample_group['avg_similarity']:.4f}")
    # for i, (label, text) in enumerate(zip(sample_group['labels'], sample_group['texts'])):
    #     label_str = "POSITIVE" if label == 1 else "NEGATIVE"
    #     print(f"\n[{i+1}] {label_str}:")
    #     print(text[:200] + "...")


    probs = helper_functions_exp.predict_with_memory_management(documents=subset_texts, model=model, tokenizer=tokenizer)

    # Get predicted labels (0 = negative, 1 = positive)
    predicted_labels = np.argmax(probs, axis=1)

    # Get confidence scores
    confidence_scores = np.max(probs, axis=1)
    

    if args.exp == 'shap':

        # def shap_predict(texts):
        #     return helper_functions_exp.predict_with_memory_management(
        #         documents=texts, model=model, tokenizer=tokenizer
        #     )

        def shap_predict(texts):
            return helper_functions_exp.predict_fast(
                documents=texts, model=model, tokenizer=tokenizer
            )


        masker_shap = shap.maskers.Text(
            tokenizer=tokenizer,
            mask_token="<pad>", # this is what "deleted/masked" words are replaced with. there may be a more appropriate choice for your task and model (depends on how it was fine-tuned)
            collapse_mask_token=True, # will collapse "This is <pad> <pad>" to "This is <pad>"
            output_type="string"
        )

        explainer_shap = shap.Explainer(
            shap_predict,
            masker_shap,
            algorithm="partition" # see choices at https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
            )

        explainer_shap_random = shap.explainers.other.Random(
            shap_predict,
            masker_shap,
            algorithm="partition" # see choices at https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
            )
        
        os.makedirs('shap/shap_raw', exist_ok=True)
        os.makedirs('shap/shap_random', exist_ok=True)



        n = subset_size  # Number of explanations to generate

        #######################
        # GENERATE EXPLANATIONS
        #######################

        for idx in tqdm(range(n), desc="Generating SHAP explanations"):

            
            # Generate SHAP values for a single sample
            shap_values = explainer_shap([subset_texts[idx]], silent=True)
            
            with open(f'shap/shap_raw/shap_values_{idx}.pkl', 'wb') as f:
                pickle.dump(shap_values, f)
    
            print(f"Saved file 'shap/shap_raw/shap_values_{idx}.pkl'")
        
        ##############################
        # GENERATE RANDOM EXPLANATIONS
        ##############################
        for idx in range(n):
            print(f"\n--- Generating RANDOM SHAP explanation {idx+1}/{n} ---")
            
            # Generate SHAP values for a single sample
            shap_values_random = explainer_shap_random([subset_texts[idx]])
            
            with open(f'shap/shap_random/shap_values_random_{idx}.pkl', 'wb') as f:
                pickle.dump(shap_values_random, f)
    
            print(f"Saved file 'shap/shap_random/shap_values_random_{idx}.pkl'")


        ####################################
        # CONVERT EXPLANATIONS IN NLP FORMAT
        ####################################
        
        # data = {}
        # for idx in range(n):
        #     with open(f'shap/shap_raw/shap_values_{idx}.pkl', 'rb') as f:
        #         shap_values = pickle.load(f)
            
        #     data[str(idx)] = {
        #         'formatted_text': helper_functions_exp.extract_shap_as_text_all(shap_values)[0],
        #         'structured_text': helper_functions_exp.extract_shap_as_structured_text_all(shap_values)[0],
        #         'top_words': helper_functions_exp.extract_top_words_with_scores_all(shap_values)[0]
        #     }

        # with open('shap/shap_converted.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)


    if args.exp == 'lime':

        from lime import lime_text
        from lime.lime_text import LimeTextExplainer
    
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        explainer_lime, predict_fn = helper_functions_exp.lime_explainer(model, tokenizer)

        explanations_lime = []
        for i, sample_text in enumerate(sample_texts):
            if i < 15:
                print(f"\n--- Generating LIME explanation for Sample {i} ---")
                explanation = explainer_lime.explain_instance(
                sample_texts[i],
                predict_fn,
                num_features=15,
                num_samples=1000
                )
                explanations_lime.append(explanation)


    if args.exp == 'formatter':

        shap_formatter = helper_functions_exp.ExplanationFormatter()
        lime_formatter = helper_functions_exp.ExplanationFormatter()

        # Load explanations
        shap_formatter.load_explanations(shap_values, 'shap')
        lime_formatter.load_explanations(explanations_lime, 'lime')

        formatted_text_LIME = lime_formatter.extract_as_text_all(threshold=0.01)
        structured_text_LIME = lime_formatter.extract_as_structured_text_all(threshold=0.01)
        top_words_LIME = lime_formatter.extract_top_words_with_scores_all()

        data = {}

        for idx in range(len(formatted_text_LIME)):
            data[idx + 15] = {
                'formatted_text': formatted_text_LIME[idx],
                'structured_text': structured_text_LIME[idx],
                'top_words': top_words_LIME[idx]
            }

        with open('lime.json', "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # save_dict_with_explanations(filename='datadata.json',
        #                             samples=sample_texts,
        #                             labels=labels_subset,
        #                             predictions=predicted_labels,
        #                             shap=shap_exp,
        #                             lime=lime_exp,
        #                             type_of_explanations=exp_types,
        #                             subset_indices=sorted(subset_indices)
        #                             )

if __name__ == '__main__':
    main()