from datasets import load_dataset, concatenate_datasets
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import helper_functions
import numpy as np
import json
import torch
import argparse
import pickle
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", 
                        type=str, 
                        choices=["lime", "shap", "formatter"], 
                        required=True)
    parser.add_argument("--subset_size",
                        type=int,
                        required=True)
    parser.add_argument("--start",
                        type=int,
                        required=True)
    parser.add_argument("--set",
                        type=str,
                        choices=["dev", "train"],
                        required=True)
    parser.add_argument("--only_positive",
                        type=bool,
                        required=False)
    args = parser.parse_args()



    FINETUNED_CLASSIFICATION_MODEL = "yash3056/Llama-3.2-1B-imdb"


    model = AutoModelForSequenceClassification.from_pretrained(
    FINETUNED_CLASSIFICATION_MODEL,
    num_labels=2,
    device_map="auto",   
    dtype=torch.float16  
    )
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_CLASSIFICATION_MODEL)

    dataset_train = load_dataset("imdb", split="train")
    dataset_test = load_dataset("imdb", split="test")

    dataset_dev = concatenate_datasets([
        dataset_test.select(range(0, 7500)),
        dataset_test.select(range(17500, 25000))
    ])

    dataset_test = dataset_test.select(range(7500, 17500))


    # ===== WORK WITH A SUBSET =====

    subset_size = args.subset_size
    start = args.start
    end = start + subset_size if args.only_positive else subset_size // 2 + start

    if args.set == 'train':
        dataset_split = dataset_train
    if args.set == 'dev':
        dataset_split = dataset_dev
    
    negative_samples = [i for i, label in enumerate(dataset_split['label']) if label == 0][start:end]
    positive_samples = [i for i, label in enumerate(dataset_split['label']) if label == 1][start:end]

    subset_indices = positive_samples if args.only_positive else negative_samples + positive_samples
    subset_texts = [dataset_split['text'][i] for i in subset_indices]
    subset_labels = [dataset_split['label'][i] for i in subset_indices]

    print(f"Working with {len(subset_texts)} samples")
    print(f"Positive: {sum(subset_labels)}")
    print(f"Negative: {len(subset_labels) - sum(subset_labels)}")

    

    if args.exp == 'shap':

        # def shap_predict(texts):
        #     return helper_functions.predict_with_memory_management(
        #         documents=texts, model=model, tokenizer=tokenizer
        #     )

        import shap

        print(f"\n\nGenerating explanations with idx in range {(start)} - {end}")

        def shap_predict(texts):
            return helper_functions.predict_fast(
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
            algorithm="partition" # see choices at https://shap.readthedocs.io/en/latest/generated/shap.explainers.other.Random.html
            )
        
        folder_raw = f'shap/{args.set}_set/shap_raw'
        folder_random = f'shap/{args.set}_set/shap_random'
        os.makedirs(folder_raw, exist_ok=True)
        os.makedirs(folder_random, exist_ok=True)


        #######################
        # GENERATE EXPLANATIONS
        #######################

        for idx in tqdm(range(subset_size), desc="Generating SHAP explanations"):
            # Generate SHAP values for a single sample
            shap_values = explainer_shap([subset_texts[idx]], silent=True)
            shap_values_random = explainer_shap_random([subset_texts[idx]], silent=True)
            
            # Save both versions
            raw_path = f'{folder_raw}/shap_values_{subset_indices[idx]}.pkl'
            random_path = f'{folder_random}/shap_values_random_{subset_indices[idx]}.pkl'
            
            with open(raw_path, 'wb') as f:
                pickle.dump(shap_values, f)
            with open(random_path, 'wb') as f:
                pickle.dump(shap_values_random, f)
            
            tqdm.write(f"Saved file '{raw_path}'")
            tqdm.write(f"Saved file '{random_path}'")


    if args.exp == 'lime':
            
        print(f"\n\nGenerating LIME explanations with idx in range {start} - {end}")
        
        explainer_lime, predict_fn = helper_functions.lime_explainer(model, tokenizer)

        folder_raw = f'lime/{args.set}_set/lime_raw'
        os.makedirs(folder_raw, exist_ok=True)

        #######################
        # GENERATE EXPLANATIONS
        #######################

        for idx in tqdm(range(subset_size), desc="Generating LIME explanations"):
            # Generate LIME explanation for a single sample
            explanation = explainer_lime.explain_instance(
                subset_texts[idx],
                predict_fn,
                num_features=len(subset_texts[idx].split()),  # Use all words as features
                num_samples=1000
            )
            
            # Save explanation
            raw_path = f'{folder_raw}/lime_explanation_{subset_indices[idx]}.pkl'
            
            with open(raw_path, 'wb') as f:
                pickle.dump(explanation, f)
            
            tqdm.write(f"Saved file '{raw_path}'")



    if args.exp == 'formatter':

        formatter = helper_functions.ExplanationFormatter()
        processor = helper_functions.ExplanationProcessor(formatter)


        folder_explanations_converted = f'explanations4NLP'
        file_explanations_def = os.path.join(folder_explanations_converted, f'explanations_{start}_{end}.json')
        os.makedirs(folder_explanations_converted, exist_ok=True)
        
        # Process and save
        processor.process_explanations_from_files(
            shap_pkl_dir=f"shap/{args.set}_set/shap_raw",
            shap_random_pkl_dir=f"shap/{args.set}_set/shap_random",
            lime_pkl_dir=None,  # or None if no LIME
            samples=subset_texts,
            labels=subset_labels,
            subset_indices=subset_indices,
            output_json=file_explanations_def,
            threshold_real=0.01,
            threshold_random=0.001
        )


if __name__ == '__main__':
    main()