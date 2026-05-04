from datasets import load_dataset
import os
import hashlib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import helper_functions
import numpy as np
import json
import torch
import argparse
import pickle
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", 
                        type=str, 
                        choices=["lime", "shap", "attention", "formatter", "merge", "visualization_example"], 
                        required=True)
    parser.add_argument("--subset_size",
                        type=int,
                        required=False)
    parser.add_argument("--start",
                        type=int,
                        required=False)
    args = parser.parse_args()

    FINETUNED_CLASSIFICATION_MODEL = "yash3056/Llama-3.2-1B-imdb"

    if args.type not in ["merge"]:
        if args.type not in ["formatter"]:
            model = AutoModelForSequenceClassification.from_pretrained(
                FINETUNED_CLASSIFICATION_MODEL,
                num_labels=2,
                device_map="auto",   
                dtype=torch.float16, 
                output_attentions=True  
            )
            model = model.to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(FINETUNED_CLASSIFICATION_MODEL)


        dataset_split = load_dataset("antonio4210/imdb-dev-test-split", split="dev")
        
        # ===== WORK WITH A SUBSET =====
        start = args.start if args.start is not None else 0
        if args.subset_size is not None:
            end = start + args.subset_size
        else:
            end = len(dataset_split['text'])

        subset_texts = dataset_split['text'][start:end]
        subset_labels = dataset_split['label'][start:end]

        print(f"Working with {len(subset_texts)} samples")
        print(f"Positive: {sum(subset_labels)}")
        print(f"Negative: {len(subset_labels) - sum(subset_labels)}")


    if args.type == 'shap':

        import shap

        print(f"\n\nGenerating SHAP explanations with idx in range {start} - {end}")

        def shap_predict(texts):
            return helper_functions.predict_fast(
                documents=texts, model=model, tokenizer=tokenizer
            )

        masker_shap = shap.maskers.Text(
            tokenizer=tokenizer,
            mask_token="<pad>",
            collapse_mask_token=True, 
            output_type="string"
        )

        explainer_shap = shap.Explainer(
            shap_predict,
            masker_shap,
            algorithm="partition"
        )

        folder_raw = f'explanations/pkl/shap/dev_set/shap_raw'
        os.makedirs(folder_raw, exist_ok=True)

        for idx in tqdm(range(len(subset_texts)), desc="Generating SHAP explanations"):
            doc_hash = hashlib.md5(subset_texts[idx].encode("utf-8")).hexdigest()
            raw_path = f'{folder_raw}/shap_{doc_hash}.pkl'

            if os.path.isfile(raw_path):
                tqdm.write(f"Skipping '{raw_path}' (already exists)")
                continue

            shap_values = explainer_shap([subset_texts[idx]], silent=True)

            with open(raw_path, 'wb') as f:
                pickle.dump(shap_values, f)

            tqdm.write(f"Saved file '{raw_path}'")


    if args.type == 'lime':
            
        print(f"\n\nGenerating LIME explanations with idx in range {start} - {end}")
        
        from lime import lime_text
        predict_fn = helper_functions.lime_explainer(model, tokenizer)

        folder_raw = f'explanations/pkl/lime/dev_set/lime_raw'
        os.makedirs(folder_raw, exist_ok=True)

        for idx in tqdm(range(len(subset_texts)), desc="Generating LIME explanations"):
            doc_hash = hashlib.md5(subset_texts[idx].encode("utf-8")).hexdigest()
            raw_path = f'{folder_raw}/lime_{doc_hash}.pkl'

            if os.path.isfile(raw_path):
                tqdm.write(f"Skipping '{raw_path}' (already exists)")
                continue

            doc_seed = int(doc_hash, 16) % (2**32)
            explainer_lime = lime_text.LimeTextExplainer(
                class_names=['Negative', 'Positive'],
                random_state=doc_seed
            )

            explanation = explainer_lime.explain_instance(
                subset_texts[idx],
                predict_fn,
                num_features=len(subset_texts[idx].split()),
                num_samples=1000
            )

            with open(raw_path, 'wb') as f:
                pickle.dump(explanation, f)

            tqdm.write(f"Saved file '{raw_path}'")


    if args.type == 'attention':
        
        print(f"\n\nGenerating Attention explanations with idx in range {start} - {end}")
        
        folder_raw = f'explanations/pkl/attention/dev_set/attention_raw'
        os.makedirs(folder_raw, exist_ok=True)
        
        for idx in tqdm(range(len(subset_texts)), desc="Generating Attention explanations"):
            doc_hash = hashlib.md5(subset_texts[idx].encode("utf-8")).hexdigest()
            raw_path = f'{folder_raw}/attention_{doc_hash}.pkl'

            if os.path.isfile(raw_path):
                tqdm.write(f"Skipping '{raw_path}' (already exists)")
                continue

            attention_explanation = helper_functions.extract_attention_explanation(
                text=subset_texts[idx],
                model=model,
                tokenizer=tokenizer
            )

            with open(raw_path, 'wb') as f:
                pickle.dump(attention_explanation, f)

            tqdm.write(f"Saved file '{raw_path}'")


    if args.type == 'formatter':

        formatter = helper_functions.ExplanationFormatter()
        processor = helper_functions.ExplanationProcessor(formatter)

        folder_explanations_converted = f'explanations/NLP_format'
        file_explanations_def = os.path.join(folder_explanations_converted, f'explanations_{start}_{end}.json')
        os.makedirs(folder_explanations_converted, exist_ok=True)
        
        processor.process_explanations_from_files(
            shap_pkl_dir=f"explanations/pkl/shap/dev_set/shap_raw",
            shap_random_pkl_dir=f"explanations/pkl/shap/dev_set/shap_random",
            lime_pkl_dir=f"explanations/pkl/lime/dev_set/lime_raw",
            attention_pkl_dir=f"explanations/pkl/attention/dev_set/attention_raw",
            samples=subset_texts,
            labels=subset_labels,
            output_json=file_explanations_def,
            top_n_shap=20,
            top_n_random=20,
            top_n_lime=20,
            top_n_attention=20,
        )


    if args.type == 'merge':

        folder_merged = "explanations/NLP_format/merged_data"
        os.makedirs(folder_merged, exist_ok=True)
        merged_data = os.path.join(folder_merged, 'merged_data.json')

        merged = helper_functions.merge_json_files_from_folder("explanations/NLP_format")
        
        with open(merged_data, 'w') as f:
            json.dump(merged, f, indent=2)
        
        print(f"Merged {len(merged)} samples")


    if args.type == 'visualization_example':
 
        import shap
 
        text  = subset_texts[0]
        label = subset_labels[0]
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
 
        print(f"\nSample index : {start}")
        print(f"True label   : {'Positive' if label == 1 else 'Negative'}")
        print(f"Doc hash     : {doc_hash}")
 
        # ── Load LIME and render with built-in .as_html() ─────────────────────
        # as_html() returns a full self-contained HTML page with highlighted
        # text + a bar chart of the top features.
        lime_path = f"explanations/pkl/lime/dev_set/lime_raw/lime_{doc_hash}.pkl"
        if not os.path.isfile(lime_path):
            raise FileNotFoundError(
                f"LIME file not found: {lime_path}\n"
                f"Run --type lime first with --start {start} --subset_size 1"
            )
        with open(lime_path, 'rb') as f:
            lime_exp = pickle.load(f)
 
        lime_out = f"explanations/html/visualization_example_{start}_lime.html"
        os.makedirs("explanations", exist_ok=True)
        with open(lime_out, 'w', encoding='utf-8') as f:
            f.write(lime_exp.as_html())
        print(f"LIME heatmap saved -> {lime_out}")
 
        # ── Load SHAP and render with built-in shap.plots.text() ──────────────
        # shap.plots.text() renders an interactive token-level heatmap.
        # Passing display=False returns the HTML string instead of
        # opening a browser window.
        shap_path = f"explanations/pkl/shap/dev_set/shap_raw/shap_{doc_hash}.pkl"
        if not os.path.isfile(shap_path):
            raise FileNotFoundError(
                f"SHAP file not found: {shap_path}\n"
                f"Run --type shap first with --start {start} --subset_size 1"
            )
        with open(shap_path, 'rb') as f:
            shap_exp = pickle.load(f)
 
        shap_out = f"explanations/html/visualization_example_{start}_shap.html"
        shap_html = shap.plots.text(shap_exp[0], display=False)
        with open(shap_out, 'w', encoding='utf-8') as f:
            f.write(shap_html)
        print(f"SHAP heatmap saved -> {shap_out}")
 
        print("\nOpen both HTML files in any browser to view the visualisations.")

if __name__ == '__main__':
    main()