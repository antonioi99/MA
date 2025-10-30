from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import helper_functions, helper_functions_exp
import numpy as np
import shap
import json
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

def main():

    HF_TOKEN = "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"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_TOKEN"] = HF_TOKEN

    FINETUNED_CLASSIFICATION_MODEL = "yash3056/Llama-3.2-1B-imdb"

    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_CLASSIFICATION_MODEL, num_labels=2, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_CLASSIFICATION_MODEL)

    dataset_train = load_dataset("imdb", split="train")
    dataset_test = load_dataset("imdb", split="test")

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset_train = dataset_train.map(preprocess, batched=True, num_proc=5)
    tokenized_dataset_test = dataset_test.map(preprocess, batched=True, num_proc=5)

    # ===== WORK WITH A SUBSET =====
    # Take 128 samples (64 positive + 64 negative for balance)
    subset_size = 128

    positive_samples = [i for i, label in enumerate(dataset_test['label']) if label == 1][:subset_size//2]
    negative_samples = [i for i, label in enumerate(dataset_test['label']) if label == 0][:subset_size//2]

    subset_indices = positive_samples + negative_samples
    subset_texts = [dataset_test['text'][i] for i in subset_indices]
    subset_labels = [dataset_test['label'][i] for i in subset_indices]

    print(f"Working with {len(subset_texts)} samples")
    print(f"Positive: {sum(subset_labels)}")
    print(f"Negative: {len(subset_labels) - sum(subset_labels)}")

    # Create groups on the subset
    groups = helper_functions.create_similarity_groups(subset_texts, subset_labels, max_features=1000)

    print(f"\nCreated {len(groups)} groups of 4")
    print(f"Total instances grouped: {len(groups) * 4}")

    # Examine a sample group
    print("\n" + "="*80)
    print("SAMPLE GROUP:")
    print("="*80)
    sample_group = groups[0]
    print(f"Average similarity: {sample_group['avg_similarity']:.4f}")
    for i, (label, text) in enumerate(zip(sample_group['labels'], sample_group['texts'])):
        label_str = "POSITIVE" if label == 1 else "NEGATIVE"
        print(f"\n[{i+1}] {label_str}:")
        print(text[:200] + "...")

    sample_texts = [dataset_test["text"][idx] for idx in subset_indices]
    labels_subset = [dataset_test["label"][idx] for idx in subset_indices]

    probs = helper_functions_exp.predict_with_memory_management(sample_texts)

    # Get predicted labels (0 = negative, 1 = positive)
    predicted_labels = np.argmax(probs, axis=1)

    # Get confidence scores
    confidence_scores = np.max(probs, axis=1)

    masker_shap = shap.maskers.Text(
        tokenizer=tokenizer,
        mask_token="<pad>", # this is what "deleted/masked" words are replaced with. there may be a more appropriate choice for your task and model (depends on how it was fine-tuned)
        collapse_mask_token=True, # will collapse "This is <pad> <pad>" to "This is <pad>"
        output_type="string"
    )


    explainer_shap = shap.Explainer(
        helper_functions_exp.predict_with_memory_management,
        shap.maskers.Text(tokenizer),
        algorithm="partition" # see choices at https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
        )

    shap_formatter = helper_functions_exp.ExplanationFormatter()

    shap_values = explainer_shap(sample_texts[:64])

    formatted_text_SHAP = helper_functions_exp.extract_shap_as_text_all(shap_values)
    structured_text_SHAP = helper_functions_exp.extract_shap_as_structured_text_all(shap_values)
    top_words_SHAP = helper_functions_exp.extract_top_words_with_scores_all(shap_values)

    data = {}

    for idx, sample in enumerate(sample_texts[:64]):
        data[idx] = {
            'formatted_text': formatted_text_SHAP[idx],
            'structured_text': structured_text_SHAP[idx],
            'top_words': top_words_SHAP[idx]
        }

    with open('shap.json', "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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

    lime_formatter = helper_functions_exp.ExplanationFormatter()

    # Load explanations
    # shap_formatter.load_explanations(shap_values, 'shap')
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