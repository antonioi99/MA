from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--split",
                        type=str,
                        choices=['dev', 'test'],
                        required=True)
    parser.add_argument("--dataset",
                        type=str,
                        default="antonio4210/imdb-dev-test-split")
    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available. This script requires a GPU to run.")
    
    device = torch.device("cuda")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("yash3056/Llama-3.2-1B-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("yash3056/Llama-3.2-1B-imdb")
    
    # Move model to GPU
    model.to(device)
    model.eval()

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Load the requested split directly from your personal dataset
    dataset = load_dataset(args.dataset, split=args.split)
    true_labels = dataset["label"]
    doc_ids = dataset["id"]
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=16)
    
    # Predict labels
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy().tolist())
    
    # Create results dictionary keyed by MD5 document ID
    results = {}
    for doc_id, prediction, label in zip(doc_ids, predictions, true_labels):
        results[doc_id] = {
            'prediction': int(prediction),
            'label': int(label)
        }
    
    # Save to JSON file
    dir_predictions = 'classification_model_predictions'
    split_folder = os.path.join(dir_predictions, f'{args.split}_set')
    os.makedirs(split_folder, exist_ok=True)
    json_predictions = os.path.join(split_folder, 'predictions.json')

    with open(json_predictions, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Calculate and print accuracy
    accuracy = sum([1 for pred, true in zip(predictions, true_labels) if pred == true]) / len(predictions)
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to '{json_predictions}'")


if __name__ == '__main__':
    main()