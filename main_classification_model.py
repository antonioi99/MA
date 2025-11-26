from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets
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
    
    dataset_test = load_dataset("imdb", split="test")

    # Load dataset
    if args.split == 'dev':
        dataset_dev = concatenate_datasets([
            dataset_test.select(range(0, 7500)),
            dataset_test.select(range(17500, 25000))
        ])
        true_labels = dataset_dev["label"]
        tokenized_dataset = dataset_dev.map(tokenize_function, batched=True)


    if args.split == 'test':
        dataset_test = dataset_test.select(range(7500, 17500))
        true_labels = dataset_test["label"]
        tokenized_dataset = dataset_test.map(tokenize_function, batched=True)
    

    
    
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
    
    
    # Create results dictionary
    results = {}
    for idx, (prediction, label) in enumerate(zip(predictions, true_labels)):
        test_idx = idx  # Adjust index to match original dataset
        results[test_idx] = {
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
    print(f"Results saved to 'predictions.json'")
    


if __name__ == '__main__':
    main()