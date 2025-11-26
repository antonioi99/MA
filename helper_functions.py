from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import torch
import gc
import os
import pickle
import random
import json


def predict_fast(documents, model, tokenizer, batch_size=64):

    documents = [str(d) for d in documents]
    all_probs = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=512, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


class ExplanationFormatter:
    """
    Unified formatter for both SHAP and LIME explanations
    Provides consistent formatting methods for both explanation types
    """

    def __init__(self):
        self.explanation_type = None
        self.processed_data = []

    def _extract_lime_data(self, lime_explanations: List) -> List[Tuple[List[str], List[float]]]:
        """
        Extract words and scores from LIME explanations
        """
        extracted_data = []

        for explanation in lime_explanations:
            # Get the list of available class labels (e.g., [0, 1])
            labels = explanation.available_labels()

            # Pick the first one (usually predicted class)
            exp_list = explanation.as_list(label=labels[0])
            lime_scores = dict(exp_list)

            # Get original raw string (call the function) and tokenize
            raw_text = explanation.domain_mapper.indexed_string.raw_string()
            word_list = raw_text.split()

            # Map scores to words
            score_list = [lime_scores.get(word, 0.0) for word in word_list]

            extracted_data.append((word_list, score_list))

        return extracted_data




    def _extract_shap_data(self, shap_values) -> List[Tuple[List[str], List[float]]]:
        """
        Extract words and scores from SHAP explanations
        """
        extracted_data = []

        for i in range(len(shap_values.data)):
            words = list(shap_values.data[i])
            scores = shap_values.values[i]

            if len(scores.shape) > 1:
                scores = scores[:, 1]  # take contribution for class 1

            extracted_data.append((words, list(scores)))

        return extracted_data

    def load_explanations(self, explanations, explanation_type: str):
        """
        Load explanations and prepare them for formatting
        """
        self.explanation_type = explanation_type.lower()

        if self.explanation_type == 'shap':
            self.processed_data = self._extract_shap_data(explanations)
        elif self.explanation_type == 'lime':
            self.processed_data = self._extract_lime_data(explanations)
        else:
            raise ValueError("explanation_type must be 'shap' or 'lime'")

    def extract_as_text_scores(self, threshold: float) -> List[str]:
        """
        Format explanations as text with inline scores
        """
        results = []

        for words, scores in self.processed_data:
            result_parts = []
            for word, score in zip(words, scores):
                if abs(score) < threshold:
                    result_parts.append(word)
                else:
                    sign = "+" if score >= 0 else ""
                    result_parts.append(f"{word}[{sign}{score:.3f}]")

            results.append(" ".join(result_parts))

        return results

    def extract_as_structured_text_scores(self, threshold: float) -> List[str]:
        """
        Format explanations as structured text with positive/negative sections
        """
        results = []

        for words, scores in self.processed_data:
            positive_words = []
            negative_words = []
            neutral_words = []

            for word, score in zip(words, scores):
                if abs(score) < threshold:
                    neutral_words.append(word)
                elif score > 0:
                    positive_words.append(f"{word}[+{score:.3f}]")
                else:
                    negative_words.append(f"{word}[{score:.3f}]")

            result_parts = []
            if positive_words:
                result_parts.append(f"POSITIVE SENTIMENT: {' '.join(positive_words)}")
            if negative_words:
                result_parts.append(f"NEGATIVE SENTIMENT: {' '.join(negative_words)}")
            if neutral_words:
                result_parts.append(f"NEUTRAL: {' '.join(neutral_words)}")

            results.append("\n".join(result_parts))

        return results

    def extract_top_words_scores(self, top_n: int = 20) -> List[str]:
        """
        Extract top N most influential words with their scores
        """
        results = []

        for words, scores in self.processed_data:
            word_scores = list(zip(words, scores))
            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            top_words = word_scores[:top_n]
            result_parts = []
            for word, score in top_words:
                sentiment = "POSITIVE" if score > 0 else "NEGATIVE"
                result_parts.append(f"{word}: {score:.3f} [{sentiment}]")

            results.append("\n".join(result_parts))

        return results
    
    def extract_as_text_labels(self, threshold: float = 0.01) -> List[str]:
        """
        Format explanations as text with POSITIVE/NEGATIVE labels instead of scores
        
        Example: "This[POSITIVE] is[NEGATIVE] a good[POSITIVE] film[POSITIVE]"
        """
        results = []

        for words, scores in self.processed_data:
            result_parts = []
            for word, score in zip(words, scores):
                if abs(score) < threshold:
                    result_parts.append(word)
                else:
                    label = "POSITIVE" if score >= 0 else "NEGATIVE"
                    result_parts.append(f"{word}[{label}]")

            results.append(" ".join(result_parts))

        return results


    def extract_as_structured_text_labels(self, threshold: float = 0.01) -> List[str]:
        """
        Format explanations as structured text with positive/negative sections but WITHOUT scores
        
        Example:
        POSITIVE SENTIMENT: This good film very funny
        NEGATIVE SENTIMENT: is after no Ernest !
        NEUTRAL: . Yet
        """
        results = []

        for words, scores in self.processed_data:
            positive_words = []
            negative_words = []
            neutral_words = []

            for word, score in zip(words, scores):
                if abs(score) < threshold:
                    neutral_words.append(word)
                elif score > 0:
                    positive_words.append(word)
                else:
                    negative_words.append(word)

            result_parts = []
            if positive_words:
                result_parts.append(f"POSITIVE SENTIMENT: {' '.join(positive_words)}")
            if negative_words:
                result_parts.append(f"NEGATIVE SENTIMENT: {' '.join(negative_words)}")
            if neutral_words:
                result_parts.append(f"NEUTRAL: {' '.join(neutral_words)}")

            results.append("\n".join(result_parts))

        return results


    def extract_top_words_labels(self, top_n: int = 20) -> List[str]:
        """
        Extract top N most influential words with POSITIVE/NEGATIVE labels (no scores)
        
        Example:
        !: NEGATIVE
        good: POSITIVE
        no: NEGATIVE
        funny: POSITIVE
        """
        results = []

        for words, scores in self.processed_data:
            word_scores = list(zip(words, scores))
            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            top_words = word_scores[:top_n]
            result_parts = []
            for word, score in top_words:
                sentiment = "POSITIVE" if score > 0 else "NEGATIVE"
                result_parts.append(f"{word}: {sentiment}")

            results.append("\n".join(result_parts))

        return results
    
class ExplanationProcessor:
    """
    Processes SHAP/LIME explanations from pkl files and converts them to JSON format
    """
    
    def __init__(self, formatter: ExplanationFormatter):
        self.formatter = formatter
        self.explanation_types = ['text', 'structured_text', 'top_words']
    
    def load_explanation_from_pkl(self, pkl_path: str):
        """Load a single explanation from a pkl file"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def process_single_explanation(self, explanation, explanation_type: str, threshold: float) -> Dict[str, str]:
        """
        Process a single explanation and return all formatted versions
        
        Returns:
            dict with keys: 'text', 'structured_text', 'top_words'
        """
        # Load explanation into formatter
        self.formatter.load_explanations(explanation, explanation_type)
        
        # Generate all formats
        formatted = {
            'text_scores': self.formatter.extract_as_text_scores(threshold)[0],
            'text_labels': self.formatter.extract_as_text_labels(threshold=threshold)[0],

            'structured_text_scores': self.formatter.extract_as_structured_text_scores(threshold)[0],
            'structured_text_labels': self.formatter.extract_as_structured_text_labels(threshold=threshold)[0],

            'top_words_scores': self.formatter.extract_top_words_scores(top_n=20)[0],
            'top_words_labels': self.formatter.extract_top_words_labels(top_n=20)[0]
            
        }
        
        return formatted
    
    def process_explanations_from_files(
        self,
        shap_pkl_dir: str,
        shap_random_pkl_dir: str,  #for random SHAP
        lime_pkl_dir: Optional[str],  # Can be None
        samples: List[str],
        labels: List[int],
        subset_indices: List[int],
        output_json: str, 
        threshold_real: float,
        threshold_random: float
    ):
        """
        Read SHAP, random SHAP, and LIME pkl files and create the final JSON structure
        
        Args:
            shap_pkl_dir: Directory containing shap_values_*.pkl files
            shap_random_pkl_dir: Directory containing shap_values_random_*.pkl files
            lime_pkl_dir: Directory containing lime_values_*.pkl files (can be None)
            samples: List of sample texts
            labels: List of true labels
            subset_indices: List of indices to process
            output_json: Output JSON file path
        """
        data_dict = {}
        shap_dir = Path(shap_pkl_dir)
        shap_random_dir = Path(shap_random_pkl_dir)
        lime_dir = Path(lime_pkl_dir) if lime_pkl_dir else None
        
        for idx, sample_idx in enumerate(subset_indices):
            # Load SHAP explanation (raw)
            shap_pkl_path = shap_dir / f"shap_values_{sample_idx}.pkl"
            
            if not shap_pkl_path.exists():
                print(f"Warning: SHAP file not found for index {sample_idx}")
                continue
            
            shap_explanation = self.load_explanation_from_pkl(shap_pkl_path)
            shap_formatted = self.process_single_explanation(shap_explanation, 'shap', threshold=threshold_real)
            
            # Load SHAP random explanation
            shap_random_pkl_path = shap_random_dir / f"shap_values_random_{sample_idx}.pkl"
            
            if not shap_random_pkl_path.exists():
                print(f"Warning: SHAP random file not found for index {sample_idx}")
                shap_random_formatted = {}
            else:
                shap_random_explanation = self.load_explanation_from_pkl(shap_random_pkl_path)
                shap_random_formatted = self.process_single_explanation(shap_random_explanation, 'shap', threshold=threshold_random)
            
            # Load LIME explanation (if available)
            lime_formatted = {}
            if lime_dir:
                lime_pkl_path = lime_dir / f"lime_values_{sample_idx}.pkl"
                if lime_pkl_path.exists():
                    lime_explanation = self.load_explanation_from_pkl(lime_pkl_path)
                    lime_formatted = self.process_single_explanation(lime_explanation, 'lime', threshold=threshold_real)
                else:
                    print(f"Warning: LIME file not found for index {sample_idx}")
            
            # Build the data structure
            data_dict[int(sample_idx)] = {
                "sample": str(samples[idx]),
                "label": int(labels[idx]),
                "shap": shap_formatted,
                "shap_random": shap_random_formatted,
                "lime": lime_formatted
            }
        
        # Save to JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        
        print(f"Saved {len(data_dict)} samples to {output_json}")


def extract_shap_as_text_all(shap_values, threshold=0.01):
    """Process all samples and return list of formatted texts"""
    results = []
    for i in range(len(shap_values.data)):
        data = shap_values.data[i]
        values = shap_values.values[i]

        if len(values.shape) > 1:
            values = values[:, 1]

        result_parts = []
        for word, value in zip(data, values):
            if abs(value) < threshold:
                result_parts.append(word)
            else:
                sign = "+" if value >= 0 else ""
                result_parts.append(f"{word}[{sign}{value:.3f}]")

        results.append(" ".join(result_parts))

    return results


def extract_shap_as_structured_text_all(shap_values, threshold=0.01):
    """
    Extract SHAP explanations as structured text with clear positive/negative sections for all samples

    Args:
        shap_values: SHAP values object from explainer
        threshold: Minimum absolute SHAP value to include

    Returns:
        list: List of structured text strings with positive and negative contributions
    """
    results = []

    for i in range(len(shap_values.data)):
        data = shap_values.data[i]
        values = shap_values.values[i]

        # Handle multi-output case
        if len(values.shape) > 1:
            values = values[:, 1]  # Use positive class

        positive_words = []
        negative_words = []
        neutral_words = []

        for word, value in zip(data, values):
            if abs(value) < threshold:
                neutral_words.append(word)
            elif value > 0:
                positive_words.append(f"{word}(+{value:.3f})")
            else:
                negative_words.append(f"{word}({value:.3f})")

        result = []
        if positive_words:
            result.append(f"POSITIVE SENTIMENT: {' '.join(positive_words)}")
        if negative_words:
            result.append(f"NEGATIVE SENTIMENT: {' '.join(negative_words)}")
        if neutral_words:
            result.append(f"NEUTRAL: {' '.join(neutral_words)}")

        results.append("\n".join(result))

    return results


def extract_top_words_with_scores_all(shap_values, top_n=20):
    """
    Extract top N most influential words with their SHAP scores for all samples

    Args:
        shap_values: SHAP values object from explainer
        top_n: Number of top words to return per sample

    Returns:
        list: List of strings with top influential words and scores for each sample
    """
    results = []

    for i in range(len(shap_values.data)):
        data = shap_values.data[i]
        values = shap_values.values[i]

        # Handle multi-output case
        if len(values.shape) > 1:
            values = values[:, 1]

        # Create list of (word, score) tuples and sort by absolute value
        word_scores = list(zip(data, values))
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top N
        top_words = word_scores[:top_n]

        result_parts = []
        for word, score in top_words:
            sentiment = "POSITIVE" if score > 0 else "NEGATIVE"
            result_parts.append(f"{word}: {score:.3f} ({sentiment})")

        results.append("\n".join(result_parts))

    return results

def merge_json_files_from_folder(folder_path: Union[str, Path]) -> Dict[int, dict]:
    """
    Merge all JSON files in a folder into one dictionary, ordered by integer keys.
    
    Args:
        folder_path: Path to folder containing JSON files
        
    Returns:
        Dictionary with integer keys in ascending order
    """
    folder = Path(folder_path)
    merged_data = {}
    
    # Get all JSON files in the folder
    json_files = sorted(folder.glob("explanations*"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {folder_path}")
        return merged_data
    
    print(f"Found {len(json_files)} JSON files")
    
    # Load and merge all JSON files
    for file_path in json_files:
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.update(data)
    
    sorted_data = dict(sorted(
        merged_data.items(),
        key=lambda x: int(x[0])
    ))
    
    return sorted_data


def lime_explainer(model, tokenizer):
    """
    LIME explainer that works directly with the LLaMA model
    """

    from lime import lime_text
    from lime.lime_text import LimeTextExplainer

    def predict_probs(list_of_texts):
        """
        Wrapper function that takes a list of text strings and returns probabilities
        """
        # Handle single string input

        probabilities = predict_with_memory_management(list_of_texts)  # already existing function

        return probabilities

    # Create LIME text explainer
    explainer = lime_text.LimeTextExplainer(
        class_names=['Negative', 'Positive'],
    )

    return explainer, predict_probs


def create_similarity_groups_from_data(json_file_path, predictions_json, output_json_path, test_texts, test_ids, max_features=5000):
    """
    Create similarity groups using test set from IMDB and dev set from JSON file.
    
    Args:
        json_file_path: Path to JSON file containing dev set texts
        predictions_json: Path to JSON file containing dev_set_predictions
        output_json_path: Path to save the output JSON file
        test_texts: texts form the test set
        max_features: Maximum features for TF-IDF vectorizer
    
    Returns:
        Dictionary mapping test_id -> {
            "test_instance": text,
            "dev_group": [list of dev instance IDs],
            "dev_predictions": [list of corresponding dev predictions]
        }
    """
    
    print(f"Test set size: {len(test_texts)}")
    
    # Load dev set from JSON
    print(f"Loading dev set from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    with open(predictions_json, 'r', encoding='utf-8') as f:
        data_predictions = json.load(f)
    
    # Extract dev information
    dev_ids = []
    dev_texts = []
    dev_predictions = []
    
    for sample_idx, data in dev_data.items():
        dev_ids.append(int(sample_idx))
        dev_texts.append(data['sample'])
    for sample_idx, data in data_predictions.items():
        dev_predictions.append(data['prediction'])
    
    print(f"Dev set size: {len(dev_texts)}")
    
    # Separate dev set by predictions
    dev_positive_indices = [i for i, pred in enumerate(dev_predictions) if pred == 1]
    dev_negative_indices = [i for i, pred in enumerate(dev_predictions) if pred == 0]
    
    print(f"Dev positive predictions: {len(dev_positive_indices)}, Dev negative predictions: {len(dev_negative_indices)}")
    
    # Compute TF-IDF on combined corpus (test + dev)
    print("\nComputing TF-IDF vectors...")
    all_texts = test_texts + dev_texts
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    all_tfidf = vectorizer.fit_transform(all_texts)
    
    # Split back into test and dev
    test_tfidf = all_tfidf[:len(test_texts)]
    dev_tfidf = all_tfidf[len(test_texts):]
    
    # Compute similarity between test and dev
    print("Computing cosine similarity between test and dev sets...")
    test_dev_similarity = cosine_similarity(test_tfidf, dev_tfidf)
    
    # Create groups
    print("Creating groups for each test instance...")
    result_dict = {}
    
    # Load existing results if file exists
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
        print(f"Loaded existing results from {output_json_path} with {len(result_dict)} instances")
    else:
        result_dict = {}
    
    for test_idx in tqdm(range(len(test_texts)), desc="Processing test instances"):
        # Skip if already processed
        if str(test_ids[test_idx]) in result_dict:
            continue
            
        # Get similarities to all dev instances
        similarities = test_dev_similarity[test_idx]
        
        # Find top 2 most similar positive dev instances
        pos_similarities = [(dev_positive_indices[i], similarities[dev_positive_indices[i]]) 
                           for i in range(len(dev_positive_indices))]
        pos_similarities.sort(key=lambda x: x[1], reverse=True)
        top_2_positive = [idx for idx, _ in pos_similarities[:2]]
        
        # Find top 2 most similar negative dev instances
        neg_similarities = [(dev_negative_indices[i], similarities[dev_negative_indices[i]]) 
                           for i in range(len(dev_negative_indices))]
        neg_similarities.sort(key=lambda x: x[1], reverse=True)
        top_2_negative = [idx for idx, _ in neg_similarities[:2]]
        
        # Combine and randomize order
        dev_group_indices = top_2_positive + top_2_negative
        random.shuffle(dev_group_indices)
        
        # Map to actual dev IDs and predictions
        dev_group_ids = [dev_ids[i] for i in dev_group_indices]
        dev_group_predictions = [dev_predictions[i] for i in dev_group_indices]
        
        # Store in result dictionary
        result_dict[test_ids[test_idx]] = {
            "test_instance": test_texts[test_idx],
            "dev_group": dev_group_ids,
            "dev_predictions": dev_group_predictions
        }
        
        # Save to JSON after each instance
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    print(f"\nCompleted! Created groups for {len(result_dict)} test instances")
    print(f"Results saved to {output_json_path}")
    return result_dict
