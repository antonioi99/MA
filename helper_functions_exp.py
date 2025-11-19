import numpy as np
import torch
import gc
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def predict_with_memory_management(documents, model, tokenizer, batch_size=8):
    """
    Memory-efficient prediction function with batching and cleanup
    """
    documents = [str(d) for d in documents]
    all_probs = []


    # Process in smaller batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Prepare inputs
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Move to CPU immediately
            batch_probs = probs.cpu().numpy()
            all_probs.extend(batch_probs)

        # Clear GPU cache after each batch
        del inputs, outputs, logits, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


    return np.array(all_probs)


def predict_with_memory_management_faster(documents, model, tokenizer, batch_size=64):
    """
    Faster prediction function for systems with sufficient memory
    """
    documents = [str(d) for d in documents]
    all_probs = []

    # Process in larger batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Prepare inputs
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,  # Increased from 128 if you need longer texts
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

        # Minimal cleanup - only delete inputs/outputs
        del inputs, outputs

    return np.concatenate(all_probs, axis=0)

def predict_fast(documents, model, tokenizer, batch_size=64):
    """
    Maximum speed version - use only if you have plenty of memory
    """
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
        predictions: List[int],
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
            predictions: List of predictions
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
                "prediction": int(predictions[idx]),
                "shap": shap_formatted,
                "shap_random": shap_random_formatted,  # NEW: random SHAP
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

def type_of_explanations(shap_json):

  for key, value in shap_json.items():
    sub_keys = []
    for sub_key, sub_value in value.items():
      sub_keys.append(sub_key)
    break

  return sub_keys

def save_dict_with_explanations(filename, samples, labels, predictions, shap, lime, type_of_explanations, subset_indices):

    data_dict = {}
    idx = 0

    for element in subset_indices:
        key = str(element)

        # Handle SHAP explanations (always assumed full length)
        shap_expl = {exp: shap[key][exp] for exp in type_of_explanations}

        # Handle LIME explanations (may be missing for IDs >= 12500)
        if key in lime and lime[key]:
            lime_expl = {exp: lime[key][exp] for exp in type_of_explanations}
        else:
            lime_expl = {}

        data_dict[int(element)] = {  # Convert to Python int
            "sample": str(samples[idx]),  # Ensure it's a string
            "label": int(labels[idx]),  # Convert to Python int
            "prediction": int(predictions[idx]),  # Convert to Python int
            "shap": shap_expl,
            "lime": lime_expl
        }
        idx += 1

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)