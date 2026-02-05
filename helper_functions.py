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
        try:
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()
        except:
            self.nlp = None
            print("Warning: spaCy model not loaded. Install with: python -m spacy download en_core_web_sm")

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

    def _reconstruct_words_from_llama_shap(self, shap_values):
        """
        Reconstruct original words from LLaMA SHAP tokens.
        
        SHAP returns tokens with regular spaces instead of Ġ markers.
        - Tokens starting with ' ' (space) indicate a new word
        - Tokens without leading space are continuations of the previous word
        
        Args:
            shap_values: SHAP Explanation object for a single sample
            
        Returns:
            Tuple of (words_list, scores_list)
        """
        tokens = shap_values.data[0]
        token_shap_values = shap_values.values[0]
        
        # Handle multi-dimensional scores (for multi-class)
        if len(token_shap_values.shape) > 1:
            token_shap_values = token_shap_values[:, 1]
        
        words = []
        scores = []
        current_word = ""
        current_shaps = []
        
        for i, (token, shap_val) in enumerate(zip(tokens, token_shap_values)):
            # Skip special tokens and empty tokens
            if token in ['<s>', '</s>', '<pad>', '<unk>', '<|begin_of_text|>', '<|end_of_text|>', '']:
                continue
            
            # Check if this starts a new word
            # Tokens starting with a space indicate a new word
            starts_new_word = token.startswith(' ')
            
            # Special case: first real token always starts a new word
            if i == 1 and not starts_new_word:  # i==1 because i==0 is usually empty
                starts_new_word = True
            
            if starts_new_word and current_word:
                # Save the previous word
                avg_shap = np.mean(current_shaps) if current_shaps else 0.0
                words.append(current_word)
                scores.append(avg_shap)
                current_word = ""
                current_shaps = []
            
            # Remove leading space and add to current word
            clean_token = token.lstrip(' ')
            current_word += clean_token
            current_shaps.append(shap_val)
        
        # Don't forget the last word
        if current_word:
            avg_shap = np.mean(current_shaps) if current_shaps else 0.0
            words.append(current_word)
            scores.append(avg_shap)
        
        return words, scores

        

    def _extract_shap_data(self, shap_values) -> List[Tuple[List[str], List[float]]]:
        """
        Extract words and scores from SHAP explanations.
        Reconstructs original words from LLaMA subtokens.
        """
        extracted_data = []


        if isinstance(shap_values, list):
            for explanation in shap_values:
                if hasattr(explanation, 'data'):
                    for i in range(len(explanation.data)):
                        single_shap = type('obj', (object,), {
                            'data': [explanation.data[i]],
                            'values': [explanation.values[i]]
                        })()
                        
                        words, scores = self._reconstruct_words_from_llama_shap(single_shap)
                        extracted_data.append((words, scores))
                else:
                    raise ValueError(f"List element is not a SHAP Explanation object: {type(explanation)}")
        
        elif hasattr(shap_values, 'data'):
            for i in range(len(shap_values.data)):
                single_shap = type('obj', (object,), {
                    'data': [shap_values.data[i]],
                    'values': [shap_values.values[i]]
                })()
                
                words, scores = self._reconstruct_words_from_llama_shap(single_shap)
                extracted_data.append((words, scores))
        
        else:
            raise ValueError(f"Unexpected SHAP values format: {type(shap_values)}")

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



    def extract_as_text(self, brackets, threshold: float = 0.01) -> List[str]:
        """
        Format explanations as text with POSITIVE/NEGATIVE labels instead of scores
        
        Example: "This[POSITIVE] is[NEGATIVE] a good[POSITIVE] film[POSITIVE]"
        """
        results = []

        for words, scores in self.processed_data:

            avg_score = sum(scores) / len(scores) if scores else 0
            prediction = "POSITIVE" if avg_score >= 0 else "NEGATIVE"

            result_parts = []

            for word, score in zip(words, scores):
                if abs(score) < threshold:
                    result_parts.append(word)
                else:
                    if brackets == 'label':
                        label = "POSITIVE" if score >= 0 else "NEGATIVE"
                        result_parts.append(f"{word}[{label}]")
                    elif brackets == 'score':
                        sign = "+" if score >= 0 else ""
                        result_parts.append(f"{word}[{sign}{score:.3f}]")
            
            if brackets == 'score':
                explanation = f"The model predicted {prediction}. Bracketed scores indicate each word's contribution to the overall sentiment (negative scores pull toward negative sentiment, positive scores pull toward positive sentiment).\n\n"
            elif brackets == 'label':
                explanation = f"The model predicted {prediction}. Bracketed labels indicate each word's contribution to the overall sentiment (NEGATIVE pulls toward negative sentiment, POSITIVE pulls toward positive sentiment).\n\n"

            results.append(explanation + " ".join(result_parts))

        return results

    def extract_as_structured_text(self, brackets, threshold: float = 0.01) -> List[str]:
        """
        Format explanations as structured text with positive/negative sections
        """
        results = []

        for words, scores in self.processed_data:

            avg_score = sum(scores) / len(scores) if scores else 0
            prediction = "POSITIVE" if avg_score >= 0 else "NEGATIVE"

            positive_words = []
            negative_words = []
            neutral_words = []

            for word, score in zip(words, scores):
                if abs(score) < threshold:
                    neutral_words.append(word)
                elif score > 0:
                    if brackets == 'score':
                        positive_words.append(f"{word}[+{score:.3f}]")
                    elif brackets == 'label':
                        positive_words.append(word)

                else:
                    if brackets == 'score':
                        negative_words.append(f"{word}[{score:.3f}]")
                    elif brackets == 'label':
                        negative_words.append(word)

            result_parts = []
            if brackets == 'score':
                explanation = f"The model predicted {prediction}. Words are grouped by sentiment: POSITIVE, NEGATIVE, and NEUTRAL. The bracketed scores indicate each word's contribution to the overall sentiment (negative scores pull toward negative sentiment, positive scores pull toward positive sentiment).\n\n"
                result_parts.append(explanation)
            elif brackets == 'label':
                explanation = f"The model predicted {prediction}. Words are grouped by their sentiment contribution: POSITIVE, NEGATIVE, and NEUTRAL.\n\n"
                result_parts.append(explanation)
            
            if positive_words:
                result_parts.append(f"POSITIVE SENTIMENT: {' '.join(positive_words)}")
            if negative_words:
                result_parts.append(f"NEGATIVE SENTIMENT: {' '.join(negative_words)}")
            if neutral_words:
                result_parts.append(f"NEUTRAL: {' '.join(neutral_words)}")

            results.append(" ".join(result_parts))

        return results

    def extract_top_words(self, brackets, top_n: int = 20) -> List[str]:
        """
        Extract top N most influential words with their scores
        """
        results = []

        for words, scores in self.processed_data:

            avg_score = sum(scores) / len(scores) if scores else 0
            prediction = "POSITIVE" if avg_score >= 0 else "NEGATIVE"

            word_scores = list(zip(words, scores))
            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            top_words = word_scores[:top_n]
            result_parts = []
            
            if brackets == 'score':
                explanation = f"The model predicted {prediction}. These are the most influential words for this prediction. Scores show each word's contribution (negative values push toward negative sentiment, positive values push toward positive sentiment):\n\n"
                for word, score in top_words:
                    result_parts.append(f"{word} [{score:+.3f}]\n")
            elif brackets == 'label':
                explanation = f"The model predicted {prediction}. These are the most influential words for this prediction:\n\n"
                for word, score in top_words:
                    sentiment = "POSITIVE" if score > 0 else "NEGATIVE"
                    result_parts.append(f"{word} [{sentiment}]\n")
                    
            results.append(explanation + ''.join(result_parts))

        return results
        


    def extract_as_natural_explanation(self, top_n: int = 5) -> List[str]:
        """
        Format explanations as natural language sentences explaining the prediction
        
        Args:
            top_n: Number of top influential words to include
        """
        import string
        
        # Common conjunctions to filter out
        conjunctions = {
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 
            'although', 'because', 'since', 'unless', 'while',
            'if', 'then', 'than', 'that', 'though', 'whether'
        }
        
        results = []
        
        for idx, (words, scores) in enumerate(self.processed_data):
            # Determine prediction based on average score
            avg_score = sum(scores) / len(scores) if scores else 0
            prediction = "POSITIVE" if avg_score >= 0 else "NEGATIVE"
            
            # Get top influential words for the predicted class
            word_scores = list(zip(words, scores))
            
            # Filter for words that align with the prediction
            if prediction == "POSITIVE":
                relevant_words = [(w, s) for w, s in word_scores if s > 0]
            else:
                relevant_words = [(w, s) for w, s in word_scores if s < 0]
            
            # Filter out punctuation and conjunctions
            filtered_words = [
                (word, score) for word, score in relevant_words
                if word not in string.punctuation 
                and word.lower() not in conjunctions
                and len(word) > 2
            ]
            
            # Sort by absolute score and get top N
            filtered_words.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top words (already reconstructed, no need for fragment reconstruction)
            top_words = []
            seen_words = set()  # Avoid duplicates
            
            for word, _ in filtered_words[:top_n * 2]:  # Get more candidates in case of duplicates
                if len(top_words) >= top_n:
                    break
                
                # Add if not already seen
                if word.lower() not in seen_words:
                    top_words.append(word)
                    seen_words.add(word.lower())
            
            # Format as natural sentence
            if top_words:
                words_str = ", ".join(top_words)
                result = f"The model predicted {prediction} because of the following words: {words_str}"
            else:
                result = f"The model predicted {prediction}"
            
            results.append(result)
        
        return results

    def extract_pos(self, top_n: int = 5, context_window: int = 2, pos = {"ADJ", "VERB", "NOUN", "ADV"}) -> List[str]:
        """
        Extract adjectives, verbs, and nouns with contextual POS tagging
        
        Args:
            top_n: Number of top influential words to include
            context_window: Number of surrounding words to use for POS tagging
            pos: Set of POS tags to include
        """
        if self.nlp is None:
            raise RuntimeError("spaCy model not loaded")
        
        import string
        
        results = []
        
        for idx, (words, scores) in enumerate(self.processed_data):
            # Determine prediction
            avg_score = sum(scores) / len(scores) if scores else 0
            prediction = "POSITIVE" if avg_score >= 0 else "NEGATIVE"
            
            # Build context for each word (use surrounding words)
            word_to_pos = {}
            for i, word in enumerate(words):
                if len(word) <= 2:
                    continue
                
                # Get context window around the word
                start = max(0, i - context_window)
                end = min(len(words), i + context_window + 1)
                context = " ".join(words[start:end])
                
                doc = self.nlp(context)
                # Find the token that matches our word
                for token in doc:
                    if token.text.lower() == word.lower():
                        word_to_pos[word] = token.pos_
                        break
            
            word_scores = list(zip(words, scores))
            
            # Filter for content words
            content_pos = pos
            
            if prediction == "POSITIVE":
                relevant_words = [(w, s) for w, s in word_scores 
                                if s > 0 
                                and word_to_pos.get(w) in content_pos
                                and len(w) > 2
                                and w not in string.punctuation]
            else:
                relevant_words = [(w, s) for w, s in word_scores 
                                if s < 0 
                                and word_to_pos.get(w) in content_pos
                                and len(w) > 2
                                and w not in string.punctuation]
            
            # Sort by absolute score and get top N
            relevant_words.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top words (already reconstructed)
            top_words = []
            seen_words = set()
            
            for word, _ in relevant_words[:top_n * 2]:  # Get more candidates
                if len(top_words) >= top_n:
                    break
                
                # Add if not already seen
                if word.lower() not in seen_words:
                    top_words.append(word)
                    seen_words.add(word.lower())
            
            if top_words:
                words_str = ", ".join(top_words)
                result = f"The model predicted {prediction} because of following words: {words_str}"
            else:
                result = f"The model predicted {prediction} (no significant words found)"
            
            results.append(result)
        
        return results
    
class ExplanationProcessor:
    """
    Processes SHAP/LIME explanations from pkl files and converts them to JSON format
    """
    
    def __init__(self, formatter: ExplanationFormatter):
        self.formatter = formatter
    
    def load_explanation_from_pkl(self, pkl_path: str):
        """Load a single explanation from a pkl file"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def process_single_explanation(self, explanation, explanation_type: str, threshold: float) -> Dict[str, str]:
        """
        Process a single explanation and return all formatted versions
        
        Returns:
            dict with keys: 'text_scores', 'text_labels', 'structured_text_scores', 
                           'structured_text_labels', 'top_words_scores', 'top_words_labels',
                           'natural_words', 'part_of_speech'
        """
        if not isinstance(explanation, list):
            explanation = [explanation]
        # Load explanation into formatter
        self.formatter.load_explanations(explanation, explanation_type)
        
        # Generate all formats
        formatted = {
            'text_scores': self.formatter.extract_as_text(brackets='score', threshold=threshold)[0],
            'text_labels': self.formatter.extract_as_text(brackets='label', threshold=threshold)[0],

            'structured_text_scores': self.formatter.extract_as_structured_text(brackets='score', threshold=threshold)[0],
            'structured_text_labels': self.formatter.extract_as_structured_text(brackets='label', threshold=threshold)[0],

            'top_words_scores': self.formatter.extract_top_words(brackets='score', top_n=20)[0],
            'top_words_labels': self.formatter.extract_top_words(brackets='label', top_n=20)[0],
            
            'natural_words': self.formatter.extract_as_natural_explanation(top_n=5)[0],
            'part_of_speech': self.formatter.extract_pos(top_n=5)[0]
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
                lime_pkl_path = lime_dir / f"lime_explanation_{sample_idx}.pkl"
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
        # Use the same prediction function as SHAP
        logits = predict_fast(documents=list_of_texts, model=model, tokenizer=tokenizer)
        
        # Convert logits to probabilities using softmax
        import torch
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        
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
