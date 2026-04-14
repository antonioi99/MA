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
import hashlib

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
    Unified formatter for both SHAP and LIME explanations.
    Provides consistent formatting methods for both explanation types.
    All methods select the top N words by absolute score rather than
    using a score threshold.
    """

    def __init__(self):
        self.explanation_type = None
        self.processed_data = []
        self.attention_predictions = None
        try:
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()
        except:
            self.nlp = None
            print("Warning: spaCy model not loaded. Install with: python -m spacy download en_core_web_sm")

    def _extract_lime_data(self, lime_explanations: List) -> List[Tuple[List[str], List[float]]]:
        """Extract words and scores from LIME explanations."""
        extracted_data = []

        for explanation in lime_explanations:
            labels = explanation.available_labels()
            exp_list = explanation.as_list(label=labels[0])
            lime_scores = dict(exp_list)

            raw_text = explanation.domain_mapper.indexed_string.raw_string()
            word_list = raw_text.split()
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

        if len(token_shap_values.shape) > 1:
            token_shap_values = token_shap_values[:, 1]

        words = []
        scores = []
        current_word = ""
        current_shaps = []

        for i, (token, shap_val) in enumerate(zip(tokens, token_shap_values)):
            if token in ['<s>', '</s>', '<pad>', '<unk>', '<|begin_of_text|>', '<|end_of_text|>', '']:
                continue

            starts_new_word = token.startswith(' ')

            if i == 1 and not starts_new_word:
                starts_new_word = True

            if starts_new_word and current_word:
                avg_shap = np.mean(current_shaps) if current_shaps else 0.0
                words.append(current_word)
                scores.append(avg_shap)
                current_word = ""
                current_shaps = []

            clean_token = token.lstrip(' ')
            current_word += clean_token
            current_shaps.append(shap_val)

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

    def _extract_attention_data(self, attention_explanations: List) -> List[Tuple[List[str], List[float]]]:
        """
        Extract words and scores from Attention explanations.
        Uses the mean attention scores (recommended aggregation method).
        """
        extracted_data = []
        self.attention_predictions = []

        for explanation in attention_explanations:
            words = [word_info['word'] for word_info in explanation['words']]
            scores = [word_info['attention_mean'] for word_info in explanation['words']]
            self.attention_predictions.append(explanation['prediction'])
            extracted_data.append((words, scores))

        return extracted_data

    def load_explanations(self, explanations, explanation_type: str):
        """Load explanations and prepare them for formatting."""
        self.explanation_type = explanation_type.lower()
        self.attention_predictions = None

        if self.explanation_type == 'shap':
            self.processed_data = self._extract_shap_data(explanations)
        elif self.explanation_type == 'lime':
            self.processed_data = self._extract_lime_data(explanations)
        elif self.explanation_type == 'attention':
            self.processed_data = self._extract_attention_data(explanations)
        else:
            raise ValueError("explanation_type must be 'shap', 'lime', or 'attention'")

    def _get_prediction(self, idx: int, scores: List[float]) -> str:
        """
        Helper method to get the prediction label for a given index.
        For attention: uses the actual prediction stored in the explanation.
        For SHAP/LIME: infers from the average score direction.
        """
        if self.explanation_type == 'attention' and self.attention_predictions is not None:
            return "POSITIVE" if self.attention_predictions[idx] == 1 else "NEGATIVE"
        else:
            avg_score = sum(scores) / len(scores) if scores else 0
            return "POSITIVE" if avg_score >= 0 else "NEGATIVE"

    def _get_top_n_mask(self, scores: List[float], top_n: int) -> List[bool]:
        """
        Return a boolean mask that is True for the top_n words by absolute score.
        Words outside the top_n are considered 'neutral'.
        """
        if top_n >= len(scores):
            return [True] * len(scores)
        indexed = sorted(enumerate(scores), key=lambda x: abs(x[1]), reverse=True)
        top_indices = {idx for idx, _ in indexed[:top_n]}
        return [i in top_indices for i in range(len(scores))]

    def extract_as_text(self, brackets, top_n: int = 20) -> List[str]:
        """
        Format explanations as annotated inline text.

        The top_n words by absolute score are annotated in brackets;
        all remaining words are left as plain text.

        For attention: top_n highest-scoring words are marked [HIGH] or [score].
        For SHAP/LIME: top_n most influential words are marked [POSITIVE/NEGATIVE]
                       or [+score/-score].
        """
        results = []

        for idx, (words, scores) in enumerate(self.processed_data):

            prediction = self._get_prediction(idx, scores)
            top_mask = self._get_top_n_mask(scores, top_n)

            result_parts = []

            for word, score, is_top in zip(words, scores, top_mask):
                if self.explanation_type == 'attention':
                    if is_top:
                        if brackets == 'label':
                            result_parts.append(f"{word}[HIGH]")
                        elif brackets == 'score':
                            result_parts.append(f"{word}[{score:.3f}]")
                    else:
                        result_parts.append(word)
                else:
                    if not is_top:
                        result_parts.append(word)
                    else:
                        if brackets == 'label':
                            label = "POSITIVE" if score >= 0 else "NEGATIVE"
                            result_parts.append(f"{word}[{label}]")
                        elif brackets == 'score':
                            sign = "+" if score >= 0 else ""
                            result_parts.append(f"{word}[{sign}{score:.3f}]")

            if self.explanation_type == 'attention':
                if brackets == 'score':
                    explanation = (
                        f"The model predicted {prediction}. Bracketed scores show how much attention "
                        f"the model paid to each word (0.0 = ignored, 1.0 = highest attention). "
                        f"Only the top {top_n} words by attention are annotated.\n\n"
                    )
                elif brackets == 'label':
                    explanation = (
                        f"The model predicted {prediction}. Words marked [HIGH] are the top {top_n} "
                        f"words by attention that the model focused on when making its prediction.\n\n"
                    )
            else:
                if brackets == 'score':
                    explanation = (
                        f"The model predicted {prediction}. Bracketed scores indicate each word's "
                        f"contribution to the overall sentiment (negative scores pull toward negative "
                        f"sentiment, positive scores pull toward positive sentiment). "
                        f"Only the top {top_n} most influential words are annotated.\n\n"
                    )
                elif brackets == 'label':
                    explanation = (
                        f"The model predicted {prediction}. Bracketed labels indicate each word's "
                        f"contribution to the overall sentiment (NEGATIVE pulls toward negative "
                        f"sentiment, POSITIVE pulls toward positive sentiment). "
                        f"Only the top {top_n} most influential words are annotated.\n\n"
                    )

            results.append(explanation + " ".join(result_parts))

        return results

    def extract_as_structured_text(self, brackets, top_n: int = 20) -> List[str]:
        """
        Format explanations as structured text with grouped word lists.

        The top_n words by absolute score are placed in the POSITIVE/NEGATIVE
        (or HIGH ATTENTION) group; all remaining words are placed in NEUTRAL.

        For attention: top_n words → HIGH ATTENTION, rest → (unlisted neutral).
        For SHAP/LIME: top_n words → POSITIVE or NEGATIVE by sign, rest → NEUTRAL.
        """
        results = []

        for idx, (words, scores) in enumerate(self.processed_data):

            prediction = self._get_prediction(idx, scores)
            top_mask = self._get_top_n_mask(scores, top_n)

            if self.explanation_type == 'attention':
                high_attention = []

                for word, score, is_top in zip(words, scores, top_mask):
                    if is_top:
                        if brackets == 'score':
                            high_attention.append(f"{word}[{score:.3f}]")
                        elif brackets == 'label':
                            high_attention.append(word)

                result_parts = []
                if brackets == 'score':
                    explanation = (
                        f"The model predicted {prediction}. The top {top_n} words by attention "
                        f"are shown (scores: 0.0 = ignored, 1.0 = highest attention).\n\n"
                    )
                elif brackets == 'label':
                    explanation = (
                        f"The model predicted {prediction}. The top {top_n} words by attention "
                        f"that the model focused on when making its prediction.\n\n"
                    )

                result_parts.append(explanation)
                if high_attention:
                    result_parts.append(f"HIGH ATTENTION: {' '.join(high_attention)}")
                else:
                    result_parts.append("HIGH ATTENTION: (none selected)")

            else:
                positive_words = []
                negative_words = []
                neutral_words = []

                for word, score, is_top in zip(words, scores, top_mask):
                    if not is_top:
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
                    explanation = (
                        f"The model predicted {prediction}. The top {top_n} words by absolute "
                        f"score are grouped by sentiment; all remaining words are NEUTRAL. "
                        f"Scores indicate contribution strength.\n\n"
                    )
                elif brackets == 'label':
                    explanation = (
                        f"The model predicted {prediction}. The top {top_n} words by absolute "
                        f"score are grouped by their sentiment contribution; all remaining words "
                        f"are NEUTRAL.\n\n"
                    )

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
        Extract the top_n most influential/attended words as a ranked list.
        (Unchanged logic — already top-N based.)
        """
        results = []

        for idx, (words, scores) in enumerate(self.processed_data):

            prediction = self._get_prediction(idx, scores)

            word_scores = list(zip(words, scores))
            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            top_words = word_scores[:top_n]
            result_parts = []

            if self.explanation_type == 'attention':
                if brackets == 'score':
                    explanation = (
                        f"The model predicted {prediction}. These are the most attended words "
                        f"(scores: 0.0 = ignored, 1.0 = highest attention):\n\n"
                    )
                    for word, score in top_words:
                        result_parts.append(f"{word} [{score:.3f}]\n")
                elif brackets == 'label':
                    explanation = (
                        f"The model predicted {prediction}. These words received the highest attention:\n\n"
                    )
                    for word, score in top_words:
                        result_parts.append(f"{word} [HIGH]\n")
            else:
                if brackets == 'score':
                    explanation = (
                        f"The model predicted {prediction}. These are the most influential words "
                        f"for this prediction. Scores show contribution (negative = toward negative, "
                        f"positive = toward positive):\n\n"
                    )
                    for word, score in top_words:
                        result_parts.append(f"{word} [{score:+.3f}]\n")
                elif brackets == 'label':
                    explanation = (
                        f"The model predicted {prediction}. These are the most influential words "
                        f"for this prediction:\n\n"
                    )
                    for word, score in top_words:
                        sentiment = "POSITIVE" if score > 0 else "NEGATIVE"
                        result_parts.append(f"{word} [{sentiment}]\n")

            results.append(explanation + ''.join(result_parts))

        return results

    def extract_as_natural_explanation(self, top_n: int = 5) -> List[str]:
        """
        Format explanations as natural language sentences.
        (Unchanged logic — already top-N based.)
        """
        import string

        conjunctions = {
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
            'although', 'because', 'since', 'unless', 'while',
            'if', 'then', 'than', 'that', 'though', 'whether'
        }

        results = []

        for idx, (words, scores) in enumerate(self.processed_data):

            prediction = self._get_prediction(idx, scores)

            word_scores = list(zip(words, scores))

            if self.explanation_type == 'attention':
                relevant_words = [(w, s) for w, s in word_scores]
            else:
                if prediction == "POSITIVE":
                    relevant_words = [(w, s) for w, s in word_scores if s > 0]
                else:
                    relevant_words = [(w, s) for w, s in word_scores if s < 0]

            filtered_words = [
                (word, score) for word, score in relevant_words
                if word not in string.punctuation
                and word.lower() not in conjunctions
                and len(word) > 2
            ]

            filtered_words.sort(key=lambda x: abs(x[1]), reverse=True)

            top_words = []
            seen_words = set()

            for word, _ in filtered_words[:top_n * 2]:
                if len(top_words) >= top_n:
                    break
                if word.lower() not in seen_words:
                    top_words.append(word)
                    seen_words.add(word.lower())

            if top_words:
                words_str = ", ".join(top_words)
                if self.explanation_type == 'attention':
                    result = f"The model predicted {prediction} while paying most attention to the following words: {words_str}"
                else:
                    result = f"The model predicted {prediction} because of the following words: {words_str}"
            else:
                result = f"The model predicted {prediction}"

            results.append(result)

        return results

    def extract_pos(self, top_n: int = 5, context_window: int = 2,
                    pos={"ADJ", "VERB", "NOUN", "ADV"}) -> List[str]:
        """
        Extract adjectives, verbs, and nouns with contextual POS tagging.
        (Unchanged logic — already top-N based.)
        """
        if self.nlp is None:
            raise RuntimeError("spaCy model not loaded")

        import string

        results = []

        for idx, (words, scores) in enumerate(self.processed_data):

            prediction = self._get_prediction(idx, scores)

            word_to_pos = {}
            for i, word in enumerate(words):
                if len(word) <= 2:
                    continue
                start = max(0, i - context_window)
                end = min(len(words), i + context_window + 1)
                context = " ".join(words[start:end])
                doc = self.nlp(context)
                for token in doc:
                    if token.text.lower() == word.lower():
                        word_to_pos[word] = token.pos_
                        break

            word_scores = list(zip(words, scores))
            content_pos = pos

            if self.explanation_type == 'attention':
                relevant_words = [(w, s) for w, s in word_scores
                                  if word_to_pos.get(w) in content_pos
                                  and len(w) > 2
                                  and w not in string.punctuation]
            else:
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

            relevant_words.sort(key=lambda x: abs(x[1]), reverse=True)

            top_words = []
            seen_words = set()

            for word, _ in relevant_words[:top_n * 2]:
                if len(top_words) >= top_n:
                    break
                if word.lower() not in seen_words:
                    top_words.append(word)
                    seen_words.add(word.lower())

            if top_words:
                words_str = ", ".join(top_words)
                if self.explanation_type == 'attention':
                    result = f"The model predicted {prediction} while paying most attention to the following words: {words_str}"
                else:
                    result = f"The model predicted {prediction} because of the following words: {words_str}"
            else:
                result = f"The model predicted {prediction} (no significant words found)"

            results.append(result)

        return results


class ExplanationProcessor:
    """
    Processes SHAP/LIME explanations from pkl files and converts them to JSON format.
    """

    def __init__(self, formatter: ExplanationFormatter):
        self.formatter = formatter

    def load_explanation_from_pkl(self, pkl_path: str):
        """Load a single explanation from a pkl file."""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def process_single_explanation(self, explanation, explanation_type: str, top_n: int = 20) -> Dict[str, str]:
        """
        Process a single explanation and return all formatted versions.

        Args:
            explanation:      Raw explanation object (SHAP / LIME / attention).
            explanation_type: One of 'shap', 'lime', 'attention'.
            top_n:            Number of top words (by absolute score) to annotate.
                              Defaults to 20.

        Returns:
            dict with keys: 'text_scores', 'text_labels', 'structured_text_scores',
                            'structured_text_labels', 'top_words_scores', 'top_words_labels',
                            'natural_words', 'part_of_speech'
        """
        if not isinstance(explanation, list):
            explanation = [explanation]

        self.formatter.load_explanations(explanation, explanation_type)

        formatted = {
            'text_scores':              self.formatter.extract_as_text(brackets='score', top_n=top_n)[0],
            'text_labels':              self.formatter.extract_as_text(brackets='label', top_n=top_n)[0],
            'structured_text_scores':   self.formatter.extract_as_structured_text(brackets='score', top_n=top_n)[0],
            'structured_text_labels':   self.formatter.extract_as_structured_text(brackets='label', top_n=top_n)[0],
            'top_words_scores':         self.formatter.extract_top_words(brackets='score', top_n=top_n)[0],
            'top_words_labels':         self.formatter.extract_top_words(brackets='label', top_n=top_n)[0],
            'natural_words':            self.formatter.extract_as_natural_explanation(top_n=5)[0],
            'part_of_speech':           self.formatter.extract_pos(top_n=5)[0],
        }

        return formatted

    def process_explanations_from_files(
        self,
        shap_pkl_dir: str,
        shap_random_pkl_dir: str,
        lime_pkl_dir: str,
        attention_pkl_dir: str,
        samples: List[str],
        labels: List[int],
        output_json: str,
        top_n_shap: int = 20,
        top_n_random: int = 20,
        top_n_lime: int = 20,
        top_n_attention: int = 20,
    ):
        output_path = Path(output_json)
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            tqdm.write(f"Resuming from existing file: {len(data_dict)} samples already processed.")
        else:
            data_dict = {}

        shap_dir = Path(shap_pkl_dir)
        shap_random_dir = Path(shap_random_pkl_dir)
        lime_dir = Path(lime_pkl_dir) if lime_pkl_dir else None
        attention_dir = Path(attention_pkl_dir) if attention_pkl_dir else None

        for idx, sample in enumerate(tqdm(samples, desc="Processing explanations", unit="sample")):

            doc_hash = hashlib.md5(sample.encode("utf-8")).hexdigest()

            if doc_hash in data_dict:
                continue

            shap_pkl_path = shap_dir / f"shap_{doc_hash}.pkl"
            if not shap_pkl_path.exists():
                tqdm.write(f"Warning: SHAP file not found for doc_hash {doc_hash}")
                continue
            shap_explanation = self.load_explanation_from_pkl(shap_pkl_path)
            shap_formatted = self.process_single_explanation(shap_explanation, 'shap', top_n=top_n_shap)

            shap_random_pkl_path = shap_random_dir / f"shap_random_{doc_hash}.pkl"
            if not shap_random_pkl_path.exists():
                tqdm.write(f"Warning: SHAP random file not found for doc_hash {doc_hash}")
                shap_random_formatted = {}
            else:
                shap_random_explanation = self.load_explanation_from_pkl(shap_random_pkl_path)
                shap_random_formatted = self.process_single_explanation(shap_random_explanation, 'shap', top_n=top_n_random)

            lime_formatted = {}
            if lime_dir:
                lime_pkl_path = lime_dir / f"lime_{doc_hash}.pkl"
                if lime_pkl_path.exists():
                    lime_explanation = self.load_explanation_from_pkl(lime_pkl_path)
                    lime_formatted = self.process_single_explanation(lime_explanation, 'lime', top_n=top_n_lime)
                else:
                    tqdm.write(f"Warning: LIME file not found for doc_hash {doc_hash}")

            attention_formatted = {}
            if attention_dir:
                attention_pkl_path = attention_dir / f"attention_{doc_hash}.pkl"
                if attention_pkl_path.exists():
                    attention_explanation = self.load_explanation_from_pkl(attention_pkl_path)
                    attention_formatted = self.process_single_explanation(attention_explanation, 'attention', top_n=top_n_attention)
                else:
                    tqdm.write(f"Warning: Attention file not found for doc_hash {doc_hash}")

            data_dict[doc_hash] = {
                "sample": str(sample),
                "label": int(labels[idx]),
                "shap": shap_formatted,
                "shap_random": shap_random_formatted,
                "lime": lime_formatted,
                "attention": attention_formatted,
            }

            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4, ensure_ascii=False)

        tqdm.write(f"Saved {len(data_dict)} samples to {output_json}")


def merge_json_files_from_folder(folder_path: Union[str, Path]) -> Dict[str, dict]:
    """Merge all JSON files in a folder into one dictionary."""
    folder = Path(folder_path)
    merged_data = {}

    json_files = sorted(folder.glob("explanations*"))

    if not json_files:
        print(f"Warning: No JSON files found in {folder_path}")
        return merged_data

    print(f"Found {len(json_files)} JSON files")

    for file_path in json_files:
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.update(data)

    return merged_data


def lime_explainer(model, tokenizer):
    """Returns the predict_probs function for LIME."""
    from lime import lime_text

    def predict_probs(list_of_texts):
        logits = predict_fast(documents=list_of_texts, model=model, tokenizer=tokenizer)
        import torch
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        return probabilities

    return predict_probs


def create_similarity_groups_from_data(json_file_path, predictions_json, output_json_path, test_texts, test_ids, max_features=5000):

    print(f"Test set size: {len(test_texts)}")

    print(f"Loading dev set from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    with open(predictions_json, 'r', encoding='utf-8') as f:
        data_predictions = json.load(f)

    dev_ids = []
    dev_texts = []
    dev_predictions = []

    for sample_idx, data in dev_data.items():
        dev_ids.append(sample_idx)
        dev_texts.append(data['sample'])
    for sample_idx, data in data_predictions.items():
        dev_predictions.append(data['prediction'])

    print(f"Dev set size: {len(dev_texts)}")

    dev_positive_indices = [i for i, pred in enumerate(dev_predictions) if pred == 1]
    dev_negative_indices = [i for i, pred in enumerate(dev_predictions) if pred == 0]

    print(f"Dev positive predictions: {len(dev_positive_indices)}, Dev negative predictions: {len(dev_negative_indices)}")

    print("\nComputing TF-IDF vectors...")
    all_texts = test_texts + dev_texts
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    all_tfidf = vectorizer.fit_transform(all_texts)

    test_tfidf = all_tfidf[:len(test_texts)]
    dev_tfidf = all_tfidf[len(test_texts):]

    print("Computing cosine similarity between test and dev sets...")
    test_dev_similarity = cosine_similarity(test_tfidf, dev_tfidf)

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
        print(f"Loaded existing results from {output_json_path} with {len(result_dict)} instances")
    else:
        result_dict = {}

    print("Creating groups for each test instance...")
    for test_idx in tqdm(range(len(test_texts)), desc="Processing test instances"):

        if test_ids[test_idx] in result_dict:
            continue

        similarities = test_dev_similarity[test_idx]

        pos_similarities = [(dev_positive_indices[i], similarities[dev_positive_indices[i]])
                            for i in range(len(dev_positive_indices))]
        pos_similarities.sort(key=lambda x: x[1], reverse=True)
        top_2_positive = [idx for idx, _ in pos_similarities[:2]]

        neg_similarities = [(dev_negative_indices[i], similarities[dev_negative_indices[i]])
                            for i in range(len(dev_negative_indices))]
        neg_similarities.sort(key=lambda x: x[1], reverse=True)
        top_2_negative = [idx for idx, _ in neg_similarities[:2]]

        dev_group_indices = top_2_positive + top_2_negative
        random.shuffle(dev_group_indices)

        dev_group_ids = [dev_ids[i] for i in dev_group_indices]
        dev_group_predictions = [dev_predictions[i] for i in dev_group_indices]

        result_dict[test_ids[test_idx]] = {
            "test_instance": test_texts[test_idx],
            "dev_group": dev_group_ids,
            "dev_predictions": dev_group_predictions
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"\nCompleted! Created groups for {len(result_dict)} test instances")
    print(f"Results saved to {output_json_path}")
    return result_dict


############################
# ATTENTION HELPER FUNCTIONS
############################

def group_tokens_into_words(tokens, offsets):
    """Group subword tokens into complete words."""
    words = []
    word_token_groups = []
    current_word = []
    current_token_indices = []

    for idx, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        if token in ['<|begin_of_text|>', '<|end_of_text|>', '<s>', '</s>', '<pad>']:
            if current_word:
                words.append(''.join(current_word).strip('Ġ'))
                word_token_groups.append(current_token_indices)
                current_word = []
                current_token_indices = []
            continue

        if token.startswith('Ġ') and current_word:
            words.append(''.join(current_word).strip('Ġ'))
            word_token_groups.append(current_token_indices)
            current_word = [token]
            current_token_indices = [idx]
        else:
            current_word.append(token)
            current_token_indices.append(idx)

    if current_word:
        words.append(''.join(current_word).strip('Ġ'))
        word_token_groups.append(current_token_indices)

    return words, word_token_groups


def aggregate_attention_to_words(attention_tensor, word_token_groups,
                                 non_pad_indices, method='mean'):
    """Aggregate subword attention scores to word level."""
    actual_last_pos = non_pad_indices[-1].item()

    selected_layers = attention_tensor[-3:]
    avg_layers = selected_layers.mean(dim=0)[0]  # (heads, seq, seq)
    avg_heads = avg_layers.mean(dim=0)            # (seq, seq)

    last_token_attn = avg_heads[actual_last_pos].cpu().numpy()

    word_scores = []
    for token_indices in word_token_groups:
        actual_positions = [non_pad_indices[i].item() for i in token_indices]
        token_scores = [last_token_attn[pos] for pos in actual_positions]

        if method == 'max':
            word_score = max(token_scores)
        elif method == 'mean':
            word_score = np.mean(token_scores)
        elif method == 'sum':
            word_score = np.sum(token_scores)

        word_scores.append(word_score)

    return np.array(word_scores)


def extract_attention_explanation(text, model, tokenizer):
    """Extract word-level attention scores for a single text."""
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    offsets = encoding['offset_mapping'][0]

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = output.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(logits).item()
        confidence = probs[0, pred_class].item()

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    non_pad_mask = attention_mask[0].cpu().bool()
    non_pad_indices = non_pad_mask.nonzero(as_tuple=True)[0]
    tokens_filtered = [all_tokens[i] for i in non_pad_indices]
    offsets_filtered = [offsets[i] for i in non_pad_indices]

    attentions = output.attentions
    attention_all_layers = torch.stack(attentions)

    words, word_token_groups = group_tokens_into_words(tokens_filtered, offsets_filtered)

    word_scores_mean = aggregate_attention_to_words(
        attention_all_layers, word_token_groups, non_pad_indices, method='mean'
    )
    word_scores_max = aggregate_attention_to_words(
        attention_all_layers, word_token_groups, non_pad_indices, method='max'
    )
    word_scores_sum = aggregate_attention_to_words(
        attention_all_layers, word_token_groups, non_pad_indices, method='sum'
    )

    word_data = []
    for i, word in enumerate(words):
        word_info = {
            'word': word,
            'attention_mean': float(word_scores_mean[i]),
            'attention_max': float(word_scores_max[i]),
            'attention_sum': float(word_scores_sum[i])
        }
        word_data.append(word_info)

    attention_explanation = {
        'text': text,
        'prediction': int(pred_class),
        'confidence': float(confidence),
        'words': word_data
    }

    return attention_explanation