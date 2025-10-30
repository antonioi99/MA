from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
import random
import json
import random
import torch
import re
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from datasets import load_dataset

# Main grouping function
def create_similarity_groups(texts, labels, group_size=4, max_features=5000):
    """
    Group texts by TF-IDF + cosine similarity with balanced labels.

    Args:
        texts: List of text strings
        labels: List of labels (0 for negative, 1 for positive)
        group_size: Size of each group (default 4: 2 positive + 2 negative)
        max_features: Maximum features for TF-IDF vectorizer

    Returns:
        List of groups, each containing indices, labels, texts, and similarity score
    """
    # Separate by labels
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]

    print(f"Total texts: {len(texts)}")
    print(f"Positive reviews: {len(positive_indices)}")
    print(f"Negative reviews: {len(negative_indices)}")

    # Compute TF-IDF
    print("\nComputing TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity
    print("Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Create groups
    print("Creating similarity groups...")
    groups = []
    used_positive = set()
    used_negative = set()

    for pos_idx in positive_indices:
        if pos_idx in used_positive:
            continue

        # Find most similar positive review
        pos_similarities = [(i, cosine_sim[pos_idx][i])
                           for i in positive_indices
                           if i != pos_idx and i not in used_positive]

        if not pos_similarities:
            continue

        pos_similarities.sort(key=lambda x: x[1], reverse=True)
        second_pos_idx = pos_similarities[0][0]

        # Find two most similar negative reviews
        avg_pos_vector = (tfidf_matrix[pos_idx] + tfidf_matrix[second_pos_idx]) / 2

        neg_similarities = []
        for neg_idx in negative_indices:
            if neg_idx not in used_negative:
                sim = cosine_similarity(avg_pos_vector, tfidf_matrix[neg_idx])[0][0]
                neg_similarities.append((neg_idx, sim))

        if len(neg_similarities) < 2:
            continue

        neg_similarities.sort(key=lambda x: x[1], reverse=True)
        first_neg_idx = neg_similarities[0][0]
        second_neg_idx = neg_similarities[1][0]

        # Create group
        group = {
            'indices': [pos_idx, second_pos_idx, first_neg_idx, second_neg_idx],
            'labels': [labels[pos_idx], labels[second_pos_idx],
                      labels[first_neg_idx], labels[second_neg_idx]],
            'texts': [texts[pos_idx], texts[second_pos_idx],
                     texts[first_neg_idx], texts[second_neg_idx]],
            'avg_similarity': np.mean([
                cosine_sim[pos_idx][second_pos_idx],
                cosine_sim[first_neg_idx][second_neg_idx],
                cosine_sim[pos_idx][first_neg_idx],
                cosine_sim[pos_idx][second_neg_idx],
                cosine_sim[second_pos_idx][first_neg_idx],
                cosine_sim[second_pos_idx][second_neg_idx]
            ])
        }

        groups.append(group)
        used_positive.add(pos_idx)
        used_positive.add(second_pos_idx)
        used_negative.add(first_neg_idx)
        used_negative.add(second_neg_idx)

    return groups


# Token filtering function (not used by default)
def filter_by_token_count(texts, labels, tokenizer, max_tokens):
    """
    Filter texts by token count.

    Args:
        texts: List of text strings
        labels: List of labels
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum token count threshold

    Returns:
        filtered_texts, filtered_labels, filtered_indices
    """
    filtered_indices = []
    filtered_texts = []
    filtered_labels = []

    for idx, (text, label) in enumerate(zip(texts, labels)):
        token_count = len(tokenizer.encode(text))
        if token_count <= max_tokens:
            filtered_indices.append(idx)
            filtered_texts.append(text)
            filtered_labels.append(label)

    print(f"Original size: {len(texts)}")
    print(f"Filtered size: {len(filtered_texts)}")
    print(f"Removed: {len(texts) - len(filtered_texts)} reviews")

    return filtered_texts, filtered_labels, filtered_indices


@dataclass
class SampleData:
    """Structure to hold individual sample data"""
    sample_id: str
    review_text: str
    true_label: int
    model_prediction: int
    shap_formatted: str
    lime_formatted: str

class DataLoader:
    """Handles loading and splitting the dataset"""

    def __init__(self, json_file_path: str, similarity_groups: list = None):
        self.json_file_path = json_file_path
        self.similarity_groups = similarity_groups
        self.all_samples = []
        self.index_to_sample = {}  # Map original indices to samples
        self.learn_group = []
        self.test_group = []
        self.learn_group_batches = []
        self.test_group_batches = []

    def load_data(self) -> None:
        """Load data from JSON file and parse into SampleData objects"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sample_id, sample_info in data.items():
            sample_data = SampleData(
                sample_id=sample_id,
                review_text=sample_info['sample'],
                true_label=sample_info['label'],
                model_prediction=sample_info['prediction'],
                shap_formatted=sample_info['shap']['formatted_text'],
                lime_formatted=sample_info['lime']['formatted_text']
            )
            self.all_samples.append(sample_data)

    def load_data_with_groups(self, texts: List[str], labels: List[int]) -> None:
        """Load data directly from texts and labels (for use with similarity groups)"""
        for idx, (text, label) in enumerate(zip(texts, labels)):
            sample_data = SampleData(
                sample_id=f"sample_{idx}",
                review_text=text,
                true_label=label,
                model_prediction=label,  # Assuming prediction matches label for now
                shap_formatted="",  # Will be empty for now
                lime_formatted=""   # Will be empty for now
            )
            self.all_samples.append(sample_data)
            self.index_to_sample[idx] = sample_data

    def split_data_with_groups(self, num_train_groups: int = 16, num_test_groups: int = 16, seed: int = 42) -> None:
        """Split data into Group A (training) and Group B (testing) using similarity groups"""
        if self.similarity_groups is None:
            raise ValueError("No similarity groups provided.")

        total_groups_needed = num_train_groups + num_test_groups
        if len(self.similarity_groups) < total_groups_needed:
            raise ValueError(f"Not enough groups. Need {total_groups_needed}, have {len(self.similarity_groups)}")

        # Set seed for reproducible splits
        random.seed(seed)

        # Shuffle groups
        shuffled_groups = random.sample(self.similarity_groups, len(self.similarity_groups))

        # Split groups into train and test
        train_groups = shuffled_groups[:num_train_groups]
        test_groups = shuffled_groups[num_train_groups:num_train_groups + num_test_groups]

        # Convert group indices to actual samples
        self.learn_group_batches = []
        for group in train_groups:
            group_indices = group['indices']
            group_indices = random.sample(group_indices, len(group_indices))  # Shuffle indices
            batch = [self.index_to_sample[idx] for idx in group_indices]
            self.learn_group_batches.append(batch)
            self.learn_group.extend(batch)

        self.test_group_batches = []
        for group in test_groups:
            group_indices = group['indices']
            group_indices = random.sample(group_indices, len(group_indices))  # Shuffle indices
            batch = [self.index_to_sample[idx] for idx in group_indices]
            self.test_group_batches.append(batch)
            self.test_group.extend(batch)

        print(f"Group A: {len(self.learn_group)} samples → {len(self.learn_group_batches)} batches of 4")
        print(f"Group B: {len(self.test_group)} samples → {len(self.test_group_batches)} batches of 4")
        print(f"Total samples used: {len(self.learn_group) + len(self.test_group)} out of {len(self.all_samples)}")

    def _create_batches(self, samples: List[SampleData], batch_size: int) -> List[List[SampleData]]:
        """Split samples into batches"""
        batches = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batches.append(batch)
        return batches

    def get_num_batches(self) -> int:
        """Get number of batches"""
        return len(self.learn_group_batches)
    
class PhaseManager:
    """Manages the four phases of the model simulation experiment with batching"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.num_batches = data_loader.get_num_batches()

        # Store predictions per batch
        self.phase2_predictions = {}
        self.phase4_predictions = {}

    def get_phase1_intro(self) -> str:
        """Phase 1: Introduction"""
        return (
            "Your task: Learn how a sentiment analysis model works by studying its predictions.\n\n"
            "The model predicts:\n"
            "0 = Negative sentiment\n"
            "1 = Positive sentiment\n\n"
            "I will show you 4 examples. Study them carefully."
        )

    def get_phase1_examples(self, batch_idx: int) -> str:
        """Phase 1: Get examples for a specific batch as single message"""
        batch = self.data_loader.learn_group_batches[batch_idx]
        examples = []

        for i, sample in enumerate(batch, 1):
            example = (
                f"Example {i}:\n"
                f"Review: {sample.review_text}\n"
                f"Model's Prediction: {sample.model_prediction}"
            )
            examples.append(example)

        # Return all examples as one message
        return "\n\n".join(examples)

    def get_phase2_intro(self) -> str:
        """Phase 2: Introduction for predictions"""
        return (
            "Now you will see 4 new reviews one at a time. "
            "For each review, predict what the model would output. "
            "Reply with only 0 or 1."
        )

    def get_phase2_reviews(self, batch_idx: int) -> List[str]:
        """Phase 2: Get reviews for prediction"""
        batch = self.data_loader.test_group_batches[batch_idx]
        return [sample.review_text for sample in batch]

    def get_phase3_intro(self) -> str:
        """Phase 3: Introduction"""
        return (
            "Now I'll show you 4 examples with explanations of how the model works.\n\n"
            "The explanation shows each word's influence:\n"
            "- Positive numbers (e.g., [+0.019]) push toward predicting 1 (positive)\n"
            "- Negative numbers (e.g., [-0.046]) push toward predicting 0 (negative)\n"
            "- Larger absolute values = stronger influence\n\n"
            "Study how the model uses different words."
        )

    def get_phase3_examples(self, batch_idx: int) -> str:
        """Phase 3: Get examples with explanations for a specific batch"""
        batch = self.data_loader.learn_group_batches[batch_idx]
        examples = []

        for i, sample in enumerate(batch, 1):
            example = (
                f"Example {i}:\n"
                f"Review: {sample.review_text}\n"
                f"Model's Prediction: {sample.model_prediction}\n"
                f"Explanation: {sample.shap_formatted}"
            )
            examples.append(example)

        # Return all examples as one message
        return "\n\n".join(examples)

    def get_phase4_intro(self) -> str:
        """Phase 4: Introduction"""
        return (
            "Now predict again for the same 4 reviews. "
            "Use what you learned from the explanations. "
            "Having seen the explanations, you have the possibility to change your predictions. "
            "Reply with only 0 (negative) or 1 (positive)."
        )

    def get_phase4_reviews(self, batch_idx: int) -> List[str]:
        """Phase 4: Get reviews (same as Phase 2)"""
        return self.get_phase2_reviews(batch_idx)

    def store_phase2_predictions(self, batch_idx: int, predictions: List[int]) -> None:
        """Store Phase 2 predictions for a batch"""
        expected = len(self.data_loader.test_group_batches[batch_idx])
        if len(predictions) != expected:
            raise ValueError(f"Batch {batch_idx}: Expected {expected} predictions, got {len(predictions)}")
        self.phase2_predictions[batch_idx] = predictions
        print(f"Stored {len(predictions)} Phase 2 predictions for batch {batch_idx}")

    def store_phase4_predictions(self, batch_idx: int, predictions: List[int]) -> None:
        """Store Phase 4 predictions for a batch"""
        expected = len(self.data_loader.test_group_batches[batch_idx])
        if len(predictions) != expected:
            raise ValueError(f"Batch {batch_idx}: Expected {expected} predictions, got {len(predictions)}")
        self.phase4_predictions[batch_idx] = predictions
        print(f"Stored {len(predictions)} Phase 4 predictions for batch {batch_idx}")

    def get_prediction_comparison(self) -> Dict[str, Any]:
        """Analyze results across all batches"""
        if not self.phase2_predictions or not self.phase4_predictions:
            return {"error": "Missing predictions from one or both phases"}

        # Aggregate all predictions
        all_phase2_preds = []
        all_phase4_preds = []
        all_true_labels = []

        for batch_idx in range(self.num_batches):
            batch = self.data_loader.test_group_batches[batch_idx]
            true_labels = [sample.model_prediction for sample in batch]

            all_phase2_preds.extend(self.phase2_predictions[batch_idx])
            all_phase4_preds.extend(self.phase4_predictions[batch_idx])
            all_true_labels.extend(true_labels)

        # Calculate overall accuracy
        phase2_correct = sum(p == t for p, t in zip(all_phase2_preds, all_true_labels))
        phase4_correct = sum(p == t for p, t in zip(all_phase4_preds, all_true_labels))
        total = len(all_true_labels)

        # Per-batch analysis
        batch_results = []
        for batch_idx in range(self.num_batches):
            batch = self.data_loader.test_group_batches[batch_idx]
            true_labels = [sample.model_prediction for sample in batch]
            p2_preds = self.phase2_predictions[batch_idx]
            p4_preds = self.phase4_predictions[batch_idx]

            batch_p2_correct = sum(p == t for p, t in zip(p2_preds, true_labels))
            batch_p4_correct = sum(p == t for p, t in zip(p4_preds, true_labels))

            batch_results.append({
                "batch_idx": batch_idx,
                "phase2_accuracy": batch_p2_correct / len(true_labels),
                "phase4_accuracy": batch_p4_correct / len(true_labels),
                "improvement": (batch_p4_correct - batch_p2_correct) / len(true_labels)
            })

        return {
            "overall": {
                "phase2_accuracy": phase2_correct / total,
                "phase4_accuracy": phase4_correct / total,
                "improvement": (phase4_correct - phase2_correct) / total,
                "phase2_correct": phase2_correct,
                "phase4_correct": phase4_correct,
                "total_samples": total,
                "predictions_changed": sum(1 for p2, p4 in zip(all_phase2_preds, all_phase4_preds) if p2 != p4)
            },
            "per_batch": batch_results
        }
    
class LLMClient:
    """Handles interaction with the Prometheus model"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Store conversation as list of messages
        self.conversation_messages = []
        self.system_message = None

        # Token IDs for "0" and "1" (set after loading tokenizer)
        self.token_0_id = None
        self.token_1_id = None

        self._load_model()
        self._setup_prediction_tokens()

    def _load_model(self):
        """Load tokenizer and model"""
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on device: {self.device}")
        print(f"Model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")

    def _setup_prediction_tokens(self):
        """Get token IDs for '0' and '1'"""
        # Try different encodings to find the single-token versions
        for test_str in ["0", " 0", "0 "]:
            tokens = self.tokenizer.encode(test_str, add_special_tokens=False)
            if len(tokens) == 1:
                self.token_0_id = tokens[0]
                break

        for test_str in ["1", " 1", "1 "]:
            tokens = self.tokenizer.encode(test_str, add_special_tokens=False)
            if len(tokens) == 1:
                self.token_1_id = tokens[0]
                break

        if self.token_0_id is None or self.token_1_id is None:
            raise ValueError("Could not find single-token IDs for '0' and '1'")

        print(f"Token IDs - 0: {self.token_0_id}, 1: {self.token_1_id}")

    def set_system_message(self, system_message: str):
        """Set the system message for the conversation"""
        self.system_message = system_message
        self.conversation_messages = []

    def _build_prompt(self, user_message: str) -> str:
        """Build prompt using Qwen format"""

        prompt = ""

        if self.system_message is not None:
          prompt += f"<|im_start|>system\n{self.system_message}<|im_end|>\n"

        # Add conversation history
        for msg in self.conversation_messages:
            if msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

        # Add current user message
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def generate_response(self, user_message: str,
                         max_tokens: int = 1500,
                         temperature: float = 0.2) -> str:
        """Generate a regular conversational response"""
        prompt = self._build_prompt(user_message)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            #max_length=4096  # Reduced from 8192
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]
        print(f"Input length: {input_length} tokens")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode only new tokens
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Add to conversation history
        self.conversation_messages.append({"role": "user", "content": user_message})
        self.conversation_messages.append({"role": "assistant", "content": response})

        return response

    def generate_constrained_prediction(self, user_message: str) -> int:
        """Generate a prediction constrained to only 0 or 1"""
        prompt = self._build_prompt(user_message)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            #max_length=4096
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]

        # Get logits for the next token
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            next_token_logits = outputs.logits[0, -1, :]

        # Only consider logits for "0" and "1" tokens
        logits_0 = next_token_logits[self.token_0_id].item()
        logits_1 = next_token_logits[self.token_1_id].item()

        # Choose the one with higher logit
        prediction = 0 if logits_0 > logits_1 else 1

        # Add to conversation history
        self.conversation_messages.append({"role": "user", "content": user_message})
        self.conversation_messages.append({"role": "assistant", "content": str(prediction)})

        return prediction

    def generate_predictions_for_reviews(self, reviews: List[str],
                                        phase_intro: str) -> List[int]:
        """Generate predictions for multiple reviews"""
        predictions = []

        # Send phase introduction
        print("Sending phase introduction...")
        self.generate_response(phase_intro, max_tokens=100)

        print(f"\nGenerating predictions for {len(reviews)} reviews...")

        for i, review in enumerate(reviews, 1):
            # Simple, direct prompt
            prompt = f"{review}\n\nSentiment (0 or 1):"

            prediction = self.generate_constrained_prediction(prompt)
            predictions.append(prediction)

            print(f"Review {i}/{len(reviews)}: {prediction}")

        return predictions

    def show_examples_sequentially(self, examples: List[str], intro: str):
        """Show learning examples one at a time"""
        # Send introduction
        print("Sending introduction...")
        self.generate_response(intro, max_tokens=100)

        print(f"\nShowing {len(examples)} examples...")

        for i, example in enumerate(examples, 1):
            # Just send the example, expect brief acknowledgment
            response = self.generate_response(example, max_tokens=50)
            print(f"Example {i}/{len(examples)} shown")
            # Optionally print first 50 chars of response for debugging
            # print(f"  Response: {response[:50]}...")

    def clear_conversation(self):
        """Clear conversation history but keep system message"""
        self.conversation_messages = []
        print("Conversation history cleared")

    def save_conversation_json(self, filepath: str = None) -> str:
        """Save conversation in JSON format"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversation_{timestamp}.json"

        conversation_data = {
            "metadata": {
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(self.conversation_messages)
            },
            "system_message": self.system_message,
            "messages": self.conversation_messages
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        print(f"Conversation saved to {filepath}")
        return filepath

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        total_chars = sum(len(m["content"]) for m in self.conversation_messages)
        if self.system_message:
            total_chars += len(self.system_message)

        return {
            "total_messages": len(self.conversation_messages),
            "user_messages": len([m for m in self.conversation_messages if m["role"] == "user"]),
            "assistant_messages": len([m for m in self.conversation_messages if m["role"] == "assistant"]),
            "has_system_message": self.system_message is not None,
            "total_characters": total_chars
        }
    
class GetPredictions:

    def __init__(self, json_file_path: str, system_template: str, model_name: str):
        self.json_file_path = json_file_path
        self.system_template = system_template
        self.model_name = model_name

        # Components
        self.data_loader = None
        self.phase_manager = None
        self.llm_client = None

        # Store conversation snapshots per batch
        self.batch_conversations = []

        # Experiment results storage
        self.experiment_results = {
            "setup": {},
            "batches": [],
            "analysis": {},
            "metadata": {
                "start_time": None,
                "end_time": None,
                "model_name": model_name
            }
        }

    def setup_experiment(self, texts: List[str] = None, labels: List[int] = None,
                        similarity_groups: list = None,
                        num_train_groups: int = 16, num_test_groups: int = 16,
                        seed: int = 42) -> None:
        """Initialize all components and prepare data"""
        print("Setting up experiment...")

        # Initialize data loader
        self.data_loader = DataLoader(self.json_file_path, similarity_groups=similarity_groups)

        # Load data based on whether we have texts/labels or JSON file
        if texts is not None and labels is not None:
            print(f"Loading data from provided texts and labels ({len(texts)} samples)")
            self.data_loader.load_data_with_groups(texts, labels)
        else:
            print("Loading data from JSON file")
            self.data_loader.load_data()

        # Use similarity-based grouping if groups provided

        print(f"Using {len(similarity_groups)} similarity groups")
        self.data_loader.split_data_with_groups(num_train_groups, num_test_groups, seed)

        # Initialize phase manager
        self.phase_manager = PhaseManager(self.data_loader)

        # Load LLM
        print("\nLoading LLM model...")
        self.llm_client = LLMClient(model_name=self.model_name)
        self.llm_client.set_system_message(self.system_template)

        # Store setup info
        self.experiment_results["setup"] = {
            "total_samples": len(self.data_loader.all_samples),
            "train_group_size": len(self.data_loader.learn_group),
            "test_group_size": len(self.data_loader.test_group),
            "batch_size": 4,
            "num_batches": self.data_loader.get_num_batches(),
            "seed": seed,
            "using_similarity_groups": similarity_groups is not None
        }

        print("\nSetup complete!")

    def run_batch(self, batch_idx: int) -> Dict[str, Any]:
        """Run all 4 phases for a single batch"""
        num_batches = self.data_loader.get_num_batches()

        print("\n" + "="*70)
        print(f"BATCH {batch_idx + 1}/{num_batches}")
        print("="*70)

        batch_results = {
            "batch_idx": batch_idx,
            "phases": {}
        }

        # Phase 1: Show examples
        print("\nPhase 1: Learning from 4 examples...")
        intro = self.phase_manager.get_phase1_intro()
        examples = self.phase_manager.get_phase1_examples(batch_idx)
        full_message = intro + "\n\n" + examples
        self.llm_client.generate_response(full_message, max_tokens=100)
        batch_results["phases"]["phase_1"] = {"completed": True}

        # Phase 2: Make predictions
        print("Phase 2: Making predictions...")
        intro = self.phase_manager.get_phase2_intro()
        reviews = self.phase_manager.get_phase2_reviews(batch_idx)
        predictions = self.llm_client.generate_predictions_for_reviews(reviews, intro)
        self.phase_manager.store_phase2_predictions(batch_idx, predictions)
        batch_results["phases"]["phase_2"] = {"predictions": predictions}

        # Store Phase 1-2 conversation snapshot before clearing
        phase12_conversation = {
            "batch_idx": batch_idx,
            "phases": "1-2",
            "messages": self.llm_client.conversation_messages.copy()
        }

        # Clear conversation history before Phase 3
        # print("\nClearing conversation history...")
        # self.llm_client.clear_conversation()

        # Phase 3: Show examples with explanations
        print("Phase 3: Learning with explanations...")
        intro = self.phase_manager.get_phase3_intro()
        examples = self.phase_manager.get_phase3_examples(batch_idx)
        full_message = intro + "\n\n" + examples
        self.llm_client.generate_response(full_message, max_tokens=100)
        batch_results["phases"]["phase_3"] = {"completed": True}

        # Phase 4: Make predictions again
        print("Phase 4: Making predictions after explanations...")
        intro = self.phase_manager.get_phase4_intro()
        reviews = self.phase_manager.get_phase4_reviews(batch_idx)
        predictions = self.llm_client.generate_predictions_for_reviews(reviews, intro)
        self.phase_manager.store_phase4_predictions(batch_idx, predictions)
        batch_results["phases"]["phase_4"] = {"predictions": predictions}


        # Store Phase 3-4 conversation snapshot before clearing
        phase34_conversation = {
            "batch_idx": batch_idx,
            "phases": "3-4",
            "messages": self.llm_client.conversation_messages.copy()
        }

        # Save both conversation snapshots
        self.batch_conversations.append(phase12_conversation)
        self.batch_conversations.append(phase34_conversation)

        # Clear conversation history before next batch
        print("\nClearing conversation history...")
        self.llm_client.clear_conversation()

        print(f"\nBatch {batch_idx + 1} complete!")

        return batch_results

    def run_complete_experiment(self) -> Dict[str, Any]:
        """Run all batches of the experiment"""
        print("\n" + "="*70)
        print("STARTING BATCHED MODEL SIMULATION EXPERIMENT")
        print("="*70)
        print(f"\nRunning {self.data_loader.get_num_batches()} batches with 4 examples each\n")

        # Record start time
        self.experiment_results["metadata"]["start_time"] = datetime.now().isoformat()

        try:
            # Run each batch
            for batch_idx in range(self.data_loader.get_num_batches()):
                batch_results = self.run_batch(batch_idx)
                self.experiment_results["batches"].append(batch_results)

            # Analyze overall results
            print("\n" + "="*70)
            print("ANALYZING RESULTS")
            print("="*70)

            analysis = self.phase_manager.get_prediction_comparison()
            self.experiment_results["analysis"] = analysis

            # Print results
            overall = analysis["overall"]
            print(f"\nOVERALL RESULTS (across all {self.data_loader.get_num_batches()} batches):")
            print(f"Phase 2 Accuracy: {overall['phase2_accuracy']:.1%} ({overall['phase2_correct']}/{overall['total_samples']})")
            print(f"Phase 4 Accuracy: {overall['phase4_accuracy']:.1%} ({overall['phase4_correct']}/{overall['total_samples']})")
            print(f"Improvement:      {overall['improvement']:+.1%}")
            print(f"Changed:          {overall['predictions_changed']}/{overall['total_samples']}")

            print(f"\nPER-BATCH RESULTS:")
            for batch_result in analysis["per_batch"]:
                idx = batch_result["batch_idx"]
                print(f"  Batch {idx + 1}: Phase2={batch_result['phase2_accuracy']:.1%}, "
                      f"Phase4={batch_result['phase4_accuracy']:.1%}, "
                      f"Δ={batch_result['improvement']:+.1%}")

            # Record end time
            self.experiment_results["metadata"]["end_time"] = datetime.now().isoformat()

            print("\n" + "="*70)
            print("EXPERIMENT COMPLETE")
            print("="*70)

        except Exception as e:
            print(f"\nExperiment failed: {str(e)}")
            self.experiment_results["error"] = str(e)
            raise

        return self.experiment_results

    def save_results(self, output_dir: str = ".") -> Dict[str, str]:
        """Save all experiment results including per-batch conversations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Save conversation for each batch
        print(f"\nSaving {len(self.batch_conversations)} conversation files...")
        for conv_data in self.batch_conversations:
            batch_idx = conv_data["batch_idx"]
            phases = conv_data["phases"]

            conv_file = os.path.join(
                output_dir,
                f"conversation_batch{batch_idx}_{phases.replace('-', '')}_{timestamp}.json"
            )

            conversation_json = {
                "metadata": {
                    "model_name": self.model_name,
                    "batch_idx": batch_idx,
                    "phases": phases,
                    "timestamp": datetime.now().isoformat(),
                    "total_messages": len(conv_data["messages"])
                },
                "system_message": self.system_template,
                "messages": conv_data["messages"]
            }

            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_json, f, indent=2, ensure_ascii=False)

            saved_files[f"conversation_batch{batch_idx}_{phases}"] = conv_file

        # Save experiment results
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_results, f, indent=2, ensure_ascii=False)
        saved_files["results"] = results_file

        print(f"\nFiles saved:")
        print(f"  - {len(self.batch_conversations)} conversation files")
        print(f"  - 1 results file: {results_file}")

        return saved_files