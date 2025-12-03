import json
from tqdm import tqdm
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class DataConfig:
    """Configuration for the experiment"""
    explanation_format: Literal["text_scores", "text_labels", "structured_text_scores", 
                                  "structured_text_labels", "top_words_scores", "top_words_labels", "none"]
    use_explanations: bool
    
class DataLoader:

    
    def __init__(self, groups_file: str, dev_data_file: str, prediction_type: str):
        """
        Initialize the experiment with the two JSON files.
        
        Args:
            groups_file: Path to JSON with test instances and their associated dev group IDs
            dev_data_file: Path to JSON with dev set samples, predictions, and (SHAP) explanations
        """

        self.prediction_type = prediction_type

        with open(groups_file, 'r') as f:
            self.groups = json.load(f)
        
        with open(dev_data_file, 'r') as f:
            self.dev_data = json.load(f)
        
        # Convert string keys to integers for dev_data if needed
        self.dev_data = {int(k): v for k, v in self.dev_data.items()}
        
        print(f"Loaded {len(self.groups)} test instances")
        print(f"Loaded {len(self.dev_data)} dev instances")
    
    def get_test_instance(self, test_id: str) -> str:
        """Get the test instance text"""
        return self.groups[test_id]["test_instance"]
    
    def get_dev_group_ids(self, test_id: str) -> List[int]:
        """Get the list of dev instance IDs associated with a test instance"""
        return self.groups[test_id]["dev_group"]
    
    def get_dev_predictions(self, test_id: str) -> List[int]:
        """Get the predictions for the dev group (useful for verification)"""
        return self.groups[test_id]["dev_predictions"]
    
    def get_dev_instance(self,
                        dev_id: int, 
                        include_explanation: bool = False, 
                        explanation_format: str = "text_labels") -> Dict:
        """
        Get a dev instance with optional SHAP explanation.
        
        Args: 
            dev_id: ID of the dev instance
            include_explanation: Whether to include SHAP explanation
            explanation_format: Which SHAP format to use (e.g., "text_labels", "top_words_scores")
        
        Returns:
            Dictionary with 'sample', 'prediction', and optionally 'explanation'
        """
        dev_instance = self.dev_data[dev_id]

        prediction_type = self.prediction_type

        if prediction_type == 'POSITIVE_NEGATIVE':
            if dev_instance['prediction'] == 0:
                prediction = 'NEGATIVE'
            elif dev_instance['prediction'] == 1:
                prediction = 'POSITIVE'

        if prediction_type == '0_1':
            prediction = dev_instance['prediction']            
        
        result = {
            'sample': dev_instance['sample'],
            'prediction': prediction
        }
        
        if include_explanation and explanation_format != "none":
            result['explanation'] = dev_instance['shap'][explanation_format]
        
        return result
    
    def format_dev_examples(self, test_id: str, config: DataConfig) -> str:
        """
        Format the 4 dev examples for presentation to the LLM.
        
        Args:
            test_id: ID of the test instance
            config: Experiment configuration
        
        Returns:
            Formatted string with dev examples
        """
        dev_ids = self.get_dev_group_ids(test_id)
        examples = []
        
        for i, dev_id in enumerate(dev_ids, 1):
            dev_instance = self.get_dev_instance(
                dev_id, 
                include_explanation=config.use_explanations,
                explanation_format=config.explanation_format
            )
            
            example_text = f"Example {i}:\n"
            example_text += f"Review: {dev_instance['sample']}\n"
            
            if config.use_explanations and 'explanation' in dev_instance:
                example_text += f"Explanation: {dev_instance['explanation']}\n"
            
            example_text += f"Model's Prediction: {dev_instance['prediction']}\n"
            examples.append(example_text)
        
        return "\n".join(examples)
    
    def get_all_test_ids(self) -> List[str]:
        """Get all test instance IDs"""
        return list(self.groups.keys())


class LLMPrompter:
    """
    Handles prompting the LLM to predict what the classification model would output.
    """
    
    def __init__(self, prediction_type: str, experiment: DataLoader):
        """
        Initialize with an experiment instance.
        
        Args:
            experiment: DataLoader instance with loaded data
        """
        self.experiment = experiment
        self.prediction_type = prediction_type
    
    def create_prompt(self, test_id: str, config: DataConfig, prediction_type: str,
                     chain_of_thought: bool = False) -> str:
        """
        Create a prompt for the LLM to predict the model's output.
        
        Args:
            test_id: ID of the test instance
            config: Experiment configuration (with/without explanations)
            chain_of_thought: Whether to ask for reasoning before prediction
        
        Returns:
            Formatted prompt string
        """
        # Get the formatted dev examples
        dev_examples = self.experiment.format_dev_examples(test_id, config)
        
        # Get the test instance
        test_instance = self.experiment.get_test_instance(test_id)
        
        # Build the prompt
        prompt = "###Task Description:\n"
        prompt += "You are given 4 examples of movie reviews with the predictions made by a sentiment classification model "
        if prediction_type == '0_1':
            prompt += "The predictions are 0 (NEGATIVE) or 1 (POSITIVE)"
        elif prediction_type == 'POSITIVE_NEGATIVE':
            prompt += "The predictions are NEGATIVE or POSITIVE"
        
        if config.use_explanations:
            prompt += "Each example includes an explanation showing which parts of the text influenced the model's decision. "
        
        prompt += "Your task is to analyze the model's behavior pattern and predict what the model would output for a new test review.\n\n"
        
        prompt += "###Examples from the model:\n"
        prompt += dev_examples + "\n\n"
        
        prompt += "###Test Review:\n"
        prompt += f"{test_instance}\n\n"
        
        prompt += "###Question:\n"
        prompt += "Based on the model's behavior in the examples above, what would this classification model predict for the test review?\n\n"
        
        if chain_of_thought:
            prompt += "###Instructions:\n"
            prompt += "First, briefly explain your reasoning. Then provide your final prediction.\n\n"
            prompt += "###Answer:\n"
            prompt += "Reasoning: [Your reasoning here]\n"
            if prediction_type == '0_1':
                prompt += "Prediction: [0 or 1]"
            elif prediction_type == 'POSITIVE_NEGATIVE':
                prompt += "Prediction: [POSITVE or NEGATIVE]"
        else:
            if prediction_type == '0_1':
                prompt += "###Answer (respond with only '0' or '1'):\n"
            elif prediction_type == 'POSITIVE_NEGATIVE':
                prompt += "###Answer (respond with only 'POSITIVE' or 'NEGATIVE'):\n"
        
        return prompt
    
    def extract_prediction(self, llm_response: str, 
                          chain_of_thought: bool = False) -> Optional[int]:
        """
        Extract the prediction (0 or 1) from the LLM's response.
        
        Args:
            llm_response: The LLM's text response
            chain_of_thought: Whether the response includes reasoning
        
        Returns:
            Predicted label (0 or 1) or None if extraction failed
        """
        response = llm_response.strip()
        
        if chain_of_thought:
            # Look for "Prediction: X" pattern
            if "Prediction:" in response:
                prediction_part = response.split("Prediction:")[-1].strip()
                response = prediction_part
        
        # Try to extract 0 or 1
        response = response.strip().strip('"\'.,!?')
        
        if response == '0' or response == 'NEGATIVE' or '0' in response[:5]:
            return 0
        elif response == '1' or response == 'POSITIVE' or '1' in response[:5]:
            return 1
        else:
            # More flexible extraction
            has_zero = '0' in response or 'NEGATIVE' in response
            has_one = '1' in response or 'POSITIVE' in response
            
            if has_zero and not has_one:
                return 0
            elif has_one and not has_zero:
                return 1
            else:
                print(f"Warning: Could not extract prediction from: {response}")
                return None


# Example usage function
def run_single_prediction(experiment: DataLoader, 
                         prompter: LLMPrompter,
                         prediction_type: str,
                         test_id: str,
                         chain_of_thought: bool,
                         config: DataConfig,
                         llm_function) -> dict:
    """
    Run a single prediction for a test instance.
    
    Args:
        experiment: The experiment instance
        prompter: The prompter instance
        test_id: ID of test instance
        config: Experiment configuration
        llm_function: Function that takes a prompt and returns LLM response
    
    Returns:
        Dictionary with results
    """
    # Create prompt
    prompt = prompter.create_prompt(test_id, config, prediction_type, chain_of_thought)
    
    # Get LLM response
    llm_response = llm_function(prompt)
    
    # Extract prediction
    predicted_label_LLM = prompter.extract_prediction(llm_response, chain_of_thought)
    
    # Get actual dev predictions for reference
    dev_predictions = experiment.get_dev_predictions(test_id)
    
    return {
        'test_id': test_id,
        'prompt': prompt,
        'llm_response': llm_response,
        'predicted_label_LLM': predicted_label_LLM,
        'dev_predictions': dev_predictions,
        'config': {
            'use_explanations': config.use_explanations,
            'explanation_format': config.explanation_format
        }
    }


class PrometheusLLM:
    """
    Wrapper for the Prometheus model to generate predictions.
    Always runs on CUDA.
    """
    
    def __init__(self, model_name: str = "Unbabel/M-Prometheus-3B", 
                 max_new_tokens: int = 512,
                 temperature: float = 0.1):
        """
        Initialize the Prometheus model.
        
        Args:
            model_name: HuggingFace model name
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        """
        print(f"Loading model {model_name}...")
        
        # Always use CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This model requires GPU.")
        
        self.device = "cuda"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16
        ).to(self.device)

        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        print("Model loaded successfully!")
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max_new_tokens
            temperature: Override default temperature
        
        Returns:
            Generated text
        """
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        
        # Format as chat message
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        # Apply chat template and tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9 if temp > 0 else None,
            )
        
        # Decode only the generated part (exclude input)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def clear_cache(self):
        """
        Clear GPU cache to free memory between predictions.
        Important for ensuring independence between test instances.
        """
        torch.cuda.empty_cache()
    
    def __call__(self, prompt: str) -> str:
        """
        Make the class callable like a function.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated response
        """
        return self.generate(prompt)


def test_experiment(groups_file: str, 
                   dev_data_file: str,
                   num_test_instances: int,
                   prediction_type: str,
                   chain_of_thought: bool,
                   explanation_format: str = "text_labels",
                   use_explanations: bool = True,
                   output_file: str = "test_results.json",
                   model_name: str = "Unbabel/M-Prometheus-3B"):
    """
    Test the experiment on a limited number of instances.
    
    Args:
        groups_file: Path to groups JSON
        dev_data_file: Path to dev data JSON
        num_test_instances: Number of test instances to process
        explanation_format: Which SHAP format to use
        use_explanations: Whether to include explanations
        output_file: Where to save results
        model_name: LLM model to use
    """
    # Initialize experiment
    print("Initializing experiment...")
    experiment = DataLoader(groups_file, dev_data_file, prediction_type)
    prompter = LLMPrompter(prediction_type, experiment)
    
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = PrometheusLLM(model_name=model_name)
    
    # Create config
    config = DataConfig(
        explanation_format=explanation_format,
        use_explanations=use_explanations
    )
    
    # Get test IDs (limit to num_test_instances)
    all_test_ids = experiment.get_all_test_ids()
    test_ids = all_test_ids[:num_test_instances]
    
    print(f"\nRunning test with:")
    print(f"  - Explanations: {use_explanations}")
    print(f"  - Format: {explanation_format}")
    print(f"  - Test instances: {len(test_ids)}")
    print(f"  - Output file: {output_file}\n")
    
    # Store results
    results = []
    
    # Run predictions for each test instance
    for i, test_id in enumerate(tqdm(test_ids, desc="Processing"), 1):
        
        result = run_single_prediction(
            experiment=experiment,
            prompter=prompter,
            test_id=test_id,
            prediction_type=prediction_type,
            chain_of_thought=chain_of_thought,
            config=config,
            llm_function=llm
        )
        
        results.append(result)
        
        
        # Clear cache to ensure independence
        llm.clear_cache()
            
    
    # Save results
    print(f"\n\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = [r for r in results if 'predicted_label_LLM' in r and r['predicted_label_LLM'] is not None]
    failed = [r for r in results if 'error' in r or r.get('predicted_label_LLM') is None]
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total instances: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        predictions = [r['predicted_label_LLM'] for r in successful]
        print(f"\nPrediction distribution:")
        print(f"  Label 0 (NEGATIVE): {predictions.count(0)}")
        print(f"  Label 1 (POSITIVE): {predictions.count(1)}")
    
    print(f"\nResults saved to: {output_file}")
    print("="*50)
    
    return results