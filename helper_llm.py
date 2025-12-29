import json
from tqdm import tqdm
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class DataConfig:
    """Configuration for the experiment"""
    explanation_format: Literal["text_scores", "text_labels", "structured_text_scores", 
                                  "structured_text_labels", "top_words_scores", "top_words_labels", "none"]
    use_explanations: bool
    
class DataLoader:

    
    def __init__(self, groups_file: str, dev_data_file: str, dev_data_predictions: str):
        """
        Initialize the experiment with the two JSON files.
        
        Args:
            groups_file: Path to JSON with test instances and their associated dev group IDs
            dev_data_file: Path to JSON with dev set samples, predictions, and (SHAP) explanations
        """


        with open(groups_file, 'r') as f:
            self.groups = json.load(f)
        
        with open(dev_data_file, 'r') as f:
            self.dev_data = json.load(f)

        with open(dev_data_predictions, 'r') as f:
            self.dev_data_predictions = json.load(f)
        

        self.dev_data = {int(k): v for k, v in self.dev_data.items()}
        self.dev_data_predictions = {int(k): v for k, v in self.dev_data_predictions.items()}
        
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

    def get_dev_instance_prediction(self, dev_id):
        """Get the predictions for the current dev_instance"""
        return self.dev_data_predictions[dev_id]['prediction']
    
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
        dev_prediction = self.dev_data_predictions[dev_id]['prediction']


        if dev_prediction == 0:
            prediction = 'NEGATIVE'
        elif dev_prediction == 1:
            prediction = 'POSITIVE'
         
        
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

    PROMPT_TEMPLATE = """###Task Description:
    You are given 4 examples of movie reviews with the predictions made by a sentiment classification model. The predictions are {pred_order}. {explanation_desc}
    Your task is to analyze the model's behavior pattern and predict what the model would output for a new test review.

    ###Examples from the model:
    {dev_examples}

    ###Test Review:
    {test_instance}

    ###Question:
    Based on the model's behavior in the examples above, what would this classification model predict for the test review?

    {answer_section}
    """

    
    def __init__(self, experiment: DataLoader, pred_order):
        """
        Initialize with an experiment instance.
        
        Args:
            experiment: DataLoader instance with loaded data
        """
        self.experiment = experiment
        self.pred_order = pred_order
    

    def create_prompt(
        self,
        test_id: str,
        config: DataConfig,
        chain_of_thought: bool = False
    ) -> tuple[str, str]:
        """
        Returns:
            full_prompt: prompt with dev examples and test instance
            base_prompt: same prompt, but without dev examples and test instance
        """

        if self.pred_order == 'pos_neg':
            pred_order_str = "'POSITIVE' or 'NEGATIVE'"
        elif self.pred_order == 'neg_pos':
            pred_order_str = "'NEGATIVE' or 'POSITIVE'"  

        # Explanation description
        explanation_desc = ""
        if config.use_explanations:
            explanation_desc = (
                "Each example includes an explanation showing which parts of the text influenced the model's decision. "
            )

        # Answer section
        if chain_of_thought:

            answer_section = (
                # "###Instructions:\n"
                # "First, briefly explain your reasoning. Then provide your final prediction.\n\n"
                f"###Answer (explain in 2-3 sentences why your answer is {pred_order_str}):"
                # "Reasoning: [Your reasoning here]\n"
                # "Prediction: [POSITIVE or NEGATIVE]"
            )
        else:
            answer_section = (
                f"###Answer (reply only with {pred_order_str}):"
            )

        # Instance-specific content
        dev_examples = self.experiment.format_dev_examples(test_id, config)
        test_instance = self.experiment.get_test_instance(test_id)

        # Full prompt
        full_prompt = LLMPrompter.PROMPT_TEMPLATE.format(
            pred_order=pred_order_str,
            explanation_desc=explanation_desc,
            dev_examples=dev_examples,
            test_instance=test_instance,
            answer_section=answer_section,
        )

        # Base prompt (without examples & test instance)
        base_prompt = LLMPrompter.PROMPT_TEMPLATE.format(
            pred_order=pred_order_str,
            explanation_desc=explanation_desc,
            dev_examples="[DEV_EXAMPLES]",
            test_instance="[TEST_INSTANCE]",
            answer_section=answer_section,
        )

        return full_prompt, base_prompt

    
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
        
        
        response = response.strip().strip('"\'.,!?')

        positive = 'positive'
        negative = 'negative'
        positive_count = response.count(positive)
        negative_count = response.count(negative)

        if 'NEGATIVE' in response:
            return 0
        elif 'POSITIVE' in response:
            return 1
        elif positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return 0
        


# Example usage function
def run_single_prediction(experiment: DataLoader, 
                         prompter: LLMPrompter,
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
    full_prompt, base_prompt = prompter.create_prompt(test_id, config, chain_of_thought)
    
    # Get LLM response
    llm_response = llm_function(full_prompt)
    
    # Extract prediction
    predicted_label_LLM = prompter.extract_prediction(llm_response, chain_of_thought)
    
    # Get actual dev predictions for reference
    dev_predictions = experiment.get_dev_predictions(test_id)
    
    return {
        'test_id': test_id,
        'prompt': base_prompt,
        'full_prompt': full_prompt,
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
    
    def __init__(self,
                max_new_tokens: int,
                model_name: str = "Unbabel/M-Prometheus-3B", 
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
                   dev_data_predictions: str,
                   num_test_instances: int,
                   pred_order: str,
                   chain_of_thought: bool,
                   start: int,
                   explanation_format: str,
                   use_explanations: bool,
                   output_file: str,
                   max_new_tokens: int,
                   model_name: str):
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
    experiment = DataLoader(groups_file, dev_data_file, dev_data_predictions)
    prompter = LLMPrompter(experiment, pred_order)
    
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = PrometheusLLM(max_new_tokens=max_new_tokens, model_name=model_name)
    
    # Create config
    config = DataConfig(
        explanation_format=explanation_format,
        use_explanations=use_explanations
    )
    
    # Get test IDs (limit to num_test_instances)
    all_test_ids = experiment.get_all_test_ids()
    end = start + num_test_instances
    test_ids = all_test_ids[start:end]
    
    print(f"\nRunning test with:")
    print(f"  - Explanations: {use_explanations}")
    print(f"  - Format: {explanation_format}")
    print(f"  - Test instances: {len(test_ids)}")
    print(f"  - Output file: {output_file}\n")
    
    # Store results
    results = []

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    
    # Run predictions for each test instance
    for i, test_id in enumerate(tqdm(test_ids, desc="Processing"), 1):
        
        result = run_single_prediction(
            experiment=experiment,
            prompter=prompter,
            test_id=test_id,
            chain_of_thought=chain_of_thought,
            config=config,
            llm_function=llm
        )
        
        results.append(result)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
             
        # Clear cache to ensure independence
        llm.clear_cache()
            
    
    # Save results
    print(f"\n\nSaving results to {output_file}...")
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)
    
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