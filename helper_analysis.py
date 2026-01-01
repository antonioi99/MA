import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class PredictionAnalyzer:
    """
    A class to analyze and compare LLM predictions against original model predictions.
    
    Attributes:
        llm_predictions (Dict[str, int]): Dictionary of LLM predictions
        original_predictions (Dict[str, int]): Dictionary of original predictions
        results (Dict): Analysis results after calculation
    """
    
    def __init__(self):
        """Initialize the PredictionAnalyzer."""
        self.llm_predictions: Dict[str, int] = {}
        self.original_predictions: Dict[str, int] = {}
        self.results: Optional[Dict] = None
    
    def load_json_file(self, filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_llm_predictions(self, data: List[Dict]) -> 'PredictionAnalyzer':
        """
        Extract and store LLM predictions from data.
        
        Args:
            data: List of dictionaries containing test_id and predicted_label_LLM
            
        Returns:
            Self for method chaining
        """
        self.llm_predictions = {}
        for item in data:
            test_id = item.get('test_id')
            llm_pred = item.get('predicted_label_LLM')
            if test_id is not None and llm_pred is not None:
                self.llm_predictions[test_id] = llm_pred
        return self
    
    def load_llm_predictions_from_file(self, filepath: str) -> 'PredictionAnalyzer':
        """
        Load LLM predictions from a JSON file.
        
        Args:
            filepath: Path to the JSON file with LLM predictions
            
        Returns:
            Self for method chaining
        """
        data = self.load_json_file(filepath)
        return self.load_llm_predictions(data)
    
    def load_original_predictions(self, data: Dict) -> 'PredictionAnalyzer':
        """
        Extract and store original predictions from data.
        
        Args:
            data: Dictionary with test_ids as keys and prediction info as values
            
        Returns:
            Self for method chaining
        """
        self.original_predictions = {}
        for test_id, item_data in data.items():
            if 'prediction' in item_data:
                self.original_predictions[test_id] = item_data['prediction']
        return self
    
    def load_original_predictions_from_file(self, filepath: str) -> 'PredictionAnalyzer':
        """
        Load original predictions from a JSON file.
        
        Args:
            filepath: Path to the JSON file with original predictions
            
        Returns:
            Self for method chaining
        """
        data = self.load_json_file(filepath)
        return self.load_original_predictions(data)
    
    def calculate_accuracy(self) -> 'PredictionAnalyzer':
        """
        Calculate accuracy and detailed statistics comparing LLM predictions 
        to original predictions.
        
        Returns:
            Self for method chaining
        """
        # Find common test_ids
        common_ids = set(self.llm_predictions.keys()) & set(self.original_predictions.keys())
        
        if not common_ids:
            self.results = {
                'error': 'No common test_ids found between the two datasets',
                'llm_count': len(self.llm_predictions),
                'original_count': len(self.original_predictions)
            }
            return self
        
        # Calculate matches
        matches = sum(1 for tid in common_ids 
                      if self.llm_predictions[tid] == self.original_predictions[tid])
        total = len(common_ids)
        accuracy = matches / total if total > 0 else 0
        
        # Detailed breakdown
        confusion = {
            'true_positives': 0,   # Both predicted 1
            'true_negatives': 0,   # Both predicted 0
            'false_positives': 0,  # LLM predicted 1, original was 0
            'false_negatives': 0   # LLM predicted 0, original was 1
        }
        
        mismatches = []
        for tid in common_ids:
            llm_pred = self.llm_predictions[tid]
            orig_pred = self.original_predictions[tid]
            
            if llm_pred == 1 and orig_pred == 1:
                confusion['true_positives'] += 1
            elif llm_pred == 0 and orig_pred == 0:
                confusion['true_negatives'] += 1
            elif llm_pred == 1 and orig_pred == 0:
                confusion['false_positives'] += 1
                mismatches.append({'test_id': tid, 'llm': llm_pred, 'original': orig_pred})
            else:  # llm_pred == 0 and orig_pred == 1
                confusion['false_negatives'] += 1
                mismatches.append({'test_id': tid, 'llm': llm_pred, 'original': orig_pred})
        
        # Calculate additional metrics
        precision = (confusion['true_positives'] / 
                    (confusion['true_positives'] + confusion['false_positives']) 
                    if (confusion['true_positives'] + confusion['false_positives']) > 0 else 0)
        
        recall_TPR = (confusion['true_positives'] / 
                 (confusion['true_positives'] + confusion['false_negatives'])
                 if (confusion['true_positives'] + confusion['false_negatives']) > 0 else 0)

        recall_TNR = (confusion['true_negatives'] / 
                (confusion['true_negatives'] + confusion['false_positives'])
                if (confusion['true_negatives'] + confusion['false_positives']) > 0 else 0)
        
        f1_score = (2 * precision * recall_TPR / (precision + recall_TPR) 
                   if (precision + recall_TPR) > 0 else 0)
        
        self.results = {
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100,
            'total_compared': total,
            'correct_predictions': matches,
            'incorrect_predictions': total - matches,
            'confusion_matrix': confusion,
            'precision': precision,
            'recall_TPR': recall_TPR,
            'recall_TNR': recall_TNR,
            'f1_score': f1_score,
            'mismatches': mismatches,
            'missing_in_llm': len(set(self.original_predictions.keys()) - set(self.llm_predictions.keys())),
            'missing_in_original': len(set(self.llm_predictions.keys()) - set(self.original_predictions.keys()))
        }
        
        return self

        
    def save_results(self, filepath: str):
        """
        Save the analysis results to a JSON file.
        
        Args:
            filepath: Path where to save the results
            
        Raises:
            ValueError: If calculate_accuracy hasn't been called yet
        """
        if self.results is None:
            raise ValueError("No results available. Call calculate_accuracy() first.")
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f" Results saved to '{filepath}'")
    
    
    def analyze(self, llm_file: str, original_file: str,
                save_to: Optional[str] = None) -> Dict:
        """
        Complete analysis pipeline: load files, calculate metrics, and optionally print/save.
        
        Args:
            llm_file: Path to JSON file with LLM predictions
            original_file: Path to JSON file with original predictions
            print_output: Whether to print results to console
            save_to: Optional path to save results as JSON
            
        Returns:
            Dictionary with analysis results
        """
        self.load_llm_predictions_from_file(llm_file)
        self.load_original_predictions_from_file(original_file)
        self.calculate_accuracy()
        
        
        if save_to:
            self.save_results(save_to)
        
        return self.results
    
    def reset(self):
        """Reset the analyzer, clearing all loaded data and results."""
        self.llm_predictions = {}
        self.original_predictions = {}
        self.results = None


# Convenience function for quick analysis
def quick_analyze(llm_file: str, original_file: str, 
                  save_to: Optional[str] = None) -> Dict:
    """
    Quick analysis function for one-line usage.
    
    Args:
        llm_file: Path to JSON file with LLM predictions
        original_file: Path to JSON file with original predictions
        save_to: Optional path to save results as JSON
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = PredictionAnalyzer()
    return analyzer.analyze(llm_file, original_file, save_to)
