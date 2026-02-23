import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import os
from statsmodels.stats.contingency_tables import mcnemar, SquareTable
from dataclasses import dataclass


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
        
        # FIXED: Sort common_ids to ensure consistent ordering
        sorted_common_ids = sorted(common_ids)
        
        # Calculate matches
        matches = sum(1 for tid in sorted_common_ids 
                      if self.llm_predictions[tid] == self.original_predictions[tid])
        total = len(sorted_common_ids)
        accuracy = matches / total if total > 0 else 0
        
        # Detailed breakdown
        confusion = {
            'true_positives': 0,   # Both predicted 1
            'true_negatives': 0,   # Both predicted 0
            'false_positives': 0,  # LLM predicted 1, original was 0
            'false_negatives': 0   # LLM predicted 0, original was 1
        }
        
        mismatches = []
        for tid in sorted_common_ids:
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
            'accuracy': round(accuracy, 4),
            'total_compared': total,
            'correct_predictions': matches,
            'incorrect_predictions': total - matches,
            'confusion_matrix': confusion,
            'precision': round(precision, 4),
            'recall_TPR': round(recall_TPR, 4),
            'recall_TNR': round(recall_TNR, 4),
            'f1_score': round(f1_score, 4),
            'mismatches': mismatches,
            'missing_in_llm': len(set(self.original_predictions.keys()) - set(self.llm_predictions.keys())),
            'missing_in_original': len(set(self.llm_predictions.keys()) - set(self.original_predictions.keys())),
            'test_ids_ordered': sorted_common_ids  # FIXED: Store ordered test_ids
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

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    llm: str  # llama, qwen, prometheus
    task_type: str  # single, pairwise
    cot: str  # no_chain_of_thought, chain_of_thought_True
    prompting: str  # neg_pos, pos_neg
    explanation: str # shap, attention, lime
    
    def __str__(self):
        return f"{self.llm}_{self.explanation}_{self.task_type}_{self.cot}_{self.prompting}"
    
    def to_label(self):
        """Generate a valid LaTeX label"""
        return f"tab:{self.llm}-{self.explanation}-{self.task_type}-{self.cot}-{self.prompting}"


class McNemarAnalyzer:
    """
    Analyzer for comparing explanation formats using McNemar's test
    """
    
    EXPLANATION_FORMATS = [
        "baseline", "text_scores", "text_labels", 
        "structured_text_scores", "structured_text_labels",
        "top_words_scores", "top_words_labels", 
        "natural_words", "part_of_speech"
    ]
    
    def __init__(self, base_path: str = "analysis"):
        self.base_path = base_path
        self.results_cache = {}
        self.latex_tables = []  # Store LaTeX output
        
    def get_file_path(self, config: ExperimentConfig, format_name: str) -> str:
        """Construct file path for a given configuration and format"""
        return os.path.join(self.base_path, config.llm, config.explanation, config.task_type, 
                           config.cot, config.prompting, f"results_{format_name}.json")
    
    def load_results(self, config: ExperimentConfig, format_name: str) -> Dict:
        """Load results from JSON file with caching"""
        cache_key = (str(config), format_name)
        
        if cache_key not in self.results_cache:
            file_path = self.get_file_path(config, format_name)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r') as f:
                self.results_cache[cache_key] = json.load(f)
        
        return self.results_cache[cache_key]
    
    def get_correctness_arrays(self, config: ExperimentConfig, format_name: str) -> np.ndarray:
        """
        FIXED: Extract correctness array (True/False for each test example)
        from results JSON using the stored ordered test_ids
        """
        results = self.load_results(config, format_name)
        
        # FIXED: Check if test_ids_ordered exists in the results
        if 'test_ids_ordered' not in results:
            # Fallback for old format - try to reconstruct
            print(f"WARNING: 'test_ids_ordered' not found in {format_name}. Using fallback method.")
            n_total = results['total_compared']
            incorrect_ids = set(m['test_id'] for m in results['mismatches'])
            
            # Try to infer test_ids from mismatches (this is a guess!)
            # This assumes test_ids are strings like "0", "1", "2", etc.
            correct = np.array([str(i) not in incorrect_ids for i in range(n_total)])
            
            calculated_correct = correct.sum()
            expected_correct = results['correct_predictions']
            
            if calculated_correct != expected_correct:
                raise AssertionError(
                    f"Mismatch in correct count for {format_name}: "
                    f"calculated {calculated_correct}, expected {expected_correct}. "
                    f"Please regenerate results files with the fixed PredictionAnalyzer."
                )
        else:
            # NEW: Use the stored ordered test_ids
            test_ids_ordered = results['test_ids_ordered']
            incorrect_ids = set(m['test_id'] for m in results['mismatches'])
            
            # Create boolean array based on actual test_ids
            correct = np.array([tid not in incorrect_ids for tid in test_ids_ordered])
            
            # Verify
            assert correct.sum() == results['correct_predictions'], \
                f"Mismatch in correct count for {format_name}"
        
        return correct
    
    def get_correctness_arrays_with_common_ids(self, config: ExperimentConfig, 
                                                baseline_format: str, 
                                                test_format: str) -> tuple:
        """
        Get correctness arrays for baseline and test formats, filtered to common test_ids.
        
        Returns:
            tuple: (correct_baseline, correct_test, common_test_ids)
        """
        baseline_results = self.load_results(config, baseline_format)
        test_results = self.load_results(config, test_format)
        
        # Get test_ids from both formats
        baseline_test_ids = baseline_results.get('test_ids_ordered', [])
        test_test_ids = test_results.get('test_ids_ordered', [])
        
        if not baseline_test_ids or not test_test_ids:
            raise ValueError(f"Both {baseline_format} and {test_format} must have test_ids_ordered")
        
        # Find common test_ids (preserving order from baseline)
        test_test_ids_set = set(test_test_ids)
        common_test_ids = [tid for tid in baseline_test_ids if tid in test_test_ids_set]
        
        if not common_test_ids:
            raise ValueError(f"No common test_ids between {baseline_format} and {test_format}")
        
        # Get incorrect IDs for both formats
        baseline_incorrect = set(m['test_id'] for m in baseline_results['mismatches'])
        test_incorrect = set(m['test_id'] for m in test_results['mismatches'])
        
        # Create correctness arrays for common test_ids only
        correct_baseline = np.array([tid not in baseline_incorrect for tid in common_test_ids])
        correct_test = np.array([tid not in test_incorrect for tid in common_test_ids])
        
        return correct_baseline, correct_test, common_test_ids
    
    def compare_formats(self, config: ExperimentConfig, 
                       baseline_format: str, 
                       test_format: str) -> Dict:
        """
        Compare two formats using McNemar's test
        Uses only common test_ids between the two formats.
        
        Returns:
            Dictionary with accuracy metrics and test results
        """
        # Get correctness arrays filtered to common test_ids
        correct_baseline, correct_test, common_test_ids = \
            self.get_correctness_arrays_with_common_ids(config, baseline_format, test_format)
        
        n_total = len(common_test_ids)
        
        # Create contingency table
        contingency_data = pd.DataFrame({
            'baseline': correct_baseline,
            'test': correct_test
        })
        
        contingency_table = SquareTable.from_data(contingency_data)
        
        # Get marginal probabilities (accuracies)
        marginal_row_prob, marginal_col_prob = contingency_table.marginal_probabilities
        accuracy_baseline = marginal_row_prob[True]
        accuracy_test = marginal_col_prob[True]
        
        # Extract contingency table cells
        table = contingency_table.table
        both_correct = table[1, 1]
        baseline_correct_test_wrong = table[1, 0]  # b: got worse
        baseline_wrong_test_correct = table[0, 1]  # c: got better
        both_wrong = table[0, 0]
        
        # Perform McNemar's test
        # Use chi-square approximation for large samples
        mcnemar_result = mcnemar(table, exact=False, correction=True)
        
        # Calculate change metrics
        absolute_change = accuracy_test - accuracy_baseline
        relative_change = (accuracy_test / accuracy_baseline - 1.0) if accuracy_baseline > 0 else 0
        
        return {
            'baseline_format': baseline_format,
            'test_format': test_format,
            'n_total': n_total,
            'accuracy_baseline': accuracy_baseline,
            'accuracy_test': accuracy_test,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'both_correct': both_correct,
            'got_worse': baseline_correct_test_wrong,
            'got_better': baseline_wrong_test_correct,
            'both_wrong': both_wrong,
            'mcnemar_statistic': mcnemar_result.statistic,
            'p_value': mcnemar_result.pvalue,
            'significant': mcnemar_result.pvalue < 0.05
        }
    
    def analyze_configuration(self, config: ExperimentConfig) -> pd.DataFrame:
        """
        Analyze all explanation formats against baseline for a single configuration
        
        Returns:
            DataFrame in the format you requested (9 formats x 11 columns)
        """
        results = []
        
        for format_name in self.EXPLANATION_FORMATS:
            if format_name == "baseline":
                # Baseline row: only show baseline accuracy
                baseline_data = self.load_results(config, "baseline")
                row = {
                    'baseline': baseline_data['accuracy'],
                    'text_scores': np.nan,
                    'text_labels': np.nan,
                    'structured_text_scores': np.nan,
                    'structured_text_labels': np.nan,
                    'top_words_scores': np.nan,
                    'top_words_labels': np.nan,
                    'natural_words': np.nan,
                    'part_of_speech': np.nan,
                    'change': np.nan,
                    'p': np.nan
                }
            else:
                # Compare format against baseline
                comparison = self.compare_formats(config, "baseline", format_name)
                
                row = {
                    'baseline': comparison['accuracy_baseline'],
                    'text_scores': np.nan,
                    'text_labels': np.nan,
                    'structured_text_scores': np.nan,
                    'structured_text_labels': np.nan,
                    'top_words_scores': np.nan,
                    'top_words_labels': np.nan,
                    'natural_words': np.nan,
                    'part_of_speech': np.nan,
                    'change': comparison['relative_change'],
                    'p': comparison['p_value']
                }
                # Fill in the test format accuracy
                row[format_name] = comparison['accuracy_test']
            
            results.append(row)
        
        # Create DataFrame with format names as index
        df = pd.DataFrame(results, index=self.EXPLANATION_FORMATS)
        
        return df
    
    def format_results_table(self, df: pd.DataFrame):
        """
        Format the results table with proper styling (requires jinja2)
        If jinja2 is not available, returns the raw DataFrame
        """
        try:
            styled = df.style.format({
                'baseline': '{:.4f}',
                'text_scores': '{:.4f}',
                'text_labels': '{:.4f}',
                'structured_text_scores': '{:.4f}',
                'structured_text_labels': '{:.4f}',
                'top_words_scores': '{:.4f}',
                'top_words_labels': '{:.4f}',
                'natural_words': '{:.4f}',
                'part_of_speech': '{:.4f}',
                'change': '{:.2%}',
                'p': '{:.4f}'
            }, na_rep='')
            return styled
        except AttributeError:
            # jinja2 not installed, return formatted DataFrame
            print("Warning: jinja2 not installed. Returning plain DataFrame.")
            return df

    def to_latex(self, df: pd.DataFrame, config: ExperimentConfig) -> str:
        """
        Convert DataFrame to LaTeX table format - simplified version
        Only shows: Format name, Baseline accuracy, Test accuracy, Change, p-value
        """
        # Format the values manually
        def format_value(val, is_change=False, is_pvalue=False):
            if pd.isna(val):
                return '--'
            if is_pvalue:
                return f'{val:.4f}'
            elif is_change:
                return f'{val*100:.2f}\\%'  # Percentage symbol with single backslash
            else:  # accuracy as decimal
                return f'{val:.4f}'
        
        # Generate caption
        caption = (f"McNemar's Test: {config.llm.upper()} - "
                f"{config.task_type.title()} - "
                f"{'CoT' if 'True' in config.cot else 'No CoT'} - "
                f"{config.prompting.replace('_', ' ')}")
        
        # Start building LaTeX table with only 5 columns
        latex_lines = []
        latex_lines.append(r'\begin{table}[htbp]')
        latex_lines.append(r'\centering')
        latex_lines.append(f'\\caption{{{caption}}}')
        latex_lines.append(f'\\label{{{config.to_label()}}}')
        latex_lines.append(r'\begin{tabular}{lrrrr}')  # Only 5 columns now
        latex_lines.append(r'\toprule')
        
        # Simplified header - use \% directly in raw string
        header = r'Format & Baseline & Accuracy & Change (\%) & p-value \\'
        latex_lines.append(header)
        latex_lines.append(r'\midrule')
        
        # First row - baseline
        first_row = df.iloc[0]
        latex_lines.append(f'Baseline & {format_value(first_row["baseline"])} & -- & -- & -- \\\\')
        
        # Data rows - only non-baseline
        for idx, row in df.iloc[1:].iterrows():
            # Find which column has the test accuracy (the non-NaN one that's not baseline/change/p)
            test_acc = row[idx]  # The column name matches the index name
            
            # Format index name nicely
            format_name = idx.replace("_", " ").title()
            
            line = (f'{format_name} & '
                    f'{format_value(row["baseline"])} & '
                    f'{format_value(test_acc)} & '
                    f'{format_value(row["change"], is_change=True)} & '
                    f'{format_value(row["p"], is_pvalue=True)} \\\\')
            latex_lines.append(line)
        
        # End table
        latex_lines.append(r'\bottomrule')
        latex_lines.append(r'\end{tabular}')
        latex_lines.append(r'\end{table}')
        
        latex_str = '\n'.join(latex_lines)
        
        # DON'T do the replace here anymore since we're handling % correctly
        return latex_str

    def save_latex_table(self, df: pd.DataFrame, config: ExperimentConfig, 
                        output_file: Optional[str] = None):
        """
        Save a single table to LaTeX file
        
        Args:
            df: Results DataFrame
            config: Experiment configuration
            output_file: Output file path (optional)
        """
        latex_str = self.to_latex(df, config)
        
        if output_file is None:
            output_file = f"results_{config}.tex"
        
        with open(output_file, 'w', encoding='UTF-8') as f:
            f.write(latex_str)
        
        print(f"Saved LaTeX table to: {output_file}")
        
        # Also store in memory for combined export
        self.latex_tables.append(latex_str)
    
    def save_all_latex_tables(self, output_file: str = "all_results.tex"):
        """
        Save all accumulated LaTeX tables to a single file
        
        Args:
            output_file: Output file path
        """
        if not self.latex_tables:
            print("No tables to save. Run analyze_configuration() first.")
            return
        
        with open(output_file, 'w', encoding='UTF-8') as f:
            # Join all tables with newlines
            combined = '\n\n'.join(self.latex_tables)
            f.write(combined)
        
        print(f"Saved {len(self.latex_tables)} LaTeX tables to: {output_file}")
    
    def analyze_and_save_all(self, output_dir: str = "latex_results"):
        """
        Analyze all configurations and save both individual and combined LaTeX files
        
        Args:
            output_dir: Directory to save output files
        """
        print(f"\n{'='*80}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_configs = self.get_all_configurations()
        successful_analyses = 0
        
        for config in all_configs:
            try:
                # Analyze configuration
                results_df = self.analyze_configuration(config)
                
                # Save individual LaTeX file
                individual_file = os.path.join(output_dir, f"results_{config}.tex")
                self.save_latex_table(results_df, config, individual_file)
                
                successful_analyses += 1
                
            except FileNotFoundError as e:
                print(f"Skipping {config}: {e}")
            except Exception as e:
                print(f"Error processing {config}: {e}")
        
        # Save combined file
        combined_file = os.path.join(output_dir, "all_results_combined.tex")
        self.save_all_latex_tables(combined_file)
        
        print(f"\n{'='*80}")
        print(f"Analysis complete!")
        print(f"Successfully analyzed: {successful_analyses}/{len(all_configs)} configurations")
        print(f"Output directory: {output_dir}")
    
    def get_all_configurations(self) -> List[ExperimentConfig]:
        """
        Generate all possible configuration combinations
        """
        llms = ['llama', 'qwen']
        task_types = ['single']
        promptings = ['neg_pos', 'pos_neg']
        explanation_types = ['attention', 'lime', 'shap']
        # CoT = ['chain_of_thought_True', 'no_chain_of_thought'] #chain of though is not being used
        CoT = ['no_chain_of_thought']
        
        configs = []
        for llm in llms:
            for task_type in task_types:
                for prompting in promptings:
                    for explanation_type in explanation_types:
                        for cot in CoT:
                            configs.append(ExperimentConfig(llm, task_type, cot, prompting, explanation_type))


        llms = ['prometheus']
        task_types = ['pairwise']
        for llm in llms:
            for task_type in task_types:
                for prompting in promptings:
                    for explanation_type in explanation_types:
                        for cot in CoT:
                            configs.append(ExperimentConfig(llm, task_type, cot, prompting, explanation_type))
        
        return configs
    
    def compare_formats_aggregated(self, llm: str, task_type: str, cot: str,
                                    baseline_format: str, test_format: str, explanation_type: str) -> Dict:
        """
        Compare two formats using McNemar's test with aggregated contingency tables
        across both prompting strategies (pos_neg and neg_pos).
        Uses only common test_ids between formats.
        
        Args:
            llm: LLM name
            task_type: Task type (single/pairwise)
            cot: Chain of thought setting
            baseline_format: Baseline format name
            test_format: Test format name
        
        Returns:
            Dictionary with aggregated accuracy metrics and test results
        """
        # Get both configurations
        config_pos_neg = ExperimentConfig(llm, task_type, cot, 'pos_neg', explanation_type)
        config_neg_pos = ExperimentConfig(llm, task_type, cot, 'neg_pos', explanation_type)
        
        # Get correctness arrays for both prompting strategies (filtered to common IDs)
        correct_baseline_pos_neg, correct_test_pos_neg, common_ids_pos_neg = \
            self.get_correctness_arrays_with_common_ids(config_pos_neg, baseline_format, test_format)
        
        correct_baseline_neg_pos, correct_test_neg_pos, common_ids_neg_pos = \
            self.get_correctness_arrays_with_common_ids(config_neg_pos, baseline_format, test_format)
        
        # Create contingency tables for both
        contingency_data_pos_neg = pd.DataFrame({
            'baseline': correct_baseline_pos_neg,
            'test': correct_test_pos_neg
        })
        contingency_table_pos_neg = SquareTable.from_data(contingency_data_pos_neg)
        table_pos_neg = contingency_table_pos_neg.table
        
        contingency_data_neg_pos = pd.DataFrame({
            'baseline': correct_baseline_neg_pos,
            'test': correct_test_neg_pos
        })
        contingency_table_neg_pos = SquareTable.from_data(contingency_data_neg_pos)
        table_neg_pos = contingency_table_neg_pos.table
        
        # Average the contingency table cells (a, b, c, d)
        # table format:
        #           test_False  test_True
        # base_False    a          c
        # base_True     b          d
        both_wrong = (table_pos_neg[0, 0] + table_neg_pos[0, 0]) / 2  # a
        got_worse = (table_pos_neg[1, 0] + table_neg_pos[1, 0]) / 2   # b (baseline correct, test wrong)
        got_better = (table_pos_neg[0, 1] + table_neg_pos[0, 1]) / 2  # c (baseline wrong, test correct)
        both_correct = (table_pos_neg[1, 1] + table_neg_pos[1, 1]) / 2 # d
        
        # Create averaged contingency table
        averaged_table = np.array([
            [both_wrong, got_better],
            [got_worse, both_correct]
        ])
        
        # Perform McNemar's test on averaged table
        mcnemar_result = mcnemar(averaged_table, exact=False, correction=True)
        
        # Calculate accuracies from averaged table
        n_total = averaged_table.sum()
        accuracy_baseline = (got_worse + both_correct) / n_total
        accuracy_test = (got_better + both_correct) / n_total
        
        # Calculate change metrics
        absolute_change = accuracy_test - accuracy_baseline
        relative_change = (accuracy_test / accuracy_baseline - 1.0) if accuracy_baseline > 0 else 0
        
        return {
            'baseline_format': baseline_format,
            'test_format': test_format,
            'llm': llm,
            'task_type': task_type,
            'cot': cot,
            'n_total': n_total,
            'n_common_pos_neg': len(common_ids_pos_neg),
            'n_common_neg_pos': len(common_ids_neg_pos),
            'accuracy_baseline': accuracy_baseline,
            'accuracy_test': accuracy_test,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'both_correct': both_correct,
            'got_worse': got_worse,
            'got_better': got_better,
            'both_wrong': both_wrong,
            'mcnemar_statistic': mcnemar_result.statistic,
            'p_value': mcnemar_result.pvalue,
            'significant': mcnemar_result.pvalue < 0.05
        }

    def analyze_configuration_aggregated(self, llm: str, task_type: str, cot: str, explanation_type: str) -> pd.DataFrame:
        """
        Analyze all explanation formats against baseline with aggregated prompting strategies
        
        Args:
            llm: LLM name
            task_type: Task type (single/pairwise)
            cot: Chain of thought setting
        
        Returns:
            DataFrame with aggregated results (9 formats x 11 columns)
        """
        results = []
        
        for format_name in self.EXPLANATION_FORMATS:
            if format_name == "baseline":
                # Baseline row: average baseline accuracy from both prompting strategies
                config_pos_neg = ExperimentConfig(llm, task_type, cot, 'pos_neg', explanation_type)
                config_neg_pos = ExperimentConfig(llm, task_type, cot, 'neg_pos', explanation_type)
                
                baseline_data_pos_neg = self.load_results(config_pos_neg, "baseline")
                baseline_data_neg_pos = self.load_results(config_neg_pos, "baseline")
                
                avg_baseline_accuracy = (baseline_data_pos_neg['accuracy'] + 
                                        baseline_data_neg_pos['accuracy']) / 2
                
                row = {
                    'baseline': avg_baseline_accuracy,
                    'text_scores': np.nan,
                    'text_labels': np.nan,
                    'structured_text_scores': np.nan,
                    'structured_text_labels': np.nan,
                    'top_words_scores': np.nan,
                    'top_words_labels': np.nan,
                    'natural_words': np.nan,
                    'part_of_speech': np.nan,
                    'change': np.nan,
                    'p': np.nan
                }
            else:
                # Compare format against baseline with aggregated tables
                comparison = self.compare_formats_aggregated(llm, task_type, cot, 
                                                            "baseline", format_name, explanation_type)
                
                row = {
                    'baseline': comparison['accuracy_baseline'],
                    'text_scores': np.nan,
                    'text_labels': np.nan,
                    'structured_text_scores': np.nan,
                    'structured_text_labels': np.nan,
                    'top_words_scores': np.nan,
                    'top_words_labels': np.nan,
                    'natural_words': np.nan,
                    'part_of_speech': np.nan,
                    'change': comparison['relative_change'],
                    'p': comparison['p_value']
                }
                # Fill in the test format accuracy
                row[format_name] = comparison['accuracy_test']
            
            results.append(row)
        
        # Create DataFrame with format names as index
        df = pd.DataFrame(results, index=self.EXPLANATION_FORMATS)
        
        return df

    def analyze_and_save_all_aggregated(self, output_dir: str = "latex_results"):
        """
        Analyze all configurations with aggregated prompting strategies and save LaTeX files
        
        Args:
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        llms = ['llama', 'qwen', 'prometheus']
        task_types = ['single', 'pairwise']
        cots = ['no_chain_of_thought', 'chain_of_thought_True']
        explanation_types = ['attention', 'lime', 'shap']
        
        successful_analyses = 0
        
        for llm in llms:
            for task_type in task_types:
                for cot in cots:
                    for exp in explanation_types:
                        try:
                            # Analyze configuration with aggregated prompting
                            results_df = self.analyze_configuration_aggregated(llm, task_type, cot, exp)
                            
                            # Create a pseudo-config for naming
                            avg_config = ExperimentConfig(llm, task_type, cot, 'aggregated', exp)
                            
                            # Save individual LaTeX file
                            individual_file = os.path.join(output_dir, f"results_{avg_config}.tex")
                            self.save_latex_table(results_df, avg_config, individual_file)
                            
                            successful_analyses += 1
                            
                        except FileNotFoundError as e:
                        # Only log for no_chain_of_thought; chain_of_thought configs are not used in this experiment
                            if cot == 'no_chain_of_thought':
                                if llm == 'prometheus':
                                    if task_type == 'pairwise':
                                        print(f"Skipping {llm}-{exp}-{task_type}-{cot}: {e}")
                                else:
                                    if task_type == 'single':
                                        print(f"Skipping {llm}-{exp}-{task_type}-{cot}: {e}")
                        except Exception as e:
                            if cot == 'no_chain_of_thought':
                                if llm == 'prometheus':
                                    if task_type == 'pairwise':
                                        print(f"Error processing {llm}-{exp}-{task_type}-{cot}: {e}")
                                else:
                                    if task_type == 'single':
                                        print(f"Error processing {llm}-{exp}-{task_type}-{cot}: {e}")

            
        # Save combined file
        combined_file = os.path.join(output_dir, "all_results_aggregated.tex")
        self.save_all_latex_tables(combined_file)
        
        print(f"\n{'='*80}")
        print(f"Aggregated analysis complete!")
        print(f"Successfully analyzed: {successful_analyses} configurations")
        print(f"Output directory: {output_dir}")