import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os

# Import your existing helper functions
import helper_functions

def load_pkl_files(directory, pattern, max_files=None):
    """Load all pkl files matching pattern from directory."""
    pkl_dir = Path(directory)
    files = sorted(pkl_dir.glob(pattern))
    
    if max_files:
        files = files[:max_files]
    
    data = []
    for file in tqdm(files, desc=f"Loading from {pkl_dir.name}"):
        with open(file, 'rb') as f:
            data.append(pickle.load(f))
    
    return data

def extract_scores_from_explanations(explanation_data, explanation_type, score_type='mean'):
    """
    Extract scores using the ExplanationFormatter class.
    
    Args:
        explanation_data: List of explanation objects (SHAP/LIME/Attention)
        explanation_type: 'shap', 'lime', or 'attention'
        score_type: For attention only - 'mean', 'max', or 'sum'
    
    Returns:
        numpy array of all scores
    """
    formatter = helper_functions.ExplanationFormatter()
    
    # Load explanations into formatter
    formatter.load_explanations(explanation_data, explanation_type)
    
    # Extract all scores from processed_data
    all_scores = []
    for words, scores in formatter.processed_data:
        all_scores.extend(scores)
    
    return np.array(all_scores)

def analyze_score_distribution(scores, name):
    """Compute statistics for score distribution."""
    stats = {
        'name': name,
        'count': len(scores),
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75),
        'q90': np.percentile(scores, 90),
        'q95': np.percentile(scores, 95),
        'q99': np.percentile(scores, 99),
    }
    
    # Additional stats for signed scores (SHAP/LIME)
    if np.any(scores < 0):
        positive_scores = scores[scores > 0]
        negative_scores = scores[scores < 0]
        abs_scores = np.abs(scores)
        
        stats.update({
            # Count stats
            'positive_count': len(positive_scores),
            'negative_count': len(negative_scores),
            
            # Mean stats
            'positive_mean': np.mean(positive_scores) if len(positive_scores) > 0 else 0,
            'negative_mean': np.mean(negative_scores) if len(negative_scores) > 0 else 0,
            
            # Absolute value stats
            'abs_mean': np.mean(abs_scores),
            'abs_std': np.std(abs_scores),
            'abs_min': np.min(abs_scores),
            'abs_max': np.max(abs_scores),
            'abs_median': np.median(abs_scores),
            
            # Absolute value percentiles
            'abs_q25': np.percentile(abs_scores, 25),
            'abs_q75': np.percentile(abs_scores, 75),
            'abs_q90': np.percentile(abs_scores, 90),
            'abs_q95': np.percentile(abs_scores, 95),
            'abs_q99': np.percentile(abs_scores, 99),
        })
    
    return stats

def print_statistics(stats):
    """Print statistics in a readable format."""
    print(f"\n{'='*80}")
    print(f"STATISTICS: {stats['name']}")
    print(f"{'='*80}")
    print(f"Total scores: {stats['count']:,}")
    
    # ========================================================================
    # RAW SCORES (signed for SHAP/LIME, positive for Attention)
    # ========================================================================
    print(f"\n{'─'*80}")
    print(f"RAW SCORE STATISTICS:")
    print(f"{'─'*80}")
    print(f"  Mean:   {stats['mean']:>10.6f}")
    print(f"  Median: {stats['median']:>10.6f}")
    print(f"  Std:    {stats['std']:>10.6f}")
    print(f"  Min:    {stats['min']:>10.6f}")
    print(f"  Max:    {stats['max']:>10.6f}")
    
    print(f"\n  Percentiles:")
    print(f"    25th: {stats['q25']:>10.6f}")
    print(f"    50th: {stats['median']:>10.6f}  (median)")
    print(f"    75th: {stats['q75']:>10.6f}")
    print(f"    90th: {stats['q90']:>10.6f}")
    print(f"    95th: {stats['q95']:>10.6f}")
    print(f"    99th: {stats['q99']:>10.6f}")
    
    # ========================================================================
    # SIGNED SCORE BREAKDOWN (SHAP/LIME only)
    # ========================================================================
    if 'positive_count' in stats:
        print(f"\n{'─'*80}")
        print(f"SIGNED SCORE BREAKDOWN:")
        print(f"{'─'*80}")
        print(f"  Positive scores: {stats['positive_count']:>10,} ({stats['positive_count']/stats['count']*100:>5.1f}%)")
        print(f"    └─ Mean:       {stats['positive_mean']:>10.6f}")
        print()
        print(f"  Negative scores: {stats['negative_count']:>10,} ({stats['negative_count']/stats['count']*100:>5.1f}%)")
        print(f"    └─ Mean:       {stats['negative_mean']:>10.6f}")
        
        # ====================================================================
        # ABSOLUTE VALUE STATISTICS (PRIMARY for thresholding!)
        # ====================================================================
        print(f"\n{'─'*80}")
        print(f"ABSOLUTE VALUE STATISTICS (for thresholding):")
        print(f"{'─'*80}")
        print(f"  Mean:   {stats['abs_mean']:>10.6f}")
        print(f"  Median: {stats['abs_median']:>10.6f}")
        print(f"  Std:    {stats['abs_std']:>10.6f}")
        print(f"  Min:    {stats['abs_min']:>10.6f}")
        print(f"  Max:    {stats['abs_max']:>10.6f}")
        
        print(f"\n  Percentiles (for threshold selection):")
        print(f"    25th: {stats['abs_q25']:>10.6f}  → Highlights top 75% of words")
        print(f"    50th: {stats['abs_median']:>10.6f}  → Highlights top 50% of words (median)")
        print(f"    75th: {stats['abs_q75']:>10.6f}  → Highlights top 25% of words")
        print(f"    90th: {stats['abs_q90']:>10.6f}  → Highlights top 10% of words")
        print(f"    95th: {stats['abs_q95']:>10.6f}  → Highlights top 5% of words")
        print(f"    99th: {stats['abs_q99']:>10.6f}  → Highlights top 1% of words")


def print_comparison_table(shap_stats, lime_stats, attention_stats):
    """Print a comparison table of absolute value statistics."""
    print("\n" + "="*80)
    print("COMPARISON: ABSOLUTE VALUE STATISTICS")
    print("="*80)
    print()
    print(f"{'Metric':<20} {'SHAP':>15} {'LIME':>15} {'Attention':>15}")
    print("─" * 80)
    
    # For SHAP and LIME, use absolute stats; for Attention, use regular stats
    shap_abs = 'abs_' if 'abs_mean' in shap_stats else ''
    lime_abs = 'abs_' if 'abs_mean' in lime_stats else ''
    attn_abs = ''  # Attention doesn't need abs_ prefix
    
    print(f"{'Mean':<20} {shap_stats.get(f'{shap_abs}mean', shap_stats['mean']):>15.6f} "
          f"{lime_stats.get(f'{lime_abs}mean', lime_stats['mean']):>15.6f} "
          f"{attention_stats['mean']:>15.6f}")
    
    print(f"{'Median':<20} {shap_stats.get(f'{shap_abs}median', shap_stats['median']):>15.6f} "
          f"{lime_stats.get(f'{lime_abs}median', lime_stats['median']):>15.6f} "
          f"{attention_stats['median']:>15.6f}")
    
    print(f"{'Std Dev':<20} {shap_stats.get(f'{shap_abs}std', shap_stats['std']):>15.6f} "
          f"{lime_stats.get(f'{lime_abs}std', lime_stats['std']):>15.6f} "
          f"{attention_stats['std']:>15.6f}")
    
    print(f"{'Min':<20} {shap_stats.get(f'{shap_abs}min', shap_stats['min']):>15.6f} "
          f"{lime_stats.get(f'{lime_abs}min', lime_stats['min']):>15.6f} "
          f"{attention_stats['min']:>15.6f}")
    
    print(f"{'Max':<20} {shap_stats.get(f'{shap_abs}max', shap_stats['max']):>15.6f} "
          f"{lime_stats.get(f'{lime_abs}max', lime_stats['max']):>15.6f} "
          f"{attention_stats['max']:>15.6f}")
    
    print()
    print("Percentiles:")
    print("─" * 80)
    
    for percentile, key in [('25th', 'q25'), ('50th', 'median'), ('75th', 'q75'), 
                           ('90th', 'q90'), ('95th', 'q95'), ('99th', 'q99')]:
        abs_key = f'{shap_abs}{key}' if shap_abs else key
        print(f"  {percentile:<18} {shap_stats.get(abs_key, shap_stats[key]):>15.6f} "
              f"{lime_stats.get(f'{lime_abs}{key}', lime_stats[key]):>15.6f} "
              f"{attention_stats[key]:>15.6f}")
    
    print()

def suggest_thresholds(stats):
    """Suggest threshold values based on statistics."""
    print(f"\n{'='*80}")
    print(f"SUGGESTED THRESHOLDS FOR: {stats['name']}")
    print(f"{'='*80}")
    
    if 'abs_median' in stats:
        # For SHAP/LIME (signed scores)
        print("\nOption 1: Conservative (keep ~75% of scores)")
        print(f"  Threshold: {stats['abs_q25']:.6f}")
        
        print("\nOption 2: Moderate (keep ~50% of scores)")
        print(f"  Threshold: {stats['abs_median']:.6f}")
        
        print("\nOption 3: Strict (keep ~10% of scores)")
        print(f"  Threshold: {stats['abs_q90']:.6f}")
        
        print("\nOption 4: Very Strict (keep ~5% of scores)")
        print(f"  Threshold: {stats['abs_q95']:.6f}")
    else:
        # For Attention (positive scores only)
        print("\nOption 1: Conservative (keep ~75% of scores)")
        print(f"  Threshold: {stats['q25']:.6f}")
        
        print("\nOption 2: Moderate (keep ~50% of scores)")
        print(f"  Threshold: {stats['median']:.6f}")
        
        print("\nOption 3: Strict (keep ~10% of scores)")
        print(f"  Threshold: {stats['q90']:.6f}")
        
        print("\nOption 4: Very Strict (keep ~5% of scores)")
        print(f"  Threshold: {stats['q95']:.6f}")

def plot_distributions(shap_scores, lime_scores, attention_scores, output_file='threshold_analysis/score_distributions.png'):
    """Create comprehensive distribution plots."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # ========================================================================
    # SHAP PLOTS
    # ========================================================================
    
    # SHAP: Regular histogram
    axes[0, 0].hist(shap_scores, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 0].set_xlabel('Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('SHAP Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # SHAP: Box plot
    axes[0, 1].boxplot([shap_scores], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('SHAP Box Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # SHAP: Absolute values with LOG SCALE (NEW!)
    axes[0, 2].hist(np.abs(shap_scores), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_yscale('log')
    axes[0, 2].axvline(np.median(np.abs(shap_scores)), color='red', linestyle='--', 
                      linewidth=2, label=f'Median: {np.median(np.abs(shap_scores)):.4f}')
    axes[0, 2].set_xlabel('Absolute Score', fontsize=12)
    axes[0, 2].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[0, 2].set_title('SHAP Absolute Value Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # ========================================================================
    # LIME PLOTS
    # ========================================================================
    
    # LIME: Regular histogram
    axes[1, 0].hist(lime_scores, bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[1, 0].set_xlabel('Score', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('LIME Score Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # LIME: Box plot
    axes[1, 1].boxplot([lime_scores], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen'))
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('LIME Box Plot', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # LIME: Absolute values with LOG SCALE (NEW!)
    axes[1, 2].hist(np.abs(lime_scores), bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[1, 2].set_yscale('log')
    axes[1, 2].axvline(np.median(np.abs(lime_scores)), color='red', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(np.abs(lime_scores)):.4f}')
    axes[1, 2].set_xlabel('Absolute Score', fontsize=12)
    axes[1, 2].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[1, 2].set_title('LIME Absolute Value Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    # ========================================================================
    # ATTENTION PLOTS
    # ========================================================================
    
    # Attention: Regular histogram
    axes[2, 0].hist(attention_scores, bins=100, alpha=0.7, color='orange', edgecolor='black')
    axes[2, 0].axvline(np.median(attention_scores), color='red', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(attention_scores):.4f}')
    axes[2, 0].set_xlabel('Score', fontsize=12)
    axes[2, 0].set_ylabel('Frequency', fontsize=12)
    axes[2, 0].set_title('Attention Score Distribution', fontsize=14, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)
    
    # Attention: Box plot
    axes[2, 1].boxplot([attention_scores], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightyellow'))
    axes[2, 1].set_ylabel('Score', fontsize=12)
    axes[2, 1].set_title('Attention Box Plot', fontsize=14, fontweight='bold')
    axes[2, 1].grid(alpha=0.3)
    
    # Attention: Log scale histogram (ALREADY HAD THIS)
    axes[2, 2].hist(attention_scores, bins=100, alpha=0.7, color='orange', edgecolor='black')
    axes[2, 2].set_yscale('log')
    axes[2, 2].axvline(np.median(attention_scores), color='red', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(attention_scores):.4f}')
    axes[2, 2].set_xlabel('Score', fontsize=12)
    axes[2, 2].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[2, 2].set_title('Attention Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[2, 2].legend()
    axes[2, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plots to: {output_file}")
    plt.close()


def plot_comparison(shap_scores, lime_scores, attention_scores, output_file='threshold_analysis/score_comparison.png'):
    """Create side-by-side comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    data_to_plot = [
        np.abs(shap_scores),
        np.abs(lime_scores),
        attention_scores
    ]
    labels = ['SHAP\n(absolute)', 'LIME\n(absolute)', 'Attention']
    colors = ['blue', 'green', 'orange']
    
    # Box plots comparison
    bp = axes[0].boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Violin plots
    parts = axes[1].violinplot(data_to_plot, positions=[1, 2, 3], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Score Density Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    # Overlaid histograms (normalized)
    axes[2].hist(np.abs(shap_scores), bins=50, alpha=0.5, color='blue', 
                label='SHAP', density=True, edgecolor='black')
    axes[2].hist(np.abs(lime_scores), bins=50, alpha=0.5, color='green',
                label='LIME', density=True, edgecolor='black')
    axes[2].hist(attention_scores, bins=50, alpha=0.5, color='orange',
                label='Attention', density=True, edgecolor='black')
    axes[2].set_xlabel('Score', fontsize=12)
    axes[2].set_ylabel('Density', fontsize=12)
    axes[2].set_title('Normalized Distribution Overlay', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plots to: {output_file}")
    plt.close()

def calculate_exact_percentile(scores, threshold, use_abs=True):
    """
    Calculate the exact percentile for a given threshold.
    
    Args:
        scores: numpy array of scores
        threshold: the threshold value
        use_abs: whether to use absolute values (for SHAP/LIME)
    
    Returns:
        percentile: the exact percentile (0-100)
    """
    if use_abs:
        scores = np.abs(scores)
    
    # Calculate what percentage of scores are below the threshold
    percentage_below = (scores < threshold).sum() / len(scores) * 100
    
    return percentage_below

def find_threshold_for_percentile(scores, percentile, use_abs=True):
    """
    Find the threshold value that corresponds to a given percentile.
    
    Args:
        scores: numpy array of scores
        percentile: target percentile (0-100)
        use_abs: whether to use absolute values (for SHAP/LIME)
    
    Returns:
        threshold: the threshold value at that percentile
    """
    if use_abs:
        scores = np.abs(scores)
    
    return np.percentile(scores, percentile)

def match_thresholds_across_methods(shap_scores, lime_scores, attention_scores, 
                                    shap_threshold=0.01):
    """
    Calculate matching thresholds for LIME and Attention based on SHAP threshold.
    
    Args:
        shap_scores, lime_scores, attention_scores: numpy arrays of scores
        shap_threshold: your chosen SHAP threshold
    
    Returns:
        dict with thresholds and statistics
    """
    # Calculate exact percentile for SHAP threshold
    shap_percentile = calculate_exact_percentile(shap_scores, shap_threshold, use_abs=True)
    
    # Find matching thresholds for LIME and Attention
    lime_threshold = find_threshold_for_percentile(lime_scores, shap_percentile, use_abs=True)
    attention_threshold = find_threshold_for_percentile(attention_scores, shap_percentile, use_abs=False)
    
    # Calculate percentage of words that will be highlighted
    percentage_highlighted = 100 - shap_percentile
    
    # Verify the thresholds
    shap_verify = 100 - calculate_exact_percentile(shap_scores, shap_threshold, use_abs=True)
    lime_verify = 100 - calculate_exact_percentile(lime_scores, lime_threshold, use_abs=True)
    attention_verify = 100 - calculate_exact_percentile(attention_scores, attention_threshold, use_abs=False)
    
    results = {
        'shap': {
            'threshold': shap_threshold,
            'percentile': shap_percentile,
            'percentage_highlighted': shap_verify
        },
        'lime': {
            'threshold': lime_threshold,
            'percentile': shap_percentile,  # Same target percentile
            'percentage_highlighted': lime_verify
        },
        'attention': {
            'threshold': attention_threshold,
            'percentile': shap_percentile,  # Same target percentile
            'percentage_highlighted': attention_verify
        },
        'target_percentage_highlighted': percentage_highlighted
    }
    
    return results

def main():

    os.makedirs('threshold_analysis', exist_ok=True)


    """Main analysis function."""
    
    # Configuration
    data_set = 'dev'  # or 'train'
    max_files = None  # Set to a number to limit files, or None for all
    
    print("="*80)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load SHAP data
    print("\n[1/3] Loading SHAP data...")
    shap_data = load_pkl_files(
        f'explanations/pkl/shap/{data_set}_set/shap_raw',
        'shap_values_*.pkl',
        max_files=max_files
    )
    shap_scores = extract_scores_from_explanations(shap_data, 'shap')
    shap_stats = analyze_score_distribution(shap_scores, 'SHAP')
    print_statistics(shap_stats)
    suggest_thresholds(shap_stats)
    
    # Load LIME data
    print("\n[2/3] Loading LIME data...")
    lime_data = load_pkl_files(
        f'explanations/pkl/lime/{data_set}_set/lime_raw',
        'lime_explanation_*.pkl',
        max_files=max_files
    )
    lime_scores = extract_scores_from_explanations(lime_data, 'lime')
    lime_stats = analyze_score_distribution(lime_scores, 'LIME')
    print_statistics(lime_stats)
    suggest_thresholds(lime_stats)
    
    # Load Attention data
    print("\n[3/3] Loading Attention data...")
    attention_data = load_pkl_files(
        f'explanations/pkl/attention/{data_set}_set/attention_raw',
        'attention_explanation_*.pkl',
        max_files=max_files
    )

    # Load and analyze all three methods
    shap_stats = analyze_score_distribution(shap_scores, 'SHAP')
    print_statistics(shap_stats)
    suggest_thresholds(shap_stats)
    
    lime_stats = analyze_score_distribution(lime_scores, 'LIME')
    print_statistics(lime_stats)
    suggest_thresholds(lime_stats)

    attention_scores = extract_scores_from_explanations(attention_data, 'attention', score_type='mean')    
    attention_stats = analyze_score_distribution(attention_scores, 'Attention (mean)')
    print_statistics(attention_stats)
    suggest_thresholds(attention_stats)
    
    # ============================================================================
    # COMPARISON TABLE
    # ============================================================================
    print_comparison_table(shap_stats, lime_stats, attention_stats)

    
    # ============================================================================
    # CALCULATE MATCHED THRESHOLDS
    # ============================================================================
    print("\n" + "="*80)
    print("CALCULATING MATCHED THRESHOLDS")
    print("="*80)
    
    # Your desired SHAP threshold
    desired_shap_threshold = 0.01
    
    # Calculate matching thresholds
    matched_thresholds = match_thresholds_across_methods(
        shap_scores, 
        lime_scores, 
        attention_scores,
        shap_threshold=desired_shap_threshold
    )
    
    # Print results
    print(f"\nDesired SHAP threshold: {desired_shap_threshold}")
    print(f"This corresponds to the {matched_thresholds['shap']['percentile']:.2f}th percentile")
    print(f"Which highlights {matched_thresholds['target_percentage_highlighted']:.2f}% of words\n")
    
    print("MATCHED THRESHOLDS:")
    print("-" * 80)
    print(f"SHAP:      {matched_thresholds['shap']['threshold']:.6f}")
    print(f"           → Highlights {matched_thresholds['shap']['percentage_highlighted']:.2f}% of words")
    print()
    print(f"LIME:      {matched_thresholds['lime']['threshold']:.6f}")
    print(f"           → Highlights {matched_thresholds['lime']['percentage_highlighted']:.2f}% of words")
    print()
    print(f"Attention: {matched_thresholds['attention']['threshold']:.6f}")
    print(f"           → Highlights {matched_thresholds['attention']['percentage_highlighted']:.2f}% of words")
    print()
    
    print("\nCOPY-PASTE READY CODE:")
    print("-" * 80)
    print(f"threshold_shap={matched_thresholds['shap']['threshold']:.6f},")
    print(f"threshold_lime={matched_thresholds['lime']['threshold']:.6f},")
    print(f"threshold_attention={matched_thresholds['attention']['threshold']:.6f},")
    
    # Save matched thresholds to JSON
    with open('threshold_analysis/matched_thresholds.json', 'w') as f:
        json.dump(matched_thresholds, f, indent=2)
    print("\nSaved matched thresholds to: matched_thresholds.json")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_distributions(shap_scores, lime_scores, attention_scores)
    plot_comparison(shap_scores, lime_scores, attention_scores)
    
    # Save statistics to JSON
    stats_summary = {
        'shap': shap_stats,
        'lime': lime_stats,
        'attention': attention_stats
    }
    
    with open('threshold_analysis/score_statistics.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print("\nSaved statistics to: score_statistics.json")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print("  - score_distributions.png (detailed distributions)")
    print("  - score_comparison.png (side-by-side comparison)")
    print("  - score_statistics.json (numerical statistics)")
    print("  - matched_thresholds.json (matched thresholds)")

if __name__ == '__main__':
    main() 