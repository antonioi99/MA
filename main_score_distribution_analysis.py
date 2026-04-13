import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import os

import helper_functions


def load_pkl_files(directory, pattern, max_files=None):
    """Load all pkl files matching pattern from directory."""
    pkl_dir = Path(directory)
    files = sorted(pkl_dir.glob(pattern))
    
    if max_files:
        files = files[:max_files]
    
    data = []
    total = len(files)
    update_interval = max(1, total // 10)

    with tqdm(total=total, desc=f"Loading from {pkl_dir.name}",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for i, file in enumerate(files):
            with open(file, 'rb') as f:
                data.append(pickle.load(f))
            
            if (i + 1) % update_interval == 0 or (i + 1) == total:
                pbar.update(update_interval if (i + 1) % update_interval == 0 else total % update_interval)
    
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
    formatter.load_explanations(explanation_data, explanation_type)
    
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
    
    if np.any(scores < 0):
        positive_scores = scores[scores > 0]
        negative_scores = scores[scores < 0]
        abs_scores = np.abs(scores)
        
        stats.update({
            'positive_count': len(positive_scores),
            'negative_count': len(negative_scores),
            'positive_mean': np.mean(positive_scores) if len(positive_scores) > 0 else 0,
            'negative_mean': np.mean(negative_scores) if len(negative_scores) > 0 else 0,
            'abs_mean': np.mean(abs_scores),
            'abs_std': np.std(abs_scores),
            'abs_min': np.min(abs_scores),
            'abs_max': np.max(abs_scores),
            'abs_median': np.median(abs_scores),
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
    
    if 'positive_count' in stats:
        print(f"\n{'─'*80}")
        print(f"SIGNED SCORE BREAKDOWN:")
        print(f"{'─'*80}")
        print(f"  Positive scores: {stats['positive_count']:>10,} ({stats['positive_count']/stats['count']*100:>5.1f}%)")
        print(f"    └─ Mean:       {stats['positive_mean']:>10.6f}")
        print()
        print(f"  Negative scores: {stats['negative_count']:>10,} ({stats['negative_count']/stats['count']*100:>5.1f}%)")
        print(f"    └─ Mean:       {stats['negative_mean']:>10.6f}")
        
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
    
    shap_abs = 'abs_' if 'abs_mean' in shap_stats else ''
    lime_abs = 'abs_' if 'abs_mean' in lime_stats else ''
    
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


def suggest_thresholds(stats):
    """Suggest threshold values based on statistics."""
    print(f"\n{'='*80}")
    print(f"SUGGESTED THRESHOLDS FOR: {stats['name']}")
    print(f"{'='*80}")
    
    if 'abs_median' in stats:
        print("\nOption 1: Conservative (keep ~75% of scores)")
        print(f"  Threshold: {stats['abs_q25']:.6f}")
        
        print("\nOption 2: Moderate (keep ~50% of scores)")
        print(f"  Threshold: {stats['abs_median']:.6f}")
        
        print("\nOption 3: Strict (keep ~10% of scores)")
        print(f"  Threshold: {stats['abs_q90']:.6f}")
        
        print("\nOption 4: Very Strict (keep ~5% of scores)")
        print(f"  Threshold: {stats['abs_q95']:.6f}")
    else:
        print("\nOption 1: Conservative (keep ~75% of scores)")
        print(f"  Threshold: {stats['q25']:.6f}")
        
        print("\nOption 2: Moderate (keep ~50% of scores)")
        print(f"  Threshold: {stats['median']:.6f}")
        
        print("\nOption 3: Strict (keep ~10% of scores)")
        print(f"  Threshold: {stats['q90']:.6f}")
        
        print("\nOption 4: Very Strict (keep ~5% of scores)")
        print(f"  Threshold: {stats['q95']:.6f}")


def calculate_exact_percentile(scores, threshold, use_abs=True):
    """Calculate the exact percentile for a given threshold."""
    if use_abs:
        scores = np.abs(scores)
    percentage_below = (scores < threshold).sum() / len(scores) * 100
    return percentage_below


def find_threshold_for_percentile(scores, percentile, use_abs=True):
    """Find the threshold value that corresponds to a given percentile."""
    if use_abs:
        scores = np.abs(scores)
    return np.percentile(scores, percentile)


def match_thresholds_across_methods(shap_scores, lime_scores, attention_scores,
                                    shap_threshold=0.01):
    
    # SHAP: compute percentile of words cut off
    shap_percentile = calculate_exact_percentile(shap_scores, shap_threshold, use_abs=True)
    percentage_highlighted = 100 - shap_percentile

    # LIME: check how many non-zero scores we already have
    lime_nonzero_percentage = (np.abs(lime_scores) > 0).sum() / len(lime_scores) * 100
    
    print(f"\nLIME non-zero scores: {lime_nonzero_percentage:.2f}% of all scores")
    print(f"SHAP target retention: {percentage_highlighted:.2f}% of all scores")

    if lime_nonzero_percentage <= percentage_highlighted:
        # LIME already has fewer non-zero words than SHAP retains
        # Just use a very small threshold to keep all non-zero scores
        lime_threshold = 1e-10
        print(f"LIME non-zero words ({lime_nonzero_percentage:.2f}%) already below "
              f"SHAP target ({percentage_highlighted:.2f}%) → keeping all non-zero scores")
    else:
        # LIME has more non-zero words than SHAP retains
        # Need to cut some non-zero LIME scores
        lime_nonzero = lime_scores[np.abs(lime_scores) > 0]
        # Find threshold that retains percentage_highlighted% of ALL lime scores
        target_nonzero_percentile = 100 - (percentage_highlighted / lime_nonzero_percentage * 100)
        lime_threshold = find_threshold_for_percentile(lime_nonzero, target_nonzero_percentile, use_abs=True)
        print(f"LIME has more non-zero words than target → applying threshold {lime_threshold:.6f}")

    # ATTENTION: unchanged
    attention_threshold = find_threshold_for_percentile(attention_scores, shap_percentile, use_abs=False)

    # Verify
    shap_verify = 100 - calculate_exact_percentile(shap_scores, shap_threshold, use_abs=True)
    lime_verify = 100 - calculate_exact_percentile(lime_scores, lime_threshold, use_abs=True)
    attention_verify = 100 - calculate_exact_percentile(attention_scores, attention_threshold, use_abs=False)

    print(f"\nFinal retention rates:")
    print(f"  SHAP:      {shap_verify:.2f}% of all words")
    print(f"  LIME:      {lime_verify:.2f}% of all words")
    print(f"  Attention: {attention_verify:.2f}% of all words")

    results = {
        'shap': {
            'threshold': shap_threshold,
            'percentile': shap_percentile,
            'percentage_highlighted': shap_verify
        },
        'lime': {
            'threshold': lime_threshold,
            'percentile': shap_percentile,
            'percentage_highlighted': lime_verify
        },
        'attention': {
            'threshold': attention_threshold,
            'percentile': shap_percentile,
            'percentage_highlighted': attention_verify
        },
        'target_percentage_highlighted': percentage_highlighted
    }

    return results


def plot_distributions(shap_scores, lime_scores, attention_scores,
                               shap_threshold,
                               lime_threshold,
                               attention_threshold,
                               output_file='threshold_analysis/score_distributions_thesis.png'):
    """
    Clean 2x3 figure for thesis:
    - Top row: full signed distribution (shared y-axis)
    - Bottom row: zoomed absolute value distribution with threshold (log scale)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {'shap': 'steelblue', 'lime': 'seagreen', 'attention': 'darkorange'}
    methods = [
        ('SHAP',      shap_scores,      shap_threshold,      colors['shap'],      True,  (-0.04, 0.04), (0, 0.7)),
        ('LIME',      lime_scores,      lime_threshold,      colors['lime'],      True,  (-0.04, 0.04), (0, 0.7)),
        ('Attention', attention_scores, attention_threshold, colors['attention'], False, (0,     0.02),  (0, 0.2)),
    ]

    # --- Top row: linear distributions ---
    top_axes = []
    for col, (name, scores, threshold, color, use_abs, xlim_linear, _) in enumerate(methods):
        ax = axes[0, col]
        ax.hist(scores, bins=150, alpha=0.75, color=color, edgecolor='none')
        if use_abs:
            ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Zero')
            ax.legend(fontsize=10)
        ax.set_xlim(xlim_linear)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} — Score Distribution', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        top_axes.append(ax)

    # Shared y-axis scale for top row
    top_y_max = max(ax.get_ylim()[1] for ax in top_axes)
    for ax in top_axes:
        ax.set_ylim(0, top_y_max)

    # --- Bottom row: log-scale absolute distributions ---
    for col, (name, scores, threshold, color, use_abs, _, xlim_log) in enumerate(methods):
        ax = axes[1, col]
        abs_scores = np.abs(scores) if use_abs else scores
        ax.hist(abs_scores, bins=150, alpha=0.75, color=color, edgecolor='none')
        ax.set_yscale('log')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold}')
        ax.set_xlim(xlim_log)
        ax.set_xlabel('Absolute Score' if use_abs else 'Score', fontsize=11)
        ax.set_ylabel('Frequency (log scale)', fontsize=11)
        ax.set_title(f'{name} — Absolute Value Distribution (Log Scale)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved thesis figure to: {output_file}")
    plt.close()


def main():

    os.makedirs('threshold_analysis', exist_ok=True)

    data_set = 'dev'
    max_files = None
    
    print("="*80)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load SHAP data
    print("\n[1/3] Loading SHAP data...")
    shap_data = load_pkl_files(
        f'explanations/pkl/shap/{data_set}_set/shap_raw',
        'shap_*.pkl',
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
        'lime_*.pkl',
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
        'attention_*.pkl',
        max_files=max_files
    )
    attention_scores = extract_scores_from_explanations(attention_data, 'attention', score_type='mean')
    attention_stats = analyze_score_distribution(attention_scores, 'Attention (mean)')
    print_statistics(attention_stats)
    suggest_thresholds(attention_stats)
    
    # Comparison table
    print_comparison_table(shap_stats, lime_stats, attention_stats)
    
    # Calculate matched thresholds
    print("\n" + "="*80)
    print("CALCULATING MATCHED THRESHOLDS")
    print("="*80)
    
    desired_shap_threshold = 0.01
    matched_thresholds = match_thresholds_across_methods(
        shap_scores,
        lime_scores,
        attention_scores,
        shap_threshold=desired_shap_threshold
    )
    
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
    
    print("\nCOPY-PASTE READY CODE:")
    print("-" * 80)
    print(f"threshold_shap={matched_thresholds['shap']['threshold']:.6f},")
    print(f"threshold_lime={matched_thresholds['lime']['threshold']:.6f},")
    print(f"threshold_attention={matched_thresholds['attention']['threshold']:.6f},")
    
    # Save matched thresholds
    with open('threshold_analysis/matched_thresholds.json', 'w') as f:
        json.dump(matched_thresholds, f, indent=2)
    print("\nSaved matched thresholds to: matched_thresholds.json")
    
    # Create thesis figure
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_distributions(
        shap_scores,
        lime_scores,
        attention_scores,
        shap_threshold=round(matched_thresholds['shap']['threshold'], 3),
        lime_threshold=round(matched_thresholds['lime']['threshold'], 3),
        attention_threshold=round(matched_thresholds['attention']['threshold'], 3)
    )
    
    # Save statistics
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
    print("  - score_distributions_thesis.png")
    print("  - score_statistics.json")
    print("  - matched_thresholds.json")


if __name__ == '__main__':
    main()