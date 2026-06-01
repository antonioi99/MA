import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict
from helper_analysis import McNemarAnalyzer, AggregatedAnalyzer, ExperimentConfig
import os
from sklearn.metrics import cohen_kappa_score
import json
from scipy import stats


def collect_aggregated_results(analyzer: McNemarAnalyzer) -> pd.DataFrame:
    """
    Collect all aggregated results into a single DataFrame for visualization.
    """
    model_configs = [
        ('llama', 'single'),
        ('qwen', 'single'),
        ('prometheus', 'pairwise')
    ]
    explanation_types = ['shap', 'lime', 'attention']
    formats = [f for f in McNemarAnalyzer.EXPLANATION_FORMATS if f != 'baseline']
    cot = 'no_chain_of_thought'

    model_display = {
        'llama': 'Llama',
        'qwen': 'Qwen',
        'prometheus': 'M-Prometheus'
    }
    explanation_display = {
        'shap': 'SHAP',
        'lime': 'LIME',
        'attention': 'Attention'
    }
    format_display = {
        f: f.replace('_', ' ').title() for f in formats
    }

    rows = []

    for llm, task_type in model_configs:
        for exp in explanation_types:
            for fmt in formats:
                try:
                    comparison = analyzer.compare_formats_aggregated(
                        llm, task_type, cot, 'baseline', fmt, exp
                    )
                    rows.append({
                        'model': model_display[llm],
                        'explanation': explanation_display[exp],
                        'format': format_display[fmt],
                        'absolute_change': float(comparison['absolute_change'] * 100),
                        'p_value': float(comparison['p_value']),
                        'significant': bool(comparison['significant']),
                        'baseline': float(comparison['accuracy_baseline'] * 100),  
                        'accuracy': float(comparison['accuracy_test'] * 100)   
                    })
                except Exception as e:
                    print(f"Skipping {llm}-{exp}-{fmt}: {e}")
                    continue

    df = pd.DataFrame(rows)
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    return df


def plot_heatmap(df: pd.DataFrame, output_file: str = 'figures/heatmap_results.png'):
    """
    Plot a heatmap of relative changes across all model-explanation combinations.
    Rows: verbalization formats
    Columns: model-explanation combinations
    Color: relative change (green=positive, red=negative) in 0.5% intervals
    Asterisk: statistically significant results
    """
    import matplotlib.colors as mcolors

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create column labels
    df['config'] = df['model'] + '\n' + df['explanation']

    # Pivot for heatmap
    pivot_change = df.pivot_table(
        index='format', columns='config', values='absolute_change'
    )
    pivot_sig = df.pivot_table(
        index='format', columns='config', values='significant'
    )

    # Define column order
    col_order = [
        'Llama\nSHAP', 'Llama\nLIME', 'Llama\nAttention',
        'Qwen\nSHAP', 'Qwen\nLIME', 'Qwen\nAttention',
        'M-Prometheus\nSHAP', 'M-Prometheus\nLIME', 'M-Prometheus\nAttention'
    ]
    col_order = [c for c in col_order if c in pivot_change.columns]

    # Define row order
    row_order = [
        'Text Scores', 'Text Labels',
        'Structured Text Scores', 'Structured Text Labels',
        'Top Words Scores', 'Top Words Labels',
        'Natural Words', 'Part Of Speech'
    ]
    row_order = [r for r in row_order if r in pivot_change.index]

    pivot_change = pivot_change.loc[row_order, col_order]
    pivot_sig = pivot_sig.loc[row_order, col_order]

    # Determine symmetric boundaries in 0.5 intervals
    abs_max = max(abs(pivot_change.values.min()), abs(pivot_change.values.max()))
    # Round up to nearest 0.5
    abs_max = np.ceil(abs_max / 0.5) * 0.5
    boundaries = np.arange(-abs_max, abs_max + 0.5, 0.5)

    # Create discrete colormap
    n_colors = len(boundaries) - 1
    base_cmap = plt.cm.RdYlGn
    colors = [base_cmap(i / (n_colors - 1)) for i in range(n_colors)]
    discrete_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, discrete_cmap.N)

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.heatmap(
        pivot_change,
        ax=ax,
        cmap=discrete_cmap,
        norm=norm,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={
            'label': 'Change (%)',
            'ticks': boundaries,
            'spacing': 'uniform'
        }
    )

    # Add asterisks for significant results
    for i, row_name in enumerate(pivot_change.index):
        for j, col_name in enumerate(pivot_change.columns):
            if pivot_sig.loc[row_name, col_name]:
                ax.text(
                    j + 0.5, i + 0.15, '*',
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black'
                )

    ax.set_xlabel('Model --- Explanation Method', fontsize=11)
    ax.set_ylabel('Verbalization Format', fontsize=11)
    ax.tick_params(axis='x', labelsize=9, rotation=45)
    ax.tick_params(axis='y', labelsize=9)

    # Add vertical lines to separate models
    for x in [3, 6]:
        ax.axvline(x=x, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")
    plt.close()


def plot_facet_bar(df: pd.DataFrame, output_file: str = 'figures/facet_bar_results.png'):
    """
    Plot a 3x3 facet grid of horizontal bar charts.
    Rows: judge models
    Columns: explanation methods
    Each panel: verbalization formats on y-axis, relative change on x-axis
    Bar color: green=positive, red=negative
    Hatching: not significant
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    models = ['Llama', 'Qwen', 'M-Prometheus']
    explanations = ['Attention', 'LIME', 'SHAP']

    row_order = [
        'Text Scores', 'Text Labels',
        'Structured Text Scores', 'Structured Text Labels',
        'Top Words Scores', 'Top Words Labels',
        'Natural Words', 'Part Of Speech'
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(13, 9),
        sharex=False, sharey=True
    )

    x_min = df['absolute_change'].min() - 0.5
    x_max = df['absolute_change'].max() + 0.5

    for row_idx, model in enumerate(models):
        for col_idx, exp in enumerate(explanations):
            ax = axes[row_idx, col_idx]

            subset = df[(df['model'] == model) & (df['explanation'] == exp)].copy()
            subset = subset.set_index('format').reindex(row_order).reset_index()

            for i, r in subset.iterrows():
                color = '#2ecc71' if r['absolute_change'] > 0 else '#e74c3c'
                hatch = '' if r['significant'] else '///'

                ax.barh(
                    r['format'],
                    r['absolute_change'],
                    color=color,
                    edgecolor='black',
                    linewidth=0.6,
                    hatch=hatch,
                    height=0.6,
                    alpha=0.85
                )

            ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
            ax.yaxis.grid(True, linestyle=':', alpha=0.4)
            ax.set_axisbelow(True)
            ax.invert_yaxis() 
            ax.set_xlim(x_min, x_max)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            if row_idx == 0:
                ax.set_title(exp, fontsize=11, fontweight='bold', pad=8)

            if col_idx == 0:
                ax.set_ylabel(model, fontsize=10, fontweight='bold', labelpad=8)
            else:
                ax.set_ylabel('')

            if row_idx == 2:
                ax.set_xlabel('Change (%)', fontsize=8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Positive change'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Negative change'),
        mpatches.Patch(facecolor='gray', edgecolor='black', hatch='///', label='Not significant'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='Significant ($p < 0.05$)'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True
    )


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved facet bar plot to: {output_file}")
    plt.close()

def collect_per_order_results(analyzer: McNemarAnalyzer) -> pd.DataFrame:
    """
    Collect results separately for each label order (pos_neg and neg_pos).
    Used for the label order comparison scatter plot.
    """
    model_configs = [
        ('llama', 'single'),
        ('qwen', 'single'),
        ('prometheus', 'pairwise')
    ]
    explanation_types = ['shap', 'lime', 'attention']
    formats = [f for f in McNemarAnalyzer.EXPLANATION_FORMATS if f != 'baseline']
    cot = 'no_chain_of_thought'

    model_display = {
        'llama': 'Llama',
        'qwen': 'Qwen',
        'prometheus': 'M-Prometheus'
    }
    explanation_display = {
        'shap': 'SHAP',
        'lime': 'LIME',
        'attention': 'Attention'
    }
    format_display = {
        f: f.replace('_', ' ').title() for f in formats
    }

    rows = []

    for llm, task_type in model_configs:
        for exp in explanation_types:
            for fmt in formats:
                try:
                    config_pos_neg = ExperimentConfig(llm, task_type, cot, 'pos_neg', exp)
                    config_neg_pos = ExperimentConfig(llm, task_type, cot, 'neg_pos', exp)

                    comp_pos_neg = analyzer.compare_formats(config_pos_neg, 'baseline', fmt)
                    comp_neg_pos = analyzer.compare_formats(config_neg_pos, 'baseline', fmt)

                    rows.append({
                        'model': model_display[llm],
                        'explanation': explanation_display[exp],
                        'format': format_display[fmt],
                        'absolute_change_pos_neg': comp_pos_neg['absolute_change'] * 100,
                        'absolute_change_neg_pos': comp_neg_pos['absolute_change'] * 100,
                        'significant_pos_neg': comp_pos_neg['significant'],
                        'significant_neg_pos': comp_neg_pos['significant'],
                        'baseline_pos_neg': comp_pos_neg['accuracy_baseline'] * 100,
                        'baseline_neg_pos': comp_neg_pos['accuracy_baseline'] * 100,
                        'p_value_pos_neg': comp_pos_neg['p_value'],
                        'p_value_neg_pos': comp_neg_pos['p_value'],
                    })
                except Exception:
                    continue

    return pd.DataFrame(rows)


def correlation_analysis(df: pd.DataFrame):
    x = df['absolute_change']
    y = df['baseline']

    # Normality checks
    _, p_norm_x = stats.shapiro(x)
    _, p_norm_y = stats.shapiro(y)

    print("=== Normality (Shapiro-Wilk) ===")
    print(f"  absolute_change: p = {p_norm_x:.4f} {'✓ normal' if p_norm_x > 0.05 else '✗ not normal'}")
    print(f"  baseline:        p = {p_norm_y:.4f} {'✓ normal' if p_norm_y > 0.05 else '✗ not normal'}")

    both_normal = p_norm_x > 0.05 and p_norm_y > 0.05

    # Pearson
    r_pearson, p_pearson = stats.pearsonr(x, y)
    # Spearman
    r_spearman, p_spearman = stats.spearmanr(x, y)

    print("\n=== Correlation Results ===")
    print(f"  Pearson  r = {r_pearson:.4f},  p = {p_pearson:.4f} {'*' if p_pearson < 0.05 else ''}")
    print(f"  Spearman ρ = {r_spearman:.4f},  p = {p_spearman:.4f} {'*' if p_spearman < 0.05 else ''}")

    print("\n=== Recommendation ===")
    if both_normal:
        print("  Both variables are normal → Pearson is appropriate.")
        print(f"  → r = {r_pearson:.4f}, p = {p_pearson:.4f}")
    else:
        print("  At least one variable is non-normal → use Spearman.")
        print(f"  → ρ = {r_spearman:.4f}, p = {p_spearman:.4f}")

    return {
        'shapiro_x': p_norm_x,
        'shapiro_y': p_norm_y,
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
    }



def plot_accuracy_vs_change(df: pd.DataFrame,
                             output_file: str = 'figures/scatter_accuracy_vs_change.png'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_markers = {
        'Llama': 'o',
        'Qwen': '*',
        'M-Prometheus': '^'
    }
    explanation_colors = {
        'SHAP': '#2196F3',
        'LIME': '#FF9800',
        'Attention': '#9C27B0'
    }

    # --- Run correlation analysis first ---
    corr_results = correlation_analysis(df)
    both_normal = corr_results['shapiro_x'] > 0.05 and corr_results['shapiro_y'] > 0.05
    if both_normal:
        corr_label = f"Pearson $r = {corr_results['pearson_r']:.3f}$, $p = {corr_results['pearson_p']:.3f}$"
    else:
        corr_label = f"Spearman $\\rho = {corr_results['spearman_r']:.3f}$, $p = {corr_results['spearman_p']:.3f}$"

    fig, ax = plt.subplots(figsize=(12, 5))

    for model, marker in model_markers.items():
        for exp, color in explanation_colors.items():
            subset = df[(df['model'] == model) & (df['explanation'] == exp)]
            ax.scatter(
                subset['absolute_change'],
                subset['baseline'],
                marker=marker,
                color=color,
                s=200,
                alpha=0.6,
                zorder=3,
            )

    ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(88, 95)

    ax.set_xlabel('Change in Accuracy (%)', fontsize=11)
    ax.set_ylabel('Baseline Accuracy (%)', fontsize=11)


    # Trend line — dashed to signal non-significant correlation
    z = np.polyfit(df['absolute_change'], df['baseline'], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(df['absolute_change'].min(), df['absolute_change'].max(), 200)
    r_squared = np.corrcoef(df['absolute_change'], df['baseline'])[0, 1] ** 2


    # Legend
    model_legend = [
        plt.scatter([], [], marker=m, color='gray', s=80, label=mod)
        for mod, m in model_markers.items()
    ]
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]

    all_handles = (
            [mpatches.Patch(color='none', label=r'$\bf{Model}$')] +
            model_legend +
            [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +
            exp_legend +
            [
                mpatches.Patch(color='none', label=r'$\bf{Correlation}$'),
                mpatches.Patch(color='none', label=f"Spearman $\\rho = {corr_results['spearman_r']:.3f}$, $p = {corr_results['spearman_p']:.3f}$ (n.s.)"),
            ]
        )
    ax.legend(handles=all_handles, loc='upper right', fontsize=9,
              frameon=True, handlelength=1.5, borderpad=0.8)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy vs change scatter to: {output_file}")
    plt.close()




def plot_label_order_comparison(df_order: pd.DataFrame,
                                 output_file: str = 'figures/scatter_label_order.png'):
    """
    Scatter plot comparing relative change for pos_neg vs neg_pos label orders.
    Four marker shapes:
    - Circle: neither ordering significant
    - Triangle up: only pos_neg significant
    - Triangle down: only neg_pos significant
    - Star: both orderings significant
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    models = ['Llama', 'Qwen', 'M-Prometheus']
    explanation_colors = {
        'SHAP': '#2196F3',
        'LIME': '#FF9800',
        'Attention': '#9C27B0'
    }

    # Marker definitions
    # (condition_function, marker, label, size)
    def get_marker(sig_pos_neg, sig_neg_pos):
        if not sig_pos_neg and not sig_neg_pos:
            return 'o', 60, 'Neither significant'
        elif sig_pos_neg and not sig_neg_pos:
            return '^', 80, 'Only POSITIVE first significant'
        elif not sig_pos_neg and sig_neg_pos:
            return 'v', 80, 'Only NEGATIVE first significant'
        else:
            return '*', 120, 'Both significant'

    # Compute global symmetric axis limits
    all_vals = pd.concat([
        df_order['absolute_change_neg_pos'],
        df_order['absolute_change_pos_neg']
    ])
    abs_max = max(abs(all_vals.min()), abs(all_vals.max())) + 0.5
    global_min = -abs_max
    global_max = abs_max

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    # Track which marker/label combinations have been added to legend
    legend_markers_added = set()
    legend_handles = []

    for col_idx, model in enumerate(models):
        ax = axes[col_idx]
        subset = df_order[df_order['model'] == model]

        for exp, color in explanation_colors.items():
            exp_subset = subset[subset['explanation'] == exp]

            for _, row in exp_subset.iterrows():
                marker, size, marker_label = get_marker(
                    row['significant_pos_neg'],
                    row['significant_neg_pos']
                )

                ax.scatter(
                    row['absolute_change_neg_pos'],
                    row['absolute_change_pos_neg'],
                    marker=marker,
                    color=color,
                    s=size,
                    alpha=0.75,
                    zorder=3
                )

                # Collect unique marker types for legend
                if marker_label not in legend_markers_added:
                    legend_markers_added.add(marker_label)
                    legend_handles.append(
                        plt.scatter([], [], marker=marker, color='gray',
                                    s=size, alpha=0.75, label=marker_label)
                    )

        # Diagonal line
        ax.plot([global_min, global_max], [global_min, global_max],
                color='black', linewidth=1, linestyle='--', alpha=0.5)

        ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        # Ticks at every 1% interval
        ticks = np.arange(np.floor(global_min), np.ceil(global_max) + 1, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xlabel('Change --- NEGATIVE first (%)', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('Change --- POSITIVE first (%)', fontsize=9)
        ax.grid(alpha=0.3)

    # Build combined legend
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]

    # Sort marker legend by a defined order
    marker_order = [
        'Neither significant',
        'Only POSITIVE first significant',
        'Only NEGATIVE first significant',
        'Both significant'
    ]
    legend_handles_sorted = sorted(
        legend_handles,
        key=lambda h: marker_order.index(h.get_label())
        if h.get_label() in marker_order else 99
    )

    from matplotlib.lines import Line2D
    all_handles = (
        exp_legend +
        legend_handles_sorted +
        [Line2D([0], [0], color='black', linestyle='--',
                label='Diagonal (consistent behavior)')]
    )


    fig.legend(
            handles=all_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.10),
            fontsize=9,
            frameon=True,
            handlelength=1.5,
            borderpad=0.8,
            ncol=len(all_handles) #// 2  # two rows
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved label order comparison scatter to: {output_file}")
    plt.close()



def plot_paired_dot(df: pd.DataFrame, output_file: str = 'figures/paired_dot_results.png'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    models = ['Llama', 'Qwen', 'M-Prometheus']
    explanation_colors = {
        'SHAP': '#2196F3',
        'LIME': '#FF9800',
        'Attention': '#9C27B0'
    }

    row_order = [
        'Text Scores', 'Text Labels',
        'Structured Text Scores', 'Structured Text Labels',
        'Top Words Scores', 'Top Words Labels',
        'Natural Words', 'Part Of Speech'
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharey=True, sharex=True)

    for col_idx, model in enumerate(models):
        ax = axes[col_idx]
        subset = df[df['model'] == model].copy()

        y_positions = {fmt: i for i, fmt in enumerate(row_order)}

        for _, row in subset.iterrows():
            if row['format'] not in y_positions:
                continue

            color = explanation_colors[row['explanation']]
            y = y_positions[row['format']]
            jitter = {'SHAP': -0.2, 'LIME': 0.0, 'Attention': 0.2}[row['explanation']]
            y_jittered = y + jitter
            improved = row['accuracy'] > row['baseline']

            ax.plot(
                [row['baseline'], row['accuracy']],
                [y_jittered, y_jittered],
                color=color, linewidth=1.4, alpha=0.6,
            )
            ax.scatter(row['baseline'], y_jittered, marker='o', color=color, s=30)
            end_marker = '>' if improved else '<'
            ax.scatter(
                row['accuracy'], y_jittered,
                marker=end_marker, facecolors='white',
                edgecolors=color, s=30, linewidths=1,
            )
            if row['significant']:
                x_offset = 0.05 if improved else -0.05
                ax.text(
                    row['accuracy'] + x_offset, y_jittered, '*',
                    color=color, fontsize=13,
                    ha='left' if improved else 'right',
                    va='center', zorder=4
                )

        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=9)


        ax.set_ylabel(model, fontsize=13, fontweight='bold', rotation=90,
                      labelpad=12, va='center')


        if col_idx == len(models) - 1:
            ax.set_xlabel('Accuracy (%)', fontsize=9)
        else:
            ax.set_xlabel('')
        ax.set_xlim(87, 95)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.grid(axis='x', alpha=0.3)
        ax.yaxis.grid(True, linestyle=':', alpha=0.3)
        ax.set_axisbelow(True)
        ax.invert_yaxis() 

    # Legend
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]
    marker_legend = [
        plt.scatter([], [], marker='o', color='gray', s=50, label='Baseline accuracy'),
        plt.scatter([], [], marker='>', facecolors='white', edgecolors='gray',
                    s=80, linewidths=1.5, label='With explanation'),
        plt.Line2D([0], [0], color='none', label='* = significant ($p < 0.05$)')
    ]
    all_handles = exp_legend + marker_legend

    fig.legend(
        handles=all_handles,
        ncol=1,
        fontsize=9,
        bbox_to_anchor=(0.96, 0.92),
        frameon=True
    )


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved paired dot plot to: {output_file}")
    plt.close()


def plot_paired_dot_single(df: pd.DataFrame, output_file: str = 'figures/paired_dot_results_single.png'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    models = ['Llama', 'Qwen', 'M-Prometheus']
    explanations = ['SHAP', 'LIME', 'Attention']

    explanation_colors = {
        'SHAP': '#2196F3',
        'LIME': '#FF9800',
        'Attention': '#9C27B0'
    }
    explanation_offsets = {
        'SHAP': -0.2,
        'LIME': 0.0,
        'Attention': 0.2
    }

    row_order = [
        'Text Scores', 'Text Labels',
        'Structured Text Scores', 'Structured Text Labels',
        'Top Words Scores', 'Top Words Labels',
        'Natural Words', 'Part Of Speech'
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_positions = {fmt: i for i, fmt in enumerate(row_order)}

    for _, row in df.iterrows():
        if row['format'] not in y_positions:
            continue

        color = explanation_colors[row['explanation']]
        y = y_positions[row['format']]
        y_jittered = y + explanation_offsets[row['explanation']]
        improved = row['accuracy'] > row['baseline']

        # Connecting line
        ax.plot(
            [row['baseline'], row['accuracy']],
            [y_jittered, y_jittered],
            color=color,
            linewidth=1.6,
            alpha=0.6,
        )

        # Baseline: filled dot
        ax.scatter(
            row['baseline'], y_jittered,
            marker='o',
            color=color,
            s=25,
            zorder=3
        )


        # Explanation end: hollow > or 
        end_marker = '>' if improved else '<'
        ax.scatter(
            row['accuracy'], y_jittered,
            marker=end_marker,
            facecolors='white',
            edgecolors=color,
            s=25,
            linewidths=1.5,
            zorder=3
        )

        # Asterisk for significant results
        if row['significant']:
            x_offset = 0.05 if improved else -0.05
            ax.text(
                row['accuracy'] + x_offset, y_jittered,
                '*',
                color=color,
                fontsize=13,
                ha='left' if improved else 'right',
                va='center',
                zorder=4
            )

    # Extend y-axis to make room at the top
    ax.set_ylim(-0.5, len(row_order) - 0.5 + 1)  

    # Write model names once, above the first row
    for model in models:
        model_data = df[df['model'] == model]
        if model_data.empty:
            continue
        x_baseline = model_data['baseline'].mean()
        ax.text(
            x_baseline, len(row_order) - 0.5 + 0.6,
            model,
            fontsize=12,
            ha='center',
            va='center',
            color='#444444',
            style='italic',
            fontweight='bold',
            transform=ax.transData
        )

    # Subtle separator line between the labels and the first row
    ax.axhline(
        y=len(row_order) - 0.5 + 0.3,
        color='gray',
        linewidth=0.5,
        linestyle=':',
        alpha=0.5
    )

    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=10)
    ax.set_xlim(87, 95)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.grid(axis='x', alpha=0.3)
    ax.yaxis.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)


    # Legend — single row at the bottom
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]
    marker_legend = [
        plt.scatter([], [], marker='o', color='gray', s=40,
                    label='Baseline accuracy'),
        plt.scatter([], [], marker='>', facecolors='white', edgecolors='gray',
                    s=40, linewidths=1,
                    label='With explanation'),
        plt.Line2D([0], [0], color='none', label='* = significant ($p < 0.05$)')
    ]

    all_handles = (
        exp_legend +
        marker_legend
    )

    fig.legend(
        handles=all_handles,
        loc='lower center',
        ncol=len(all_handles),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        handlelength=1.5,
        columnspacing=0.8
    )


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved paired dot plot to: {output_file}")
    plt.close()