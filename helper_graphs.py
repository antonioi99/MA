import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict
from helper_analysis import McNemarAnalyzer, AggregatedAnalyzer, ExperimentConfig
import os


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
                        'relative_change': float(comparison['relative_change'] * 100),
                        'p_value': float(comparison['p_value']),
                        'significant': bool(comparison['significant']),
                        'baseline': float(comparison['accuracy_baseline'] * 100),  # added this
                        'accuracy': float(comparison['accuracy_test'] * 100)       # added this
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
        index='format', columns='config', values='relative_change'
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
            'label': 'Relative Change (%)',
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

    ax.set_title(
        'Relative Change in Accuracy by Verbalization Format and Model\n'
        '(* = statistically significant, $p < 0.05$)',
        fontsize=13, pad=15
    )
    ax.set_xlabel('Model --- Explanation Method', fontsize=11)
    ax.set_ylabel('Verbalization Format', fontsize=11)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Add vertical lines to separate models
    for x in [3, 6]:
        ax.axvline(x=x, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")
    plt.close()


def plot_facet_dotplot(df: pd.DataFrame, output_file: str = 'figures/facet_dotplot_results.png'):
    """
    Plot a 3x3 facet grid of dot plots.
    Rows: judge models
    Columns: explanation methods
    Each panel: verbalization formats on y-axis, relative change on x-axis
    Filled dot: significant, empty dot: not significant
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    models = ['Llama', 'Qwen', 'M-Prometheus']
    explanations = ['SHAP', 'LIME', 'Attention']

    row_order = [
        'Text Scores', 'Text Labels',
        'Structured Text Scores', 'Structured Text Labels',
        'Top Words Scores', 'Top Words Labels',
        'Natural Words', 'Part Of Speech'
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(16, 14),
        sharex=False, sharey=True
    )

    # Determine global x range for consistency
    x_min = df['relative_change'].min() - 0.5
    x_max = df['relative_change'].max() + 0.5

    for row_idx, model in enumerate(models):
        for col_idx, exp in enumerate(explanations):
            ax = axes[row_idx, col_idx]

            subset = df[(df['model'] == model) & (df['explanation'] == exp)].copy()
            subset = subset.set_index('format').reindex(row_order).reset_index()

            for i, r in subset.iterrows():
                color = '#2ecc71' if r['relative_change'] > 0 else '#e74c3c'
                marker = 'o' if r['significant'] else 'o'
                fill = color if r['significant'] else 'white'

                ax.scatter(
                    r['relative_change'], r['format'],
                    color=color,
                    facecolors=fill,
                    edgecolors=color,
                    s=80,
                    linewidths=1.5,
                    zorder=3
                )

            # Reference line at 0
            ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)

            # Subtle horizontal grid lines
            ax.yaxis.grid(True, linestyle=':', alpha=0.4)
            ax.set_axisbelow(True)

            ax.set_xlim(x_min, x_max)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # Column headers
            if row_idx == 0:
                ax.set_title(exp, fontsize=12, fontweight='bold', pad=10)

            # Row headers
            if col_idx == 0:
                ax.set_ylabel(model, fontsize=11, fontweight='bold', labelpad=10)
            else:
                ax.set_ylabel('')

            if row_idx == 2:
                ax.set_xlabel('Relative Change (%)', fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Positive change'),
        mpatches.Patch(facecolor='#e74c3c', label='Negative change'),
        plt.scatter([], [], facecolors='gray', edgecolors='gray',
                    s=80, label='Significant ($p < 0.05$)'),
        plt.scatter([], [], facecolors='white', edgecolors='gray',
                    s=80, linewidths=1.5, label='Not significant'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True
    )

    fig.suptitle(
        'Relative Change in Accuracy by Model, Explanation Method, and Verbalization Format',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved facet dot plot to: {output_file}")
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
                        'relative_change_pos_neg': comp_pos_neg['relative_change'] * 100,
                        'relative_change_neg_pos': comp_neg_pos['relative_change'] * 100,
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


def plot_accuracy_vs_change(df: pd.DataFrame, 
                             output_file: str = 'figures/scatter_accuracy_vs_change.png'):
    """
    Scatter plot: baseline accuracy (x) vs relative change (y).
    One point per model-explanation-format combination.
    Shows whether higher baseline accuracy correlates with smaller improvements.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_markers = {
        'Llama': 'o',
        'Qwen': 's',
        'M-Prometheus': '^'
    }
    explanation_colors = {
        'SHAP': '#2196F3',
        'LIME': '#FF9800',
        'Attention': '#9C27B0'
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for model, marker in model_markers.items():
        for exp, color in explanation_colors.items():
            subset = df[(df['model'] == model) & (df['explanation'] == exp)]

            # Significant points --- filled
            sig = subset[subset['significant']]
            ax.scatter(
                sig['relative_change'],
                sig['baseline'],
                marker=marker,
                color=color,
                s=80,
                alpha=0.85,
                zorder=3,
                label=f'{model} — {exp}' if len(sig) > 0 else None
            )

            # Non-significant points --- empty
            nonsig = subset[~subset['significant']]
            ax.scatter(
                nonsig['relative_change'],
                nonsig['baseline'],
                marker=marker,
                facecolors='none',
                edgecolors=color,
                s=80,
                alpha=0.6,
                linewidths=1.5,
                zorder=3
            )

    ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(y=df['baseline'].mean(), color='gray', linewidth=1,
               linestyle=':', alpha=0.5, label='Mean baseline')

    ax.set_xlabel('Relative Change in Accuracy (%)', fontsize=11)
    ax.set_ylabel('Baseline Accuracy (%)', fontsize=11)
    ax.set_title(
        'Baseline Accuracy vs Relative Change\n'
        '(filled = significant $p < 0.05$, empty = not significant)',
        fontsize=12
    )

    # Custom legend for models and explanation methods
    model_legend = [
        plt.scatter([], [], marker=m, color='gray', s=80, label=mod)
        for mod, m in model_markers.items()
    ]
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]


    all_handles = (
        [mpatches.Patch(color='none', label=r'$\bf{Model}$')] +  # group title
        model_legend +
        [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +  # group title
        exp_legend 
    )

    ax.legend(
        handles=all_handles,
        loc='upper right',
        fontsize=9,
        frameon=True,
        handlelength=1.5,
        borderpad=0.8
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy vs change scatter to: {output_file}")
    plt.close()


def plot_volcano(df: pd.DataFrame,
                 output_file: str = 'figures/scatter_volcano.png'):
    """
    Volcano plot: relative change (x) vs -log10(p-value) (y).
    Points in top right: significant improvements.
    Points in top left: significant decreases.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_markers = {
        'Llama': 'o',
        'Qwen': 's',
        'M-Prometheus': '^'
    }
    explanation_colors = {
        'SHAP': '#2196F3',
        'LIME': '#FF9800',
        'Attention': '#9C27B0'
    }

    # Avoid log(0) by clipping p-values
    df = df.copy()
    df['neg_log_p'] = -np.log10(df['p_value'].clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(10, 7))

    significance_threshold = -np.log10(0.05)

    for model, marker in model_markers.items():
        for exp, color in explanation_colors.items():
            subset = df[(df['model'] == model) & (df['explanation'] == exp)]

            ax.scatter(
                subset['relative_change'],
                subset['neg_log_p'],
                marker=marker,
                color=color,
                s=80,
                alpha=0.75,
                zorder=3
            )

    # Significance threshold line
    ax.axhline(
        y=significance_threshold,
        color='black', linewidth=1.2, linestyle='--', alpha=0.7,
        label=f'$p = 0.05$ threshold'
    )

    # Zero change line
    ax.axvline(x=0, color='gray', linewidth=1, linestyle=':', alpha=0.5)

    # Shade quadrants
    x_min, x_max = ax.get_xlim()
    y_max = df['neg_log_p'].max() + 0.5
    ax.fill_betweenx(
        [significance_threshold, y_max], x_min, 0,
        alpha=0.05, color='red', label='Significant decrease'
    )
    ax.fill_betweenx(
        [significance_threshold, y_max], 0, x_max,
        alpha=0.05, color='green', label='Significant improvement'
    )

    ax.set_xlabel('Relative Change in Accuracy (%)', fontsize=11)
    ax.set_ylabel(r'$-\log_{10}(p\text{-value})$', fontsize=11)
    ax.set_title(
        'Volcano Plot: Effect Size vs Statistical Significance\n'
        '(points above dashed line are statistically significant)',
        fontsize=12
    )

    # Custom legend
    model_legend = [
        plt.scatter([], [], marker=m, color='gray', s=80, label=mod)
        for mod, m in model_markers.items()
    ]
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]
    region_legend = [
        mpatches.Patch(color='green', alpha=0.2, label='Significant improvement'),
        mpatches.Patch(color='red', alpha=0.2, label='Significant decrease'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='$p = 0.05$')
    ]


    all_handles = (
        [mpatches.Patch(color='none', label=r'$\bf{Model}$')] +  # group title
        model_legend +
        #[separator] +
        [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +  # group title
        exp_legend +
        #[separator] +
        [mpatches.Patch(color='none', label=r'$\bf{Region}$')] +  # group title
        region_legend
    )

    ax.legend(
        handles=all_handles,
        loc='lower right',
        fontsize=9,
        frameon=True,
        handlelength=1.5,
        borderpad=0.8
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved volcano plot to: {output_file}")
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
        df_order['relative_change_neg_pos'],
        df_order['relative_change_pos_neg']
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
                    row['relative_change_neg_pos'],
                    row['relative_change_pos_neg'],
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
        ax.set_xlabel('Relative Change --- NEGATIVE first (%)', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('Relative Change --- POSITIVE first (%)', fontsize=9)
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
        # [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +
        exp_legend +
        # [mpatches.Patch(color='none', label=r'$\bf{Significance}$')] +
        legend_handles_sorted +
        # [mpatches.Patch(color='none', label=r'$\bf{Reference}$')] +
        [Line2D([0], [0], color='black', linestyle='--',
                label='Diagonal (consistent behavior)')]
    )

    # fig.legend(
    #     handles=all_handles,
    #     loc='upper left',
    #     bbox_to_anchor=(1.01, 1.0),
    #     fontsize=9,
    #     frameon=True,
    #     handlelength=1.5,
    #     borderpad=0.8
    # )

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

    fig.suptitle(
        'Relative Change by Label Order: NEGATIVE first vs POSITIVE first\n'
        '(points on the diagonal are unaffected by positional bias)',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved label order comparison scatter to: {output_file}")
    plt.close()


