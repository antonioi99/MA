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
import glob
import krippendorff


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
    explanations = ['SHAP', 'LIME', 'Attention']

    row_order = [
        'Text Scores', 'Text Labels',
        'Structured Text Scores', 'Structured Text Labels',
        'Top Words Scores', 'Top Words Labels',
        'Natural Words', 'Part Of Speech'
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(13, 9),  # reduced from (16, 14)
        sharex=False, sharey=True
    )

    x_min = df['relative_change'].min() - 0.5
    x_max = df['relative_change'].max() + 0.5

    for row_idx, model in enumerate(models):
        for col_idx, exp in enumerate(explanations):
            ax = axes[row_idx, col_idx]

            subset = df[(df['model'] == model) & (df['explanation'] == exp)].copy()
            subset = subset.set_index('format').reindex(row_order).reset_index()

            for i, r in subset.iterrows():
                color = '#2ecc71' if r['relative_change'] > 0 else '#e74c3c'
                hatch = '' if r['significant'] else '///'

                ax.barh(
                    r['format'],
                    r['relative_change'],
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
                ax.set_xlabel('Relative Change (%)', fontsize=8)

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

    fig.suptitle(
        'Relative Change in Accuracy by Model, Explanation Method, and Verbalization Format',
        fontsize=12, fontweight='bold', y=1.01
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

    # Trend line
    z = np.polyfit(df['relative_change'], df['baseline'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['relative_change'].min(), df['relative_change'].max(), 200)
    correlation = np.corrcoef(df['relative_change'], df['baseline'])[0, 1]
    r_squared = correlation ** 2
    ax.plot(x_line, p(x_line), color='black', linewidth=1.5, linestyle='-', alpha=0.6,
            label=f'Trend line ($R^2={r_squared:.2f}$)')

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
        [mpatches.Patch(color='none', label=r'$\bf{Model}$')] +
        model_legend +
        [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +
        exp_legend +
        [plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', alpha=0.6, label=f'Trend line ($R^2={r_squared:.2f}$)')]  # added
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





def compute_agreement_from_raw(
    base_path: str = 'test_results',
    output_file: str = 'tables/agreement/agreement_raw.tex'
):
    """
    For each model x verbalization format combination, load per-item classifications
    for pos_neg and neg_pos orderings, align them by test_id, discard missing/hallucinated
    responses, then compute Krippendorff's alpha (nominal).
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_configs = [
        ('llama',      'single'),
        ('qwen',       'single'),
        ('prometheus', 'pairwise')
    ]
    explanation_types = ['shap', 'lime', 'attention']
    cot = 'no_chain_of_thought'

    format_files = [
        'baseline',
        'text_scores', 'text_labels',
        'structured_text_scores', 'structured_text_labels',
        'top_words_scores', 'top_words_labels',
        'natural_words', 'part_of_speech'
    ]
    format_display = {f: f.replace('_', ' ').title() for f in format_files}

    model_display = {
        'llama': 'Llama',
        'qwen': 'Qwen',
        'prometheus': 'M-Prometheus'
    }

    VALID_LABELS = {0, 1}

    def load_classifications(path: str) -> dict:
        """Load a json file and return {test_id: predicted_label}.
        Discards items with missing or invalid labels."""
        with open(path, 'r') as f:
            data = json.load(f)
        result = {}
        for item in data:
            label = item.get('predicted_label_LLM')
            if label in VALID_LABELS:
                result[item['test_id']] = label
        return result

    rows = []

    for llm, task_type in model_configs:
        for exp in explanation_types:
            for fmt in format_files:
                path_pos_neg = f'{base_path}/{llm}/{exp}/{task_type}/{cot}/pos_neg/{fmt}.json'
                path_neg_pos = f'{base_path}/{llm}/{exp}/{task_type}/{cot}/neg_pos/{fmt}.json'

                if not os.path.exists(path_pos_neg) or not os.path.exists(path_neg_pos):
                    print(f"Missing files for {llm}-{exp}-{fmt}, skipping.")
                    continue

                labels_pos_neg = load_classifications(path_pos_neg)
                labels_neg_pos = load_classifications(path_neg_pos)

                # Align by test_id — keep only items present and valid in both
                common_ids = set(labels_pos_neg.keys()) & set(labels_neg_pos.keys())
                n_total = max(len(labels_pos_neg), len(labels_neg_pos))
                n_discarded = n_total - len(common_ids)

                if len(common_ids) < 2:
                    print(f"Not enough aligned items for {llm}-{exp}-{fmt}, skipping.")
                    continue

                vec_pos_neg = np.array([labels_pos_neg[i] for i in common_ids])
                vec_neg_pos = np.array([labels_neg_pos[i] for i in common_ids])

                # Krippendorff's alpha (nominal)
                reliability_data = np.array([vec_pos_neg, vec_neg_pos])
                alpha = krippendorff.alpha(
                    reliability_data,
                    level_of_measurement='nominal'
                )

                rows.append({
                    'model': model_display[llm],
                    'explanation': exp.upper(),
                    'format': format_display[fmt],
                    'alpha': alpha,
                    'n_aligned': len(common_ids),
                    'n_discarded': n_discarded
                })

    df = pd.DataFrame(rows)

    # --- Aggregate across explanation types (mean per model x format) ---
    df_agg = df.groupby(['model', 'format']).agg(
        alpha=('alpha', 'mean'),
        n_aligned=('n_aligned', 'sum'),
        n_discarded=('n_discarded', 'sum')
    ).reset_index()

    # --- Build LaTeX table ---
    models = ['Llama', 'Qwen', 'M-Prometheus']
    row_order = [format_display[f] for f in format_files]

    col_spec = 'l' + 'r' * len(models)

    header = ' & '.join(
        [r'\textbf{Format}'] +
        [rf'\textbf{{{m}}}' for m in models]
    ) + r' \\'

    subheader = ' & '.join(
        [''] + [r'$\alpha_{K}$' for _ in models]
    ) + r' \\'

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header,
        subheader,
        r'\midrule',
    ]

    for fmt in row_order:
        # Add a visual separator before baseline vs the rest
        if fmt == format_display['baseline']:
            pass  # baseline is first, no separator needed
        cells = [fmt]
        for model in models:
            match = df_agg[(df_agg['model'] == model) & (df_agg['format'] == fmt)]
            if match.empty:
                cells += ['---']
            else:
                alpha = match['alpha'].values[0]
                cells += [f'{alpha:.3f}']
        lines.append(' & '.join(cells) + r' \\')

        # Add a midrule after baseline to visually separate it
        if fmt == format_display['baseline']:
            lines.append(r'\midrule')

    # Overall row
    lines.append(r'\midrule')
    overall_cells = [r'\textit{Overall}']
    for model in models:
        model_rows = df_agg[df_agg['model'] == model]
        mean_alpha = model_rows['alpha'].mean()
        overall_cells += [rf'\textit{{{mean_alpha:.3f}}}']
    lines.append(' & '.join(overall_cells) + r' \\')

    # Compute totals for caption
    total_discarded = df['n_discarded'].sum()
    total_items = df['n_aligned'].sum() + total_discarded
    pct_discarded = total_discarded / total_items * 100

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        rf'\caption{{Per-item agreement between positive-first and negative-first label orderings, '
        rf"measured by Krippendorff's $\alpha$ (nominal), averaged across explanation methods "
        rf'(SHAP, LIME, Attention). A total of {total_discarded} items '
        rf'({pct_discarded:.1f}\% of all classifications) were discarded due to missing '
        rf'or hallucinated responses. A horizontal rule separates the baseline (no explanation) '
        rf'from verbalization formats.}}',
        r'\label{tab:agreement_raw}',
        r'\end{table}',
    ]

    latex = '\n'.join(lines)
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"Saved LaTeX table to: {output_file}")
    df['pct_discarded'] = df['n_discarded'] / (df['n_aligned'] + df['n_discarded']) * 100
    print(f"Mean discard rate per configuration: {df['pct_discarded'].mean():.2f}%")

    return df, df_agg

# def plot_beeswarm(df: pd.DataFrame, output_file: str = 'figures/beeswarm_results.png'):
#     """
#     Beeswarm / strip plot showing distribution of relative changes.
#     Rows: verbalization formats
#     Columns: models
#     Each dot: one explanation method (SHAP, LIME, Attention)
#     Color: explanation method
#     Filled: significant, empty: not significant
#     """
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     models = ['Llama', 'Qwen', 'M-Prometheus']
#     explanations = ['SHAP', 'LIME', 'Attention']

#     row_order = [
#         'Text Scores', 'Text Labels',
#         'Structured Text Scores', 'Structured Text Labels',
#         'Top Words Scores', 'Top Words Labels',
#         'Natural Words', 'Part Of Speech'
#     ]

#     explanation_colors = {
#         'SHAP': '#2196F3',
#         'LIME': '#FF9800',
#         'Attention': '#9C27B0'
#     }

#     fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True, sharex=False)

#     x_min = df['relative_change'].min() - 0.5
#     x_max = df['relative_change'].max() + 0.5

#     for col_idx, model in enumerate(models):
#         ax = axes[col_idx]
#         subset = df[df['model'] == model].copy()

#         # Map format to y position directly without set_index
#         y_positions = {fmt: i for i, fmt in enumerate(row_order)}

#         for _, row in subset.iterrows():
#             if pd.isna(row['relative_change']) or row['format'] not in y_positions:
#                 continue

#             color = explanation_colors[row['explanation']]
#             y = y_positions[row['format']]
#             jitter = np.random.uniform(-0.15, 0.15)

#             ax.scatter(
#                 row['relative_change'],
#                 y + jitter,
#                 color=color,
#                 facecolors=color if row['significant'] else 'white',
#                 edgecolors=color,
#                 s=70,
#                 linewidths=1.5,
#                 alpha=0.85,
#                 zorder=3
#             )


#         ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
#         ax.set_xlim(x_min, x_max)
#         ax.set_yticks(range(len(row_order)))
#         ax.set_yticklabels(row_order if col_idx == 0 else [], fontsize=9)
#         ax.set_title(model, fontsize=12, fontweight='bold')
#         ax.set_xlabel('Relative Change (%)', fontsize=9)
#         ax.grid(axis='x', alpha=0.3)
#         ax.yaxis.grid(True, linestyle=':', alpha=0.3)
#         ax.set_axisbelow(True)

#     # Legend
#     exp_legend = [
#         mpatches.Patch(color=c, label=exp)
#         for exp, c in explanation_colors.items()
#     ]
#     sig_legend = [
#         plt.scatter([], [], facecolors='gray', edgecolors='gray',
#                     s=70, label='Significant ($p < 0.05$)'),
#         plt.scatter([], [], facecolors='white', edgecolors='gray',
#                     s=70, linewidths=1.5, label='Not significant'),
#     ]

#     all_handles = (
#         [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +
#         exp_legend +
#         [mpatches.Patch(color='none', label=r'$\bf{Significance}$')] +
#         sig_legend
#     )

#     fig.legend(
#         handles=all_handles,
#         loc='lower center',
#         ncol=6,
#         fontsize=9,
#         bbox_to_anchor=(0.5, -0.05),
#         frameon=True
#     )

#     fig.suptitle(
#         'Distribution of Relative Change by Model and Verbalization Format\n'
#         '(each dot = one explanation method, filled = significant $p < 0.05$)',
#         fontsize=12, fontweight='bold'
#     )

#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"Saved beeswarm plot to: {output_file}")
#     plt.close()

def plot_paired_dot(df: pd.DataFrame, output_file: str = 'figures/paired_dot_results.png'):
    """
    Paired dot plot showing baseline accuracy and accuracy-with-explanation
    connected by a line for each verbalization format.
    Rows: verbalization formats
    Columns: models
    Color: explanation method
    Line color: green if improvement, red if decrease
    """
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True, sharex=False)

    for col_idx, model in enumerate(models):
        ax = axes[col_idx]
        subset = df[df['model'] == model].copy()

        y_positions = {fmt: i for i, fmt in enumerate(row_order)}

        for _, row in subset.iterrows():
            if row['format'] not in y_positions:
                continue

            color = explanation_colors[row['explanation']]
            y = y_positions[row['format']]

            # Small jitter per explanation method to avoid overlap
            jitter = {'SHAP': -0.2, 'LIME': 0.0, 'Attention': 0.2}[row['explanation']]
            y_jittered = y + jitter

            # Line connecting baseline to accuracy
            line_color = '#2ecc71' if row['accuracy'] > row['baseline'] else '#e74c3c'
            line_style = '-' if row['significant'] else '--'

            ax.plot(
                [row['baseline'], row['accuracy']],
                [y_jittered, y_jittered],
                color=line_color,
                linewidth=1.5,
                linestyle=line_style,
                alpha=0.7,
                zorder=2
            )

            # Baseline dot (hollow)
            ax.scatter(
                row['baseline'], y_jittered,
                color=color,
                facecolors='white',
                edgecolors=color,
                s=60,
                linewidths=1.5,
                zorder=3
            )

            # Accuracy dot (filled)
            ax.scatter(
                row['accuracy'], y_jittered,
                color=color,
                facecolors=color,
                s=60,
                zorder=3
            )

        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels(row_order if col_idx == 0 else [], fontsize=9)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xlabel('Accuracy (%)', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.yaxis.grid(True, linestyle=':', alpha=0.3)
        ax.set_axisbelow(True)

    # Legend
    exp_legend = [
        mpatches.Patch(color=c, label=exp)
        for exp, c in explanation_colors.items()
    ]
    line_legend = [
        plt.Line2D([0], [0], color='#2ecc71', linewidth=1.5, label='Improvement'),
        plt.Line2D([0], [0], color='#e74c3c', linewidth=1.5, label='Decrease'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5,
                   linestyle='-', label='Significant ($p < 0.05$)'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5,
                   linestyle='--', label='Not significant'),
    ]
    dot_legend = [
        plt.scatter([], [], facecolors='white', edgecolors='gray',
                    s=60, linewidths=1.5, label='Baseline accuracy'),
        plt.scatter([], [], facecolors='gray', edgecolors='gray',
                    s=60, label='Accuracy with explanation'),
    ]

    all_handles = (
        [mpatches.Patch(color='none', label=r'$\bf{Explanation}$')] +
        exp_legend +
        [mpatches.Patch(color='none', label=r'$\bf{Direction}$')] +
        line_legend +
        [mpatches.Patch(color='none', label=r'$\bf{Dots}$')] +
        dot_legend
    )

    fig.legend(
        handles=all_handles,
        loc='lower center',
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True
    )

    fig.suptitle(
        'Baseline vs. Accuracy with Explanation by Model and Verbalization Format\n'
        '(hollow = baseline, filled = with explanation, line style = significance)',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved paired dot plot to: {output_file}")
    plt.close()