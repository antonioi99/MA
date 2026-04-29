import helper_graphs
from helper_analysis import McNemarAnalyzer

def main():

    analyzer = McNemarAnalyzer(base_path="analysis")

    print("Collecting aggregated results...")
    df = helper_graphs.collect_aggregated_results(analyzer)

    print("Collecting per-order results...")
    df_order = helper_graphs.collect_per_order_results(analyzer)

    print("\nGenerating heatmap...")
    helper_graphs.plot_heatmap(df, output_file='figures/heatmap_results.png')

    print("\nGenerating facet dot plot...")
    helper_graphs.plot_facet_bar(df, output_file='figures/facet_bar_results.png')

    print("\nGenerating accuracy vs change scatter...")
    helper_graphs.plot_accuracy_vs_change(df, output_file='figures/scatter_accuracy_vs_change.png')


    print("\nGenerating label order comparison scatter...")
    helper_graphs.plot_label_order_comparison(df_order, output_file='figures/scatter_label_order.png')

    print("\n Generating agreement table...")
    helper_graphs.compute_agreement_from_raw()


    print("\nGenerating paired dot plot...")
    helper_graphs.plot_paired_dot(df)

    print("\nGenerating paired dot plot single...")
    helper_graphs.plot_paired_dot_single(df)


    print("\nDone!")


if __name__ == '__main__':
    main()