import helper_model
import argparse


def main():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--explanations", 
    #                     type=bool,  
    #                     required=True)
    parser.add_argument("--data_size",
                        type=int,
                        required=True)
    args = parser.parse_args()


    # Test WITHOUT explanations
    print("=" * 70)
    print("TEST 1: WITHOUT EXPLANATIONS")
    print("=" * 70)
    
    groups_file = "similarity_groups/similarity_groups.json"
    dev_data_file = "explanations4NLP/merged_data/merged_data.json"

    results_baseline = helper_model.test_experiment(
        groups_file=groups_file,
        dev_data_file=dev_data_file,
        num_test_instances=args.data_size,
        use_explanations=False,
        output_file="test_results_baseline.json"
    )
    
    print("\n\n")
    
    # Test WITH explanations
    print("=" * 70)
    print("TEST 2: WITH EXPLANATIONS")
    print("=" * 70)
    
    results_with_exp = helper_model.test_experiment(
        groups_file=groups_file,
        dev_data_file=dev_data_file,
        num_test_instances=args.data_size,
        use_explanations=True,
        explanation_format="text_labels",
        output_file="test_results_with_explanations.json"
    )

if __name__ == "__main__":
    main()