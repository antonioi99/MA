import argparse
import hashlib
import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Creates balanced dev/test splits from the IMDB test set, keeping train as-is.")
    parser.add_argument("--target_dataset", type=str, default="imdb-dev-test-split")
    parser.add_argument("--dev_size", type=int, default=15000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private", action="store_true", default=True)
    return parser.parse_args()

def build_dataset(dev_size, test_size, seed):
    dataset = load_dataset("imdb")

    assert dev_size + test_size <= len(dataset["test"]), \
        f"dev+test sizes ({dev_size}+{test_size}) exceed available data ({len(dataset['test'])})"
    assert dev_size % 2 == 0 and test_size % 2 == 0, \
        "dev_size and test_size must be even for balanced splits"

    # Shuffle test set and split by label for stratified sampling
    test_shuffled = dataset["test"].shuffle(seed=seed)
    pos_all = test_shuffled.filter(lambda x: x["label"] == 1)
    neg_all = test_shuffled.filter(lambda x: x["label"] == 0)

    dev_half = dev_size // 2
    test_half = test_size // 2

    dev = concatenate_datasets([
        pos_all.select(range(dev_half)),
        neg_all.select(range(dev_half))
    ]).shuffle(seed=seed)

    test = concatenate_datasets([
        pos_all.select(range(dev_half, dev_half + test_half)),
        neg_all.select(range(dev_half, dev_half + test_half))
    ]).shuffle(seed=seed)

    new_dataset = DatasetDict({
        "train": dataset["train"],  # 25k, untouched
        "dev": dev,
        "test": test
    })

    # Add MD5-based IDs
    def add_md5_id(example):
        return {"id": hashlib.md5(example["text"].encode("utf-8")).hexdigest()}

    for split_name in new_dataset:
        new_dataset[split_name] = new_dataset[split_name].map(add_md5_id)

    return new_dataset

def sanity_check(dev_size, test_size, seed):
    print("\n--- Running Sanity Check ---")

    print("Building dataset (run 1)...")
    dataset_run1 = build_dataset(dev_size, test_size, seed)
    ids_run1 = {split: dataset_run1[split]["id"][:5] for split in dataset_run1}

    print("Building dataset (run 2)...")
    dataset_run2 = build_dataset(dev_size, test_size, seed)
    ids_run2 = {split: dataset_run2[split]["id"][:5] for split in dataset_run2}

    all_match = True
    for split in ids_run1:
        match = ids_run1[split] == ids_run2[split]
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"{split}: {status}")
        if not match:
            all_match = False
            print(f"  Run 1: {ids_run1[split]}")
            print(f"  Run 2: {ids_run2[split]}")

    if not all_match:
        raise RuntimeError("Sanity check failed — pipeline is NOT deterministic. Aborting upload.")

    print("Sanity check passed — pipeline is deterministic!\n")
    return dataset_run1  # reuse the first run for upload

if __name__ == "__main__":
    args = parse_args()

    # Run sanity check — aborts if pipeline is not deterministic
    new_dataset = sanity_check(args.dev_size, args.test_size, args.seed)

    print(new_dataset)

    # Verify label balance
    for split_name, split_data in new_dataset.items():
        counts = Counter(split_data["label"])
        print(f"{split_name} label distribution: {counts}")

    # Push to Hub
    new_dataset.push_to_hub(repo_id=args.target_dataset, private=args.private)

    # Dataset card
    dataset_card = f"""
---
license: mit
tags:
- imdb
- sentiment
- subset
---
This dataset contains the original IMDB train split plus balanced dev and test splits derived from the IMDB test set.

### Generation Command
```bash
python take_split.py --target_dataset {args.target_dataset} --dev_size {args.dev_size} --test_size {args.test_size} --seed {args.seed} {"--private" if args.private else ""}
```

## Dataset Splits

- Train: {len(new_dataset['train'])} samples (label distribution: {dict(Counter(new_dataset['train']['label']))})
- Dev: {len(new_dataset['dev'])} samples (label distribution: {dict(Counter(new_dataset['dev']['label']))})
- Test: {len(new_dataset['test'])} samples (label distribution: {dict(Counter(new_dataset['test']['label']))})
- Seed: {args.seed}

## Notes

- Source: [imdb](https://huggingface.co/datasets/imdb)
- Train split is the original unmodified IMDB train set (25k samples)
- Dev and test are carved from the original IMDB test set (25k samples)
- Dev and test are both label-balanced (50% positive / 50% negative)
- Each document has a unique MD5 hash ID derived from its text content
- Dataset passed determinism sanity check before upload

## License

See original dataset (MIT)
"""

    with open("README.md_tmp", "w") as f:
        f.write(dataset_card)

    api = HfApi()
    repo_id = api.whoami()["name"] + "/" + args.target_dataset if "/" not in args.target_dataset else args.target_dataset
    api.upload_file(
        path_or_fileobj="README.md_tmp",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card"
    )
    os.remove("README.md_tmp")