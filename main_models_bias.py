import math
import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

# --- Config ---
MODELS = [
    "Unbabel/M-Prometheus-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "yash3056/Llama-3.2-1B-imdb",
]

CHUNK_SIZE = 500

# --- GPU check ---
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. This script requires a CUDA-capable GPU.")

device = 0  # use first GPU

# --- Load dataset ---
dataset = load_dataset("imdb", split="test").select(range(7500, 17500))
texts = dataset["text"]
true_labels = dataset["label"]  # 0 = neg, 1 = pos

# --- Label normalization ---
def to_binary(label_str):
    label_str = label_str.upper()
    if label_str in ("POSITIVE", "POS", "LABEL_1", "1"):
        return 1
    elif label_str in ("NEGATIVE", "NEG", "LABEL_0", "0"):
        return 0
    else:
        return -1  # unknown

# --- Run each model ---
results = {}

for model_name in tqdm(MODELS, desc="Models", position=0):
    print(f"\n=== Loading {model_name} ===")
    try:
        pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )

        # Fix missing pad token for causal/generative models
        # (same approach used in LLMInference.generate via pad_token_id=eos_token_id)
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
            print(f"  Set pad_token_id = eos_token_id ({pipe.tokenizer.pad_token_id})")

        preds = []
        n_chunks = math.ceil(len(texts) / CHUNK_SIZE)
        for i in tqdm(range(0, len(texts), CHUNK_SIZE), total=n_chunks, desc=f"  Classifying", position=1, leave=False):
            batch = texts[i:i + CHUNK_SIZE]
            batch_preds = pipe(batch, batch_size=16)
            preds.extend(batch_preds)

        binary_preds = [to_binary(p["label"]) for p in preds]
        n_pos = sum(p == 1 for p in binary_preds)
        n_neg = sum(p == 0 for p in binary_preds)
        accuracy = sum(p == t for p, t in zip(binary_preds, true_labels)) / len(true_labels)

        results[model_name] = {
            "predictions": binary_preds,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "pos_ratio": n_pos / len(binary_preds),
            "accuracy": accuracy,
        }

        print(f"  Positive: {n_pos} | Negative: {n_neg} | Pos ratio: {n_pos/len(binary_preds):.2%} | Accuracy: {accuracy:.2%}")

    except Exception as e:
        print(f"  ERROR with {model_name}: {e}")
        results[model_name] = {"error": str(e)}

# --- Summary ---
print("\n=== Bias Summary ===")
for model, r in results.items():
    if "error" not in r:
        print(f"{model}: pos_ratio={r['pos_ratio']:.2%}, accuracy={r['accuracy']:.2%}")
    else:
        print(f"{model}: ERROR - {r['error']}")