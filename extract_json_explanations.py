import json
import random

random.seed(42)

with open("explanations/NLP_format/merged_data/merged_data.json", "r") as f:
    data = json.load(f)

entries = list(data.items())
midpoint = len(entries) // 2

first_half = entries[:midpoint]
second_half = entries[midpoint:]

sampled = random.sample(first_half, 25) + random.sample(second_half, 25)
random.shuffle(sampled)

results = []
for entry_id, entry in sampled:
    results.append({
        "id": entry_id,
        "label": entry["label"],
        "shap_text_scores": entry["shap"]["text_scores"],
        "lime_text_scores": entry["lime"]["text_scores"],
        "attention_text_scores": entry["attention"]["text_scores"],
    })

with open("examples/extracted_explanations.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n\nDone! Extracted 50 explanations (25 from each half).")