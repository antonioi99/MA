import json

with open("examples/explanations_examples.json", "r") as f:
    data = json.load(f)

results = []
for entry_id, entry in data.items():
    results.append({
        "id": entry_id,
        "label": entry["label"],
        "shap_structured_text_scores": entry["shap"]["structured_text_scores"],
        "lime_structured_text_scores": entry["lime"]["structured_text_scores"],
        "attention_structured_text_scores": entry["attention"]["structured_text_scores"],
    })


with open("examples/extracted_explanations.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n\n\n\nDone!")