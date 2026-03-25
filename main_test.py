from pathlib import Path

data_set = 'dev'

paths = {
    'shap':      (f'explanations/pkl/shap/{data_set}_set/shap_raw',       'shap_values_*.pkl'),
    'lime':      (f'explanations/pkl/lime/{data_set}_set/lime_raw',       'lime_explanation_*.pkl'),
    'attention': (f'explanations/pkl/attention/{data_set}_set/attention_raw', 'attention_explanation_*.pkl'),
}

for name, (directory, pattern) in paths.items():
    p = Path(directory)
    print(f"\n{'='*60}")
    print(f"{name.upper()}")
    print(f"  Directory: {p.resolve()}")
    print(f"  Exists:    {p.exists()}")
    if p.exists():
        all_files = list(p.iterdir())
        matched   = list(p.glob(pattern))
        print(f"  All files in dir ({len(all_files)}): {[f.name for f in all_files[:10]]}")
        print(f"  Matched '{pattern}' ({len(matched)}): {[f.name for f in matched[:10]]}")
    else:
        # Walk up to find where the path breaks
        for i in range(len(p.parts), 0, -1):
            parent = Path(*p.parts[:i])
            if parent.exists():
                print(f"  Last existing parent: {parent.resolve()}")
                print(f"  Contents: {[x.name for x in parent.iterdir()]}")
                break