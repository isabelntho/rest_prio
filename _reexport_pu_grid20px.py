"""Temporary script: re-export all pu_grid20px runs with the fixed pixel expansion."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_to_r import export_results

root = Path("r_inputs")

parent_dirs = [d for d in root.iterdir() if d.is_dir() and "pu_grid20px" in d.name]
print(f"Found {len(parent_dirs)} pu_grid20px parent folders")

tasks = []
for parent in sorted(parent_dirs):
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        pkl_name = meta.get("pkl_file")
        if not pkl_name:
            continue
        pkl_path = Path("results_files") / pkl_name
        tasks.append((pkl_path, sub))

print(f"Total runs to re-export: {len(tasks)}")

n_ok = n_missing = n_failed = 0
for pkl_path, out_dir in tasks:
    if not pkl_path.exists():
        print(f"  MISSING  {pkl_path.name}")
        n_missing += 1
        continue
    print(f"  Exporting {pkl_path.name} -> {out_dir}")
    try:
        export_results(str(pkl_path), output_dir=str(out_dir), nondom_pixels_only=True)
        n_ok += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        n_failed += 1

print(f"\nDone -- exported: {n_ok} | missing: {n_missing} | failed: {n_failed}")
