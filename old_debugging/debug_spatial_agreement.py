import pickle
import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
import sys
import warnings

# Safe unpickler to handle missing HVCallback
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'HVCallback':
            return lambda *args, **kwargs: None
        # Handle numpy version issues by using current numpy
        if 'numpy' in module:
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                # Try with numpy instead
                if 'numpy._core' in module or 'numpy.core' in module:
                    try:
                        return getattr(np, name, None) or super().find_class('numpy', name)
                    except:
                        pass
        return super().find_class(module, name)

def safe_pickle_load(filepath):
    with open(filepath, 'rb') as f:
        return SafeUnpickler(f).load()

# Load base results
base_results_file = os.path.join('results_files', 'results_fg_2003_1.pkl')
print(f"Base results file: {base_results_file}")
print(f"File exists: {os.path.exists(base_results_file)}\n")

base_results = safe_pickle_load(base_results_file)

# Inspect base results structure
print("BASE RESULTS STRUCTURE:")
print(f"  Type: {type(base_results)}")
if isinstance(base_results, dict):
    print(f"  Keys: {list(base_results.keys())}")
    for key in ['X', 'F', 'objectives', 'decisions']:
        if key in base_results:
            val = base_results[key]
            if hasattr(val, 'shape'):
                print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"    {key}: type={type(val)}")
else:
    print(f"  Attributes: {[attr for attr in dir(base_results) if not attr.startswith('_')][:10]}")
    if hasattr(base_results, 'X'):
        print(f"    X: shape={base_results.X.shape}, dtype={base_results.X.dtype}")
    if hasattr(base_results, 'F'):
        print(f"    F: shape={base_results.F.shape}, dtype={base_results.F.dtype}")

# Extract base X and F
if isinstance(base_results, dict):
    base_X = base_results.get('X') or base_results.get('decisions')
    base_F = base_results.get('F') or base_results.get('objectives')
else:
    base_X = getattr(base_results, 'X', None)
    base_F = getattr(base_results, 'F', None)

print(f"\nExtracted base_X shape: {base_X.shape if base_X is not None else 'None'}")
print(f"Extracted base_F shape: {base_F.shape if base_F is not None else 'None'}\n")

# Now check a run result
run_pkl_files = sorted(glob.glob(os.path.join('multi_seed_results', 'IDMRF*_PS*_SE*_PA', 'results_files', 'results_fg_*.pkl')))
print(f"Run pkl files found: {len(run_pkl_files)}")

if run_pkl_files:
    run_pkl = run_pkl_files[0]
    print(f"\nLoading first run: {run_pkl}")
    
    run_results = safe_pickle_load(run_pkl)
    
    print("RUN RESULTS STRUCTURE:")
    print(f"  Type: {type(run_results)}")
    if isinstance(run_results, dict):
        print(f"  Keys: {list(run_results.keys())}")
        for key in ['X', 'F', 'objectives', 'decisions']:
            if key in run_results:
                val = run_results[key]
                if hasattr(val, 'shape'):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"    {key}: type={type(val)}")
    else:
        if hasattr(run_results, 'X'):
            print(f"    X: shape={run_results.X.shape}, dtype={run_results.X.dtype}")
        if hasattr(run_results, 'F'):
            print(f"    F: shape={run_results.F.shape}, dtype={run_results.F.dtype}")
    
    # Extract run X and F
    if isinstance(run_results, dict):
        run_X = run_results.get('X') or run_results.get('decisions')
        run_F = run_results.get('F') or run_results.get('objectives')
    else:
        run_X = getattr(run_results, 'X', None)
        run_F = getattr(run_results, 'F', None)
    
    print(f"\nExtracted run_X shape: {run_X.shape if run_X is not None else 'None'}")
    print(f"Extracted run_F shape: {run_F.shape if run_F is not None else 'None'}")
    
    # Now test spatial agreement computation
    print("\n" + "="*60)
    print("TESTING SPATIAL AGREEMENT COMPUTATION:")
    print("="*60)
    
    if base_X is not None and run_X is not None:
        print(f"base_X shape: {base_X.shape}")
        print(f"run_X shape: {run_X.shape}")
        print(f"base_X dtype: {base_X.dtype}")
        print(f"run_X dtype: {run_X.dtype}")
        print(f"base_X dtype: {base_X.dtype}")
        print(f"base_X min/max: {base_X.min():.4f}/{base_X.max():.4f}")
        print(f"run_X min/max: {run_X.min():.4f}/{run_X.max():.4f}")
        
        # Binarize
        base_bin = (base_X > 0.5).astype(np.uint8)
        run_bin = (run_X > 0.5).astype(np.uint8)
        
        print(f"\nAfter binarization:")
        print(f"base_bin shape: {base_bin.shape}")
        print(f"run_bin shape: {run_bin.shape}")
        print(f"base_bin sum per row (min/max): {base_bin.sum(axis=1).min()}/{base_bin.sum(axis=1).max()}")
        print(f"run_bin sum per row (min/max): {run_bin.sum(axis=1).min()}/{run_bin.sum(axis=1).max()}")
        
        # Check if dimensions match
        if base_bin.shape[1] != run_bin.shape[1]:
            print(f"\n⚠️  DIMENSION MISMATCH! ncols differ: {base_bin.shape[1]} vs {run_bin.shape[1]}")
        else:
            print(f"\n✓ Dimensions match (both have {base_bin.shape[1]} columns)")
            
            # Compute Jaccard similarity
            jaccard_sim = 1.0 - cdist(base_bin, run_bin, metric='jaccard')
            jaccard_sim = np.nan_to_num(jaccard_sim, nan=0.0)
            
            print(f"\nJaccard similarity matrix shape: {jaccard_sim.shape}")
            print(f"Jaccard similarity min/max/mean: {jaccard_sim.min():.4f}/{jaccard_sim.max():.4f}/{jaccard_sim.mean():.4f}")
            
            # Compute symmetric agreement
            best_base_to_run = np.mean(np.max(jaccard_sim, axis=1))
            best_run_to_base = np.mean(np.max(jaccard_sim, axis=0))
            symmetric_agreement = 0.5 * (best_base_to_run + best_run_to_base)
            
            print(f"\nSymmetric spatial agreement: {symmetric_agreement:.4f}")
            print(f"  best_base_to_run: {best_base_to_run:.4f}")
            print(f"  best_run_to_base: {best_run_to_base:.4f}")
