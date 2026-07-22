#!/usr/bin/env python
"""Debug script to diagnose spatial agreement NA issue."""

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Add safe unpickler
import pickle
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'HVCallback':
            return lambda *args, **kwargs: None
        return super().find_class(module, name)

def safe_pickle_load(filepath):
    with open(filepath, 'rb') as f:
        return SafeUnpickler(f).load()

# Replicate the extraction logic
def extract_objectives_and_decisions(results_obj):
    """Extract objective (F) and decision (X) matrices from a results object."""
    F = None
    X = None

    if isinstance(results_obj, dict):
        if 'objectives' in results_obj and results_obj['objectives'] is not None:
            F = np.asarray(results_obj['objectives'], dtype=float)
        elif 'F' in results_obj and results_obj['F'] is not None:
            F = np.asarray(results_obj['F'], dtype=float)

        if 'decisions' in results_obj and results_obj['decisions'] is not None:
            X = np.asarray(results_obj['decisions'])
        elif 'X' in results_obj and results_obj['X'] is not None:
            X = np.asarray(results_obj['X'])
    else:
        if hasattr(results_obj, 'F') and results_obj.F is not None:
            F = np.asarray(results_obj.F, dtype=float)
        if hasattr(results_obj, 'X') and results_obj.X is not None:
            X = np.asarray(results_obj.X)

    if F is None or X is None:
        raise ValueError("Could not extract objective and decision arrays from results object")

    if F.ndim != 2 or X.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got F:{getattr(F, 'shape', None)}, X:{getattr(X, 'shape', None)}")

    if F.shape[0] != X.shape[0]:
        raise ValueError(f"Mismatch in number of solutions, F:{F.shape[0]}, X:{X.shape[0]}")

    return F, X

# Load base
base_file = os.path.join('results_files', 'results_fg_2003_1.pkl')
print(f"Loading base from: {base_file}")
base_results = safe_pickle_load(base_file)
base_F, base_X = extract_objectives_and_decisions(base_results)
print(f"Base: X shape={base_X.shape}, F shape={base_F.shape}")

# Try loading one run
run_files = sorted(glob.glob(os.path.join('multi_seed_results', 'IDMRF*_PS*_SE*_PA', 'results_files', 'results_fg_*.pkl')))
print(f"\nFound {len(run_files)} run files")

if len(run_files) > 0:
    run_file = run_files[0]
    print(f"\nLoading first run from: {run_file}")
    
    try:
        run_results = safe_pickle_load(run_file)
        run_F, run_X = extract_objectives_and_decisions(run_results)
        print(f"Run: X shape={run_X.shape}, F shape={run_F.shape}")
        
        # Test spatial agreement computation
        print("\nTesting spatial agreement computation:")
        print(f"  base_X shape: {base_X.shape}")
        print(f"  run_X shape: {run_X.shape}")
        print(f"  Shapes match: {base_X.shape[1] == run_X.shape[1]}")
        
        # Try binarization
        base_bin = (base_X > 0.5).astype(np.uint8)
        run_bin = (run_X > 0.5).astype(np.uint8)
        
        print(f"  base_bin shape: {base_bin.shape}")
        print(f"  run_bin shape: {run_bin.shape}")
        
        # Compute Jaccard
        if base_bin.shape[1] == run_bin.shape[1]:
            jaccard_sim = 1.0 - cdist(base_bin, run_bin, metric='jaccard')
            jaccard_sim = np.nan_to_num(jaccard_sim, nan=0.0)
            
            best_base_to_run = np.mean(np.max(jaccard_sim, axis=1))
            best_run_to_base = np.mean(np.max(jaccard_sim, axis=0))
            symmetric_agreement = 0.5 * (best_base_to_run + best_run_to_base)
            
            print(f"  Jaccard similarity computed successfully")
            print(f"  Symmetric agreement: {symmetric_agreement:.4f}")
        else:
            print(f"  ERROR: Dimension mismatch! {base_bin.shape[1]} vs {run_bin.shape[1]}")
            
    except Exception as e:
        print(f"ERROR loading or processing run: {e}")
        import traceback
        traceback.print_exc()
