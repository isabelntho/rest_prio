"""
Diagnosis: Check exact X shapes for base and experiment runs
"""
import os
import pickle
import numpy as np
import warnings

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'HVCallback':
            # Try to import HVCallback from visualisations
            try:
                from visualisations import HVCallback
                return HVCallback
            except:
                class DummyHVCallback:
                    def __init__(self, *args, **kwargs):
                        pass
                return DummyHVCallback
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            warnings.warn(f'Could not import {module}.{name}, using dummy class: {e}')
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
            return DummyClass

def safe_pickle_load(file_path):
    '''Load pickle file with graceful handling of missing classes'''
    with open(file_path, 'rb') as f:
        return SafeUnpickler(f).load()

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
        raise ValueError("Could not extract objective and decision arrays")
    
    return F, X

# Load base
base_path = 'results_files/results_fg_2303_1.pkl'
print(f"Loading base: {base_path}")
try:
    base_results = safe_pickle_load(base_path)
    base_F, base_X = extract_objectives_and_decisions(base_results)
    print(f"  ✓ base_X shape: {base_X.shape}")
    print(f"  ✓ base_F shape: {base_F.shape}")
    print(f"  ✓ base_X dtype: {base_X.dtype}")
    print(f"  ✓ base_X sample values: {base_X[0, :10]}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load experiment runs
import glob

print("\nChecking experiment runs:")
exp_pattern = os.path.join('multi_seed_results', 'IDMRF*_PS*_SE*_PA')
exp_dirs = sorted(glob.glob(exp_pattern))
print(f"Found {len(exp_dirs)} experiment directories\n")

mismatches = []
for exp_dir in exp_dirs[:3]:  # Check first 3
    exp_id = os.path.basename(exp_dir)
    run_pkl_pattern = os.path.join(exp_dir, 'results_files', f'results_fg_*.pkl')
    run_pkl_files = sorted(glob.glob(run_pkl_pattern))
    
    if not run_pkl_files:
        print(f"{exp_id}: No pickle files found!")
        continue
    
    run_pkl = run_pkl_files[0]
    
    try:
        run_results = safe_pickle_load(run_pkl)
        run_F, run_X = extract_objectives_and_decisions(run_results)
        print(f"{exp_id}:")
        print(f"  run_X shape: {run_X.shape}")
        print(f"  run_F shape: {run_F.shape}")
        print(f"  run_X dtype: {run_X.dtype}")
        print(f"  run_X sample values: {run_X[0, :10]}")
        
        if base_X.shape[1] != run_X.shape[1]:
            print(f"  ⚠️  COLUMN MISMATCH: base has {base_X.shape[1]} cols, run has {run_X.shape[1]} cols")
            mismatches.append((exp_id, base_X.shape[1], run_X.shape[1]))
        else:
            print(f"  ✓ Column count matches")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print(f"Summary: {'Column mismatch confirmed!' if mismatches else 'Column counts match'}")
if mismatches:
    for exp_id, base_cols, run_cols in mismatches:
        print(f"  {exp_id}: base={base_cols}, run={run_cols}")
