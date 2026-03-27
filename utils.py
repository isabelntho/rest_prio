"""
Shared utilities for the restoration prioritisation pipeline.
"""

import pickle
import sys
import numpy as np


# =============================================================================
# PICKLE LOADING
# =============================================================================

class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that handles two common compatibility issues:

    1. ``HVCallback`` (and any other class) that is not present in the current
       environment — replaced with a harmless dummy so that result files saved
       with older environments load cleanly.
    2. ``numpy._core`` vs ``numpy.core`` module path changes between numpy
       builds — the old path is remapped to the current path on the fly.
    """

    def find_class(self, module, name):
        # Remap old numpy internal paths to the current layout.
        if module in ("numpy._core", "numpy._core.multiarray"):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                fixed_module = module.replace("numpy._core", "numpy.core")
                try:
                    return super().find_class(fixed_module, name)
                except (ModuleNotFoundError, AttributeError):
                    return getattr(np, name, None) or type(name, (), {})

        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            # Unknown class: return a dummy so the rest of the pickle loads.
            return type(name, (), {})


def pickle_load(path):
    """Load a pickle file with compatibility handling for missing classes and
    numpy version differences.

    Parameters
    ----------
    path : str or path-like
        Path to the ``.pkl`` file.

    Returns
    -------
    object
        The unpickled object.
    """
    # First attempt: standard load with the safe unpickler.
    try:
        with open(path, "rb") as f:
            return _SafeUnpickler(f).load()
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise

    # Second attempt: patch the sys.modules alias and retry (handles edge cases
    # where the module reference is checked before find_class is called).
    sys.modules.setdefault("numpy._core", np.core)
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()
