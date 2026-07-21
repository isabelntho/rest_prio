"""
Canonical output locations for the restoration optimisation pipeline.

Single source of truth for where the pipeline writes results, logs, reports, and
R-export inputs. Import the constants from here rather than hardcoding directory-name
strings, so relocating the output tree is a one-line change in this file.

All pipeline outputs live under OUTPUTS (currently <project_root>/outputs). The R
analysis layer reads r_inputs/, results_files/, and figs/ from this same tree; if you
move OUTPUTS, update the corresponding paths in Documentation/*.R and Documentation/*.qmd.
"""

from pathlib import Path

# Repo root = parent of the Core_optimisation package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root of the consolidated input-data tree (rasters, anomaly scenarios, prep scripts).
DATA_DIR = PROJECT_ROOT / "data"

# Root of the consolidated output tree.
OUTPUTS = PROJECT_ROOT / "outputs"

# Base passed as `output_dir` to the save/report functions. Those functions create
# results_files/, summary_files/, optimisation_reports/, evolution_reports/, r_inputs/,
# intermediate_results/, and run_registry.jsonl *inside* this base.
OUTPUT_DIR = OUTPUTS

# Directories not created relative to `output_dir`, so they need explicit paths.
LOGS_DIR = OUTPUTS / "logs"
MULTISEED_DIR = OUTPUTS / "multi_seed_results"
PATCH_TESTS_DIR = OUTPUTS / "patch_tests"
FIGS_DIR = OUTPUTS / "figs"

# Convenience sub-locations under OUTPUT_DIR, for callers that build paths directly.
RESULTS_DIR = OUTPUTS / "results_files"
SUMMARY_DIR = OUTPUTS / "summary_files"
OPTIMISATION_REPORTS_DIR = OUTPUTS / "optimisation_reports"
EVOLUTION_REPORTS_DIR = OUTPUTS / "evolution_reports"
INTERMEDIATE_DIR = OUTPUTS / "intermediate_results"
R_INPUTS_DIR = OUTPUTS / "r_inputs"


def ensure(path):
    """Create `path` (and any missing parents) if absent; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
