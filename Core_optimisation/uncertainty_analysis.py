"""Robustness of restoration plans to the choices made when building the objective.

Three formulation choices are now searched independently, each replicated over seeds:

  weighting     which indicator gets more weight in the condition composite
                (campaign 20260817_1654_weighting_vertex_grid)
  scaling       how ambitious the benchmark the indicators are z-scored against is
  construction  which indicator is left out entirely (leave-one-out)
                (both from campaign 20260821_0821_benchmark_x_construction_grid, which
                 crosses them fully: 4 scalings x 12 constructions x 5 seeds)

Each cell is normally read only on its own terms. This module answers the cross-cutting
questions the grids were built for: which axis actually moves plans relative to search
stochasticity, whether the leave-one-out sensitivity depends on how ambitious the
benchmark is, and which plans stay good no matter which of these choices you believe.

Doing that needs an evaluator decoupled from the search, so a plan found under variant A
can be scored against variant B's condition rasters. Three facts make it cheap and exact:

  1. Benefit is exactly decomposable (resto_anom.restoration_effect):
         benefit(S) = sum_{c in S} d[c] + decay * sum_{c in dilate(S)\\S} d[c]
         d[c]       = ab_eff*w(ab0[c]) + bi_eff*w(bi0[c]), zeroed outside the eligible mask
     so a variant is fully summarised by ONE raster d_v, and the dilation depends only on
     the plan, not the variant: one dilation per plan, then a gather per variant.
  2. Cost is variant-invariant - every condition tag loads the same cost raster. This is
     ASSERTED at extract (every run's cost raster is hashed), not assumed.
  3. Eligible sets DIFFER across tags (437110 for every weighting tag and for *_all, but
     436965 / 436984 / 437056 for the drop_* constructions), so `decisions` vectors are
     NOT index-aligned across variants. Plans are therefore carried as FLAT RASTER
     INDICES throughout, never as 1-D eligible indices.

Stages (pixi run python -m Core_optimisation.uncertainty_analysis <stage>):

  extract   Stream the pkls named by CAMPAIGNS and cache what the rest of the module
            needs. RESUMABLE: one shard per run under outputs/uncertainty/shards/plans/
            and one per condition tag under shards/layers/, both skipped when already
            present and newer than their pkl, so re-running after adding a campaign
            costs only the new runs. Then pools the plan shards into archive.npz.
            Expensive (the pkls are ~459 MB each); everything downstream reads the caches.
            `extract --dry-run` does discovery and the design-balance check only.

  layers    THE VALIDATION GATE. Re-checks the fast evaluator against the real
            RestorationProblem.evaluate_raw_objectives and against stored objectives_raw,
            on a tag set STRATIFIED over the three axes, plus degenerate-input checks.
            Nothing downstream is meaningful until it passes.

  cross     Score every pooled plan under every variant -> the plan x variant benefit
            matrix (cross_raw.npz, cross_matrix.csv). This is the expensive shared
            computation: `discrim`, `noise`, `interact` and `classify` all read its
            matrix rather than recomputing it, which is why it now runs before them.

  discrim   THE SATURATION GUARD - read this before any robustness number. Per variant,
            how far apart the objective can hold two plans that are actually on the table
            (the archive's front span). A variant that cannot separate plans reads as
            perfectly robust for the wrong reason, and sum-form benefit does flatten
            toward high benchmark quantiles - the span falls from 0.74 at `global` to
            0.21 at `upper_q75`. Flagged tags are excluded from the worst case in
            `classify` (EXCLUDE_SATURATED). Also reports a scattered-random comparator,
            which outscores the optimised archive under 57 of 61 variants: that is the
            price of the contiguity constraint, not a discriminability result.

  noise     Noise floor, per axis: within-cell (seed) regret against each axis's
            across-level regret, and the same split for spatial (Jaccard) agreement,
            ranked so the axes can be compared with each other. Read the verdict BEFORE
            any map. Writes noise_by_axis.csv plus the distributions in long form.

  interact  Only on the crossed benchmark grid. Does the leave-one-out sensitivity
            depend on the benchmark quantile? Balanced two-way decomposition of a
            per-run displacement response over scaling x construction with seed as the
            replicate term -> interaction.csv.

  classify  Cost-matched benefit regret against the reference chosen by REFERENCE_MODE,
            worst-case ranking over the pooled uncertainty set, robust subset = bottom
            quartile of max regret -> plan_summary.csv. NOTE: the budget is a hard
            equality constraint on pixel COUNT, so each variant's own front spans ~1% of
            the cost axis and the fronts are nearly disjoint; "vs that variant's own
            best" is therefore ill-posed on these runs and the pooled envelope is the
            default. See REFERENCE_MODE.

  maps      Selection frequency, full archive vs robust subset, and the difference map.
            The gap between those two maps is the result. Writes a PNG (the analyst-facing
            diagnostic) plus freq_full / freq_robust / delta_frequency GeoTIFFs, which are
            what the manuscript figure is redrawn from.

  all       extract -> layers -> cross -> discrim -> noise -> interact -> classify -> maps.

Every stage also merges the headline numbers it prints into outputs/uncertainty/report.json
(see `report_update`), each under its own key. That file is the manuscript's data source -
paper2/_results_uncertainty.qmd reads it rather than re-deriving any statistic in R - so
re-running a stage is what updates the paper. Nothing in the report is computed twice: the
values recorded are the ones the stage already printed.

Configuration is the module-level block below (the Core_optimisation driver convention);
the stage name is the only required command-line argument.
"""
# --- cap nested numpy/BLAS threading BEFORE importing numpy-heavy modules ----
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import sys
import json
import time
import hashlib
import itertools
import numpy as np
import pandas as pd
from scipy import ndimage

from Core_optimisation.paths import (
    PROJECT_ROOT, OUTPUTS, RESULTS_DIR, R_INPUTS_DIR, FIGS_DIR, ensure,
)

# `utils` and `visualisations` live at the repo root, which is on sys.path when this is
# run as `python -m Core_optimisation.uncertainty_analysis` from there; make that explicit
# so the module also imports cleanly from elsewhere. utils.pickle_load absorbs the numpy
# version differences across the age of the pkl archive.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import pickle_load

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ===========================================================================
# configuration
# ===========================================================================
# Formulation space is described one CAMPAIGN at a time. A campaign is a grid of runs
# sharing a label grammar and a design; `axes` names the factors it varies and
# `reference` the level of each axis that counts as "no perturbation".
#
# AXIS_NAMES is the union over campaigns, so the pooled archive carries one column per
# axis with NOT_APPLICABLE where a campaign does not vary that factor. Per-axis
# statistics are campaign-scoped by construction - comparing levels of one axis needs
# the others held at their reference, which only exists inside the campaign that varies
# them - while `cross`, `classify` and `maps` work on the pooled archive across all of
# them.
#
# `objective_construction` separates two structurally different objectives that must NOT
# be read as two levels of one axis:
#   "two_block"        the usual separate abiotic and biotic composites, so the objective
#                      is eff*w(A) + eff*w(B) with a hardcoded 1:1 block balance.
#   "single_composite" the weighting family (data/ec_anomalies.r) writes ONE
#                      all-indicator weighted composite to BOTH rasters, so the objective
#                      is 2*eff*w(C) and the block balance is carried by the weight
#                      vector instead. anomaly_improvement_weight is nonlinear, so
#                      global_all ("two_block") and global_w_cat ("single_composite") are
#                      NOT the same scenario - their stored benefit normalisation scales
#                      are 437166 and 284345. global_w_cat, not global_all, is therefore
#                      the weighting family's own reference level.
#
# RETIRED: "20260811_1648_region_evolve_condgrid". Its 50 runs predate the 2026-08-12
# region_evolve grow-move fix (spatial_operations.py, the 'grow' branch that added
# c.size + n_edits and so roughly doubled every grown region), and its design is a strict
# subset of the benchmark grid below. Re-add it only with that caveat attached.
NOT_APPLICABLE = "n/a"

CAMPAIGNS = {
    "weighting": {
        # ONE weighting axis, run in two batches and pooled. Both batches are the same
        # kind of perturbation - one all-indicator weighted composite written to both
        # rasters - and differ only in HOW FAR from equal weighting their levels sit,
        # so they are levels of one axis rather than two axes:
        #
        #   structured  w_flat (1/n each), w_cat (equal abiotic/biotic blocks) and
        #               w_<k> (indicator k doubled to 2/n, the rest shrunk
        #               proportionally). Departures from flat of 0 to 0.150.
        #   sampled     w_d{BB}{a-d}, explicit weight vectors drawn uniformly and kept
        #               by rejection at departures 0.20 / 0.35 / 0.50, four per band.
        #
        # Pooling them is what makes the axis reportable as a DOSE-RESPONSE. Split into
        # two campaigns each would get one regret ratio, and that ratio is an artefact
        # of how many near vs far levels the split happened to contain - add near levels
        # and it falls, add far ones and it rises. Displacement as a function of
        # departure has no such dependence, and the structured levels supply the 0-0.15
        # coverage the sampled ones lack.
        #
        # global_w_flat was run in BOTH batches. Discovery keys on (campaign, tag, seed),
        # so the later folder's copy wins - list the newest LAST. That is deliberate: the
        # reference cell defines the seed floor, and it should be measured in the same
        # batch as the levels that dominate the far end of the curve.
        "source": ("r_inputs", ["20260817_1654_weighting_vertex_grid",
                                "20260826_1732_weighting_simplex_grid"]),
        "label_re": (r"^(?P<weighting>global_w_(?:[a-z]+|[a-z_]+|d\d{2}[a-d]))"
                     r"_seed(?P<seed>\d+)$"),
        "axes": ("weighting",),
        # Flat, not w_cat: it is the origin the departure measure is defined from, and
        # the centre of the distribution the sampled levels are drawn around.
        "reference": {"weighting": "global_w_flat"},
        "objective_construction": "single_composite",
        # Levels the design must contain, for the balance check.
        "expect": {"weighting": ["global_w_flat", "global_w_cat"]
                                + [f"global_w_{k}" for k in
                                   ("smd", "sbd", "soc", "uzl", "cdi", "swf_h",
                                    "swf_t", "ndvi", "tsd", "can", "lai")]
                                + [f"global_w_d{b}{d}"
                                   for b in ("20", "35", "50") for d in "abcd"]},
    },
    "benchmark_x_construction": {
        # Fully crossed scaling x construction. The condition raster tag is
        # f"{scaling}_{construction}", which is what run_config['condition_scenario']
        # carries and what selects the raster pair.
        "source": ("registry", r"^form-sum__scal-.+__pol-status_quo__seed\d+$"),
        # Every field is lazy up to the "__" separator, NOT [^_]+: the level names
        # themselves carry single underscores (upper_q25, drop_swf_h, status_quo).
        "label_re": (r"^form-(?P<form>.+?)__scal-(?P<scaling>.+?)__"
                     r"con-(?P<construction>.+?)__pol-(?P<policy>.+?)__seed(?P<seed>\d+)$"),
        "axes": ("scaling", "construction"),
        "reference": {"scaling": "global", "construction": "all"},
        "objective_construction": "two_block",
        "expect": {
            "scaling": ["global", "upper_q25", "upper_q50", "upper_q75", "zones"],
            "construction": ["all", "drop_smd", "drop_sbd", "drop_soc", "drop_uzl",
                             "drop_cdi", "drop_swf_h", "drop_swf_t", "drop_ndvi",
                             "drop_tsd", "drop_can", "drop_lai"],
        },
        # Levels whose runs are NOT yet replicated over the full SEEDS list.
        #
        # The distinction that makes a partial level safe to carry at all: a variant is
        # used in two ways, and only one of them depends on how many seeds ran.
        #   as a LENS   - a d_v layer that pooled plans are SCORED UNDER (cross, discrim,
        #                 classify). Built from the condition rasters alone, so it is
        #                 seed-independent. Held-out levels take part here in full.
        #   as a SOURCE - the plans that level's own runs contributed to the pooled
        #                 archive (noise's per-axis distributions, interact, and the
        #                 composition-by-axis shares). Pooling 2 seeds against 5 would
        #                 bias every one of those. Held-out levels are kept out.
        #
        # So a held-out level is extracted, cached and validated, and appears in the
        # design table, while every reported number stays exactly what it would be if the
        # level did not exist. Verified: with this set, the pooled archive, the noise
        # axis rows and the interact decomposition are all bit-identical to a run with no
        # zones runs on disk at all.
        #
        # The value is the seeds ADMITTED, pinned rather than open-ended so the numbers
        # stay reproducible while the campaign is still running - without it, re-running
        # `extract` mid-campaign would fold in whichever cells happened to have finished
        # by then. List only COMPLETE seeds: a seed part-way through the construction
        # levels would make the level internally ragged.
        #
        # To let the level into the reported results, delete the two `is_holdout_run`
        # guards (in `_pool_shards` and `cmd_noise._one_axis_apart`) so its plans are
        # pooled, and keep `balanced_levels` on check_design and cmd_interact - the
        # two-way decomposition is a balanced-design formula and cannot take an
        # under-replicated level whatever else changes.
        "holdout": {"scaling": {"zones": [101, 102]}},
    },
}

# Union of every campaign's axes, in a stable order. Archive columns follow this.
AXIS_NAMES = tuple(dict.fromkeys(a for c in CAMPAIGNS.values() for a in c["axes"]))

SEEDS = [101, 102, 103, 104, 105]

RUN_OBJECTIVES = ["restoration_benefit", "implementation_cost"]
CONSISTENT_PARAMS = ("max_restoration_fraction", "min_patch_size", "sampling_strategy")

# Cap on non-dominated plans kept per run (None = keep all). The pooled archive is
# ~9k plans of ~21.8k int32 indices, so raising this is a memory decision.
MAX_PLANS_PER_RUN = None

# Robust subset = this bottom quantile of worst-case regret.
ROBUST_QUANTILE = 0.25

# Which reference front a plan's cost-matched benefit regret is measured against.
#
#   "own"     each variant's own pooled front (its own seeds' runs, scored under itself).
#             This is the natural reading of "deviation vs that variant's own best", but
#             it is NOT usable on these runs: the budget is a hard equality constraint on
#             pixel COUNT, so each variant's search converges to a narrow, variant-specific
#             cost level and its front spans ~1% of the cost axis. The fronts are nearly
#             disjoint, so most plans fall off the bottom of most references and would
#             collect an artificial zero regret. `cross` prints the coverage per variant.
#   "pooled"  the non-dominated envelope of EVERY pooled plan scored under v - "the best
#             anyone found under this variant". Spans the whole cost range by construction,
#             so the regret is defined for every plan x variant pair. Default.
#
# Both are computed and stored by `cross`; this only picks which one drives `classify`.
REFERENCE_MODE = "pooled"
# A cell belongs to a group's "core" if this fraction of its plans select it.
#
# 0.50 ("a majority of these plans put restoration here"), not the 0.90 used when the
# archive held one campaign. At 0.90 the pooled archive has an EMPTY core - the highest
# selection frequency any cell reaches across all 61 variants is 0.72 - so every
# comparison built on it was a Jaccard of near-empty sets, which is what produced the
# uninterpretable 0.00 medians. The budget puts only ~5% of the eligible landscape in any
# one plan, so 0.50 is already a strong consensus requirement: `maps` prints the frequency
# percentiles and the counts at 0.50 / 0.75 / 0.90 so this choice stays visible.
CORE_FREQ = 0.50

# -- discriminability guard (stage `discrim`) -------------------------------
# The saturation criterion is the FRONT SPAN: (worst - best) / |best| over the pooled
# archive under that variant, i.e. how far apart the objective can hold two plans that
# are actually on the table. Both ends are real region-grown plans, so the measure is
# contiguity-matched by construction.
#
# It is NOT the best-plan-vs-random-plan ratio. That was tried and is confounded: the
# random comparator is a scattered selection, spillover credit accrues to un-restored
# neighbours, and a scattered plan therefore has a far larger spillover ring than a
# contiguous plan of the same size. The measured ratio came out BELOW 1 for 60 of 61
# variants - random scattered plans outscore the optimised contiguous ones - so it
# measures the price of contiguity, not whether the objective discriminates. It is still
# computed and reported, because that price is itself worth knowing.
DISCRIM_MIN_SPAN = 0.10
DISCRIM_N_RANDOM = 50
DISCRIM_RANDOM_SEED = 20260824
# Drop saturated variants from the worst-case max in `classify`.
EXCLUDE_SATURATED = True

# -- validation gate (stage `layers`) ---------------------------------------
VALIDATE_N_PLANS = 24
# Tags checked against the real engine, one drawn per pattern so all three axes and both
# objective constructions are exercised rather than the first few tags alphabetically.
# `^global_w_d\d` is listed separately from `^global_w_`: the latter matches a weighting
# VERTEX first (alphabetically global_w_can), so without its own pattern the extended
# simplex campaign would never be exercised against the engine despite being a third of
# the variant layers.
VALIDATE_TAG_PATTERNS = (r"^global_drop_", r"^upper_q\d+_drop_", r"^upper_q\d+_all$",
                         r"^global_w_[a-z]", r"^global_w_d\d",
                         # `zones_` has its own pattern for the same reason the simplex
                         # campaign does: without it the benchmark patterns above never
                         # draw a zone-wise tag. The evaluator path is the same whatever
                         # built the rasters, so this is checking the RASTERS - that the
                         # zone-wise pair loads and scores like any other.
                         r"^zones_")
# Relative tolerance for benefit against the engine. The floor is set by the ENGINE, not
# by this module: restoration_effect writes the improved anomaly back into a float32
# raster (`updated_values[action_mask] = baseline + improvement`, resto_anom.py:462), so
# each cell's improvement is stored with ~1e-7 relative precision and a 21855-cell sum
# lands ~2e-6 away from an exact float64 accumulation. Measured across sampled plans of
# global_drop_can: 1.7e-6 to 2.1e-6, and identical whether the cached d_v is float32 or
# float64 - so this tolerance is the engine's precision, with ~5x headroom.
#
# It therefore does NOT police the float32 layer cache; check [8] below does that
# separately, at a tolerance the cache alone has to meet.
VALIDATE_RTOL_BENEFIT = 1e-5
# The float32 cache against a float64 rebuild of the same layer. Measured 2e-11 to 2e-10.
VALIDATE_RTOL_CACHE = 1e-8
# Cost is a float32 gather summed the same way on both paths, so it must agree to
# floating-point noise, not merely to a tolerance. A miss here means the flat-index
# mapping is wrong, not that precision was lost.
VALIDATE_RTOL_COST = 1e-9

OUT_DIR = OUTPUTS / "uncertainty"
FIG_DIR = FIGS_DIR / "uncertainty"
SHARD_DIR = OUT_DIR / "shards"
PLAN_SHARD_DIR = SHARD_DIR / "plans"
LAYER_SHARD_DIR = SHARD_DIR / "layers"
COMMON_NPZ = SHARD_DIR / "common.npz"
ARCHIVE_NPZ = OUT_DIR / "archive.npz"
PROVENANCE_JSON = OUT_DIR / "plan_provenance.json"
# Machine-readable copy of every headline number the stages print, for the manuscript
# (paper2/_results_uncertainty.qmd reads this instead of re-deriving statistics in R).
REPORT_JSON = OUT_DIR / "report.json"


# ===========================================================================
# formulation axes + run discovery
# ===========================================================================
def axes_of(run_label, camp):
    """`run_label` -> (axes dict over AXIS_NAMES, seed), per one campaign's grammar.

    Axes the campaign does not vary are NOT_APPLICABLE. Seed is a replicate, never an
    axis. Raises ValueError when the label does not match, which is how a foreign run
    in a shared folder gets skipped rather than mis-parsed.
    """
    m = re.match(camp["label_re"], str(run_label))
    if m is None:
        raise ValueError(f"run_label does not match this campaign's grammar: {run_label!r}")
    g = m.groupdict()
    axes = {a: g.get(a, NOT_APPLICABLE) or NOT_APPLICABLE for a in AXIS_NAMES}
    return axes, int(g["seed"])


def tag_from_axes(axes, camp):
    """The condition raster tag the axes imply, for cross-checking run_config.

    The factorial builds it as f"{scaling}_{construction}" (grid_parallel.run_factorial_cell);
    the weighting grid's single axis IS the tag.
    """
    if camp["axes"] == ("scaling", "construction"):
        return f"{axes['scaling']}_{axes['construction']}"
    if len(camp["axes"]) == 1:
        return axes[camp["axes"][0]]
    raise ValueError(f"no tag rule for axes {camp['axes']}")


def cell_key(axes, camp):
    """Hashable identity of a formulation cell, over the campaign's own axes only."""
    return tuple(axes[a] for a in camp["axes"])


def is_reference_cell(axes, camp, except_axis=None):
    """True when every axis except `except_axis` sits at its reference level."""
    return all(axes[a] == camp["reference"][a]
               for a in camp["axes"] if a != except_axis)


def holdout_levels(camp, axis):
    """{level: [admitted seeds]} held out of the pooled statistics for `axis`.

    See the "holdout" note in CAMPAIGNS: these levels take part as a LENS (scored under)
    but not as a SOURCE (their runs' plans are not pooled).
    """
    return camp.get("holdout", {}).get(axis, {})


def balanced_levels(camp, axis):
    """`expect` levels for `axis` minus any held out - the balanced core of the design.

    Everything that assumes equal seed depth across levels (check_design's balance test,
    interact's decomposition, noise's per-axis distributions) must read levels through
    this rather than off `expect` directly.
    """
    held = holdout_levels(camp, axis)
    return [lv for lv in camp["expect"][axis] if lv not in held]


def is_holdout_run(axes, camp):
    """True when this run belongs to a held-out level (so its PLANS are not pooled).

    A run whose level is held out but whose seed is not in the admitted list is dropped
    at discovery instead, so it never reaches the archive at all.
    """
    for axis, held in camp.get("holdout", {}).items():
        if axes.get(axis) in held:
            return True
    return False


def holdout_admits(axes, seed, camp):
    """False only for a run on a held-out level whose seed is outside the pinned list."""
    for axis, held in camp.get("holdout", {}).items():
        lv = axes.get(axis)
        if lv in held and int(seed) not in held[lv]:
            return False
    return True


def _gather_r_inputs(camp_name, camp, folder, seeds, out, skipped):
    """Discovery backend 1: outputs/r_inputs/<folder>/*/metadata.json.

    `folder` may be one name or a list of them, for a campaign whose levels were run in
    more than one batch. Folders are read IN ORDER and a repeated (tag, seed) keeps the
    LAST one seen, matching the registry backend's re-run convention - so list the batch
    whose copy should win last.
    """
    if not isinstance(folder, str):
        for one in folder:
            _gather_r_inputs(camp_name, camp, one, seeds, out, skipped)
        return
    base = R_INPUTS_DIR / folder
    if not base.is_dir():
        raise FileNotFoundError(
            f"CAMPAIGNS[{camp_name!r}] names r_inputs folder {folder!r}, which does not "
            f"exist under {R_INPUTS_DIR}. Available: "
            f"{', '.join(sorted(p.name for p in R_INPUTS_DIR.iterdir() if p.is_dir()))}")
    for sub in sorted(base.iterdir()):
        meta_path = sub / "metadata.json"
        if not sub.is_dir() or not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8", errors="replace") as f:
            meta = json.load(f)
        label = str(meta.get("run_label") or sub.name)
        try:
            axes, seed = axes_of(label, camp)
        except ValueError:
            skipped.append((label, camp_name, "label does not match the campaign grammar"))
            continue
        if seed not in seeds:
            continue
        # Same seed pin as the registry backend, so a hold-out means the same thing
        # whichever backend a campaign is discovered through.
        if not holdout_admits(axes, seed, camp):
            skipped.append((label, camp_name,
                            "held-out level, seed outside the pinned list"))
            continue
        if list(meta.get("objective_names") or []) != RUN_OBJECTIVES:
            skipped.append((label, camp_name, f"objectives {meta.get('objective_names')}"))
            continue
        pkl = RESULTS_DIR / str(meta.get("pkl_file") or "")
        if not pkl.exists():
            skipped.append((label, camp_name, f"pkl missing: {meta.get('pkl_file')}"))
            continue
        tag = str((meta.get("run_config") or {}).get("condition_scenario") or
                  tag_from_axes(axes, camp))
        out[(camp_name, tag, seed)] = {"pkl": str(pkl), "label": label, "axes": axes,
                                       "campaign": camp_name, "tag": tag, "seed": seed}


def _gather_registry(camp_name, camp, label_re, seeds, out, skipped):
    """Discovery backend 2: outputs/run_registry.jsonl.

    Needed because the 20260821 benchmark grid has no metadata.json: its R export died
    after objectives.csv (a non-ASCII print under a redirected cp1252 stdout, since
    fixed in export_to_r.py). The registry is the authoritative record of what ran and
    where its pkl is, so nothing has to be re-exported for this module to read it.
    """
    reg = OUTPUTS / "run_registry.jsonl"
    if not reg.exists():
        raise FileNotFoundError(f"{reg} missing - it is the discovery source for "
                                f"CAMPAIGNS[{camp_name!r}].")
    pat = re.compile(label_re)
    with open(reg, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            label = str(row.get("run_label") or "")
            if not pat.match(label):
                continue
            try:
                axes, seed = axes_of(label, camp)
            except ValueError:
                skipped.append((label, camp_name, "label does not match the campaign grammar"))
                continue
            if seed not in seeds:
                continue
            # Held-out levels admit only their pinned seeds. Without this a re-run while
            # the campaign is still going would pick up whichever cells had finished by
            # then, so the reported numbers would drift between renders.
            if not holdout_admits(axes, seed, camp):
                skipped.append((label, camp_name,
                                "held-out level, seed outside the pinned list"))
                continue
            if list(row.get("objectives") or []) != RUN_OBJECTIVES:
                skipped.append((label, camp_name, f"objectives {row.get('objectives')}"))
                continue
            pkl = str((row.get("files") or {}).get("pickle_file") or "")
            if not pkl or not os.path.exists(pkl):
                skipped.append((label, camp_name, f"pkl missing: {pkl}"))
                continue
            tag = tag_from_axes(axes, camp)
            # A repeated label (re-run) keeps the LAST registry entry, which is the
            # newest pkl - the registry is append-only.
            out[(camp_name, tag, seed)] = {"pkl": pkl, "label": label, "axes": axes,
                                           "campaign": camp_name, "tag": tag, "seed": seed}


def gather_runs(campaigns=None, seeds=None):
    """{(campaign, tag, seed): run dict} over every configured campaign, + 'skipped'."""
    seeds = set(SEEDS if seeds is None else seeds)
    names = list(CAMPAIGNS if campaigns is None else campaigns)
    out, skipped = {}, []
    for camp_name in names:
        camp = CAMPAIGNS[camp_name]
        kind, arg = camp["source"]
        if kind == "r_inputs":
            _gather_r_inputs(camp_name, camp, arg, seeds, out, skipped)
        elif kind == "registry":
            _gather_registry(camp_name, camp, arg, seeds, out, skipped)
        else:
            raise ValueError(f"CAMPAIGNS[{camp_name!r}]['source'] kind {kind!r} unknown "
                             "- use 'r_inputs' or 'registry'.")
    out["skipped"] = skipped
    return out


def check_design(found, verbose=True):
    """Per-campaign coverage table + balance check.

    The two-way decomposition in `interact` is a BALANCED-design formula, so an
    incomplete grid has to be visible here rather than silently unbalancing that test.
    Returns {campaign: {"complete": bool, "missing": [...], "n_found": int, ...}}.
    """
    report = {}
    for camp_name, camp in CAMPAIGNS.items():
        runs = {k: v for k, v in found.items()
                if isinstance(k, tuple) and k[0] == camp_name}
        got = {(cell_key(v["axes"], camp), v["seed"]) for v in runs.values()}
        # Balance is judged on the BALANCED CORE only. A held-out level is present on
        # purpose and under-replicated on purpose, so counting its absent seeds as
        # "missing" would report a complete design as broken - and counting its cells as
        # "unexpected" would read as an accident rather than a decision. It gets its own
        # line below instead.
        levels = [balanced_levels(camp, a) for a in camp["axes"]]
        want = {(cells, s) for cells in itertools.product(*levels) for s in SEEDS}
        held_got = sorted(c for c in got
                          if is_holdout_run(dict(zip(camp["axes"], c[0])), camp))
        core_got = got - set(held_got)
        missing = sorted(want - core_got)
        extra = sorted(core_got - want)
        n_expected = len(want)
        report[camp_name] = {
            "axes": list(camp["axes"]), "n_found": len(runs), "n_expected": n_expected,
            "complete": not missing,
            "missing": [[list(c), int(s)] for c, s in missing],
            "unexpected": [[list(c), int(s)] for c, s in extra],
            "held_out": [[list(c), int(s)] for c, s in held_got],
            "holdout_levels": {a: sorted(h) for a, h in camp.get("holdout", {}).items()},
            "objective_construction": camp["objective_construction"],
        }
        if not verbose:
            continue
        print(f"\n{camp_name}  [{' x '.join(camp['axes'])} x seed]  "
              f"{camp['objective_construction']}")
        # One row per cell, one column per seed.
        for cells in itertools.product(*levels):
            marks = "".join("   x" if (cells, s) in got else "   ." for s in SEEDS)
            print(f"  {'|'.join(cells):<34}{marks}")
        # Count the balanced core against the core's own expectation - held-out runs are
        # reported on their own line below, so folding them in here would read as a
        # surplus against a number they were never part of.
        print(f"  {'TOTAL':<34}{len(core_got):>4} of {n_expected}"
              f"{'' if not missing else f'   {len(missing)} MISSING'}")
        # Under-replicated levels, named so a partial level is never silently invisible.
        for axis, held in camp.get("holdout", {}).items():
            for lv, admitted in sorted(held.items()):
                n_here = sum(1 for c, _ in held_got if lv in c)
                print(f"  held out of pooled statistics: {axis}={lv} "
                      f"({n_here} run(s), seeds {','.join(map(str, admitted))}) - "
                      "scored as a variant, plans not pooled")
        if extra:
            print(f"  {len(extra)} cell(s) present but not in this campaign's `expect` "
                  "levels - they are still extracted:")
            for c, s in extra[:10]:
                print(f"    {'|'.join(c)} seed{s}")
    return report


def load_selected_runs():
    """{(campaign, tag, seed): run dict} chosen by the last `extract`."""
    p = OUT_DIR / "selected_runs.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the `extract` stage first.")
    with open(p, encoding="ascii") as f:
        doc = json.load(f)
    return {tuple([v["campaign"], v["tag"], int(v["seed"])]): v for v in doc.values()}


def _obj_indices(names):
    """Column indices of (restoration_benefit, cost) - the cost name varies by caller."""
    ben = names.index("restoration_benefit")
    cost = names.index("implementation_cost") if "implementation_cost" in names else names.index("cost")
    return ben, cost


# ===========================================================================
# the standalone evaluator: (selection, variant layers) -> (benefit, cost)
# ===========================================================================
def effect_params_of(scenario_params):
    """Mirror RestorationProblem.__init__ (resto_anom.py:541-551) exactly.

    anomaly_weight_shape / _scale are deliberately absent: restoration_effect reads them
    off effect_params, which __init__ never populates, so they are always the defaults.
    """
    return {
        "abiotic_effect": float(scenario_params.get("abiotic_effect", 0.01)),
        "biotic_effect": float(scenario_params.get("biotic_effect", 0.01)),
        "neighbor_radius": int(scenario_params.get("neighbor_radius", 3)),
        "neighbor_effect_decay": float(scenario_params.get("neighbor_effect_decay", 0.2)),
        "anomaly_weight_shape": "exponential",
        "anomaly_weight_scale": 1.0,
    }


def anomaly_weight(anomaly_values, shape="exponential", scale=1.0):
    """Copy of resto_anom.anomaly_improvement_weight, kept local so the fast path has no
    hidden dependency on the search engine. `layers` asserts the two agree."""
    neg_mask = anomaly_values < 0
    abs_anomaly = np.abs(anomaly_values)
    if shape == "exponential":
        weights = 1 - np.exp(-abs_anomaly / scale)
    elif shape == "gaussian":
        weights = np.exp(-(abs_anomaly ** 2) / (2 * scale ** 2))
    else:
        raise ValueError(f"Unknown weight shape: {shape}")
    weights = weights ** 3.0
    weights = np.where(neg_mask, weights, 0.0)
    return np.clip(weights, 0.0, 1.0)


def build_variant_layer(abiotic0, biotic0, eligible_mask, eff):
    """Per-cell direct benefit raster d_v, zero outside the variant's eligible mask.

    Zeroing outside the mask reproduces the engine's
    `np.where(restoration_eligible_mask, updated, original)` clip (resto_anom.py:492) for
    both the direct term and the spillover term, so a plan gets no credit for cells this
    variant considers ineligible.
    """
    shp, scl = eff["anomaly_weight_shape"], eff["anomaly_weight_scale"]
    d = (eff["abiotic_effect"] * anomaly_weight(np.asarray(abiotic0, float), shp, scl)
         + eff["biotic_effect"] * anomaly_weight(np.asarray(biotic0, float), shp, scl))
    return np.where(np.asarray(eligible_mask, bool), d, 0.0)


_KERNELS = {}


def disc_kernel(radius):
    """The engine's spillover kernel (resto_anom.py:467-469): x^2 + y^2 <= r^2."""
    if radius not in _KERNELS:
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        _KERNELS[radius] = (x * x + y * y) <= radius * radius
    return _KERNELS[radius]


def plan_masks(sel_flat, shape, radius):
    """(selected, spillover) 2-D masks. Variant-independent, so computed ONCE per plan."""
    sel_m = np.zeros(shape, dtype=bool)
    sel_m.flat[np.asarray(sel_flat, np.int64)] = True
    if radius <= 0:
        return sel_m, np.zeros(shape, dtype=bool)
    nb_m = ndimage.binary_dilation(sel_m, structure=disc_kernel(radius)) & ~sel_m
    return sel_m, nb_m


def evaluate(sel_flat, d_v, cost_raster, shape, radius, decay, masks=None):
    """Standalone (selection, variant layers) -> (benefit, cost). No problem, no search.

    `benefit` carries the engine's sign convention: NEGATED, so more negative = more
    improvement. `masks` lets a caller reuse one dilation across every variant.

    Gathers on flat indices rather than boolean-masking the full raster: the selection and
    its spillover ring are ~2 % of the grid, so this is ~10x cheaper per variant and the
    cross-evaluation is dominated by the (variant-independent) dilation instead.
    """
    sel_m, nb_m = plan_masks(sel_flat, shape, radius) if masks is None else masks
    sel_idx = np.asarray(sel_flat, np.int64)
    nb_idx = np.flatnonzero(nb_m.ravel())
    d_flat = np.asarray(d_v, np.float64).ravel()
    benefit = -(float(d_flat[sel_idx].sum()) + decay * float(d_flat[nb_idx].sum()))
    cost = float(np.asarray(cost_raster).ravel()[sel_idx].sum())
    return benefit, cost


# ===========================================================================
# 2-D Pareto helpers (both objectives minimised; benefit is already negated)
# ===========================================================================
def nondominated_2d(F):
    """Boolean mask of the non-dominated rows of an (n, 2) minimisation matrix.

    Duplicated points keep one representative (the sweep is strict on the second column).
    """
    F = np.asarray(F, float)
    keep = np.zeros(F.shape[0], dtype=bool)
    best = np.inf
    for i in np.lexsort((F[:, 1], F[:, 0])):
        if F[i, 1] < best:
            keep[i] = True
            best = F[i, 1]
    return keep


def front_reference(benefit, cost):
    """Cost-matched reference step function from a set of (benefit, cost) points.

    Returns (cost_sorted, best_benefit_cum) where best_benefit_cum[k] is the best (lowest,
    i.e. most negative) benefit achievable at cost <= cost_sorted[k] on this front.
    """
    F = np.column_stack([np.asarray(benefit, float), np.asarray(cost, float)])
    nd = nondominated_2d(F)
    b, c = F[nd, 0], F[nd, 1]
    order = np.argsort(c, kind="mergesort")
    return c[order], np.minimum.accumulate(b[order])


def regret_against(ref, benefit, cost):
    """Cost-matched benefit regret of each (benefit, cost) against a reference front.

        regret = max(0, benefit - ref(cost)) / |ref(cost)|

    Zero when the plan is cheaper than every point on the reference front: it extends the
    front rather than falling short. Those cases are returned separately so they can be
    counted rather than silently folded into the "perfectly robust" bucket.
    """
    c_ref, b_ref = ref
    benefit = np.asarray(benefit, float)
    cost = np.asarray(cost, float)
    idx = np.searchsorted(c_ref, cost, side="right") - 1
    uncovered = idx < 0
    safe = np.clip(idx, 0, None)
    ref_b = b_ref[safe]
    with np.errstate(divide="ignore", invalid="ignore"):
        reg = np.maximum(0.0, benefit - ref_b) / np.abs(ref_b)
    reg = np.where(np.isfinite(reg), reg, 0.0)
    reg[uncovered] = 0.0
    ref_b = np.where(uncovered, np.nan, ref_b)
    return reg, ref_b, uncovered


def jaccard(a, b):
    """Jaccard index of two sorted flat-index arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=False).size
    union = a.size + b.size - inter
    return float(inter) / union if union else 1.0


def _hash_array(a):
    """Stable content hash of an array, for the invariance assertions."""
    a = np.ascontiguousarray(a)
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


class ExtractMismatch(ValueError):
    """A run disagrees with its manifest or with an invariant of the pooled analysis.

    Distinct from a load failure so `extract` can keep going past a half-written pkl
    while still refusing to pool runs that are not comparable.
    """


# ===========================================================================
# shard + archive caches
# ===========================================================================
def _plan_shard(campaign, tag, seed):
    return PLAN_SHARD_DIR / f"{campaign}__{tag}__seed{seed}.npz"


def _layer_shard(tag):
    return LAYER_SHARD_DIR / f"{tag}.npz"


def _shard_is_fresh(shard, pkl):
    """A shard is reusable when it exists and is not older than the pkl it came from."""
    try:
        return shard.exists() and shard.stat().st_mtime >= os.path.getmtime(pkl)
    except OSError:
        return False


def save_archive(plans, campaigns, tags, seeds, axes_cols, native_benefit, native_cost,
                 dup_counts, provenance):
    ensure(OUT_DIR)
    offsets = np.zeros(len(plans) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([p.size for p in plans])
    flat = (np.concatenate(plans) if plans else np.empty(0, np.int32)).astype(np.int32)
    payload = {
        "sel_flat": flat, "sel_offsets": offsets,
        "plan_campaign": np.array(campaigns, dtype=object),
        "plan_tag": np.array(tags, dtype=object),
        "plan_seed": np.asarray(seeds, np.int32),
        "native_benefit": np.asarray(native_benefit, float),
        "native_cost": np.asarray(native_cost, float),
        "dup_count": np.asarray(dup_counts, np.int32),
        "axis_names": np.array(list(AXIS_NAMES), dtype=object),
    }
    for a in AXIS_NAMES:
        payload[f"axis__{a}"] = np.array(axes_cols[a], dtype=object)
    np.savez_compressed(ARCHIVE_NPZ, **payload)
    with open(PROVENANCE_JSON, "w", encoding="ascii") as f:
        json.dump(provenance, f, indent=1)
    print(f"  archive -> {ARCHIVE_NPZ}  ({ARCHIVE_NPZ.stat().st_size / 1e6:.0f} MB)")


def load_archive():
    """The pooled archive across every campaign, as flat raster indices + axis columns."""
    if not ARCHIVE_NPZ.exists():
        raise FileNotFoundError(f"{ARCHIVE_NPZ} missing - run the `extract` stage first.")
    z = np.load(ARCHIVE_NPZ, allow_pickle=True)
    off, flat = z["sel_offsets"], z["sel_flat"]
    n = len(off) - 1
    out = {
        "plans": [flat[off[i]:off[i + 1]] for i in range(n)],
        "campaign": np.array([str(t) for t in z["plan_campaign"]]),
        "tag": np.array([str(t) for t in z["plan_tag"]]),
        "seed": z["plan_seed"],
        "native_benefit": z["native_benefit"], "native_cost": z["native_cost"],
        "dup_count": z["dup_count"],
    }
    out["axes"] = {a: np.array([str(t) for t in z[f"axis__{a}"]]) for a in AXIS_NAMES}
    return out


def load_layers(tags=None):
    """Variant layers from the per-tag shards + the shared common.npz.

    Deliberately NOT one monolithic npz: 62 tags of float32 is ~340 MB, and merging them
    into a single archive would double that at write time while making an incremental
    campaign addition rewrite the whole file. `d` is float32 (see VALIDATE_RTOL_BENEFIT);
    eligible masks are de-duplicated by content hash, since eligibility follows the
    construction and not the scaling.
    """
    if not COMMON_NPZ.exists():
        raise FileNotFoundError(f"{COMMON_NPZ} missing - run the `extract` stage first.")
    available = sorted(p.stem for p in LAYER_SHARD_DIR.glob("*.npz"))
    tags = available if tags is None else [t for t in tags if t in available]
    d, elig, elig_hash = {}, {}, {}
    mask_of_hash = {}
    for t in tags:
        # Closed per shard rather than left to the GC: 62 lazily-open NpzFiles would sit
        # on 62 file handles for the life of the stage.
        with np.load(_layer_shard(t), allow_pickle=True) as z:
            d[t] = z["d"]
            h = str(z["elig_hash"])
            if h not in mask_of_hash:
                mask_of_hash[h] = z["elig"]
        elig[t] = mask_of_hash[h]
        elig_hash[t] = h
    with np.load(COMMON_NPZ, allow_pickle=True) as z:
        c = {k: z[k] for k in z.files}
    return {
        "tags": tags, "d": d, "elig": elig, "elig_hash": elig_hash,
        "n_distinct_masks": len(mask_of_hash),
        "cost": c["cost"], "shape": tuple(int(v) for v in c["shape"]),
        "radius": int(c["radius"]), "decay": float(c["decay"]),
        "eff": json.loads(str(c["eff"])), "crs": str(c["crs"]),
        "transform": c["transform"],
        "elig_counts": json.loads(str(c["elig_counts"])),
        "scenario_params": json.loads(str(c["scenario_params"])),
    }


def tag_axes_table():
    """{tag: (campaign, axes dict)} from the run manifest, for grouping variant columns."""
    out = {}
    for (camp_name, tag, _seed), v in load_selected_runs().items():
        out.setdefault(tag, (camp_name, v["axes"]))
    return out


# ===========================================================================
# the report: every printed headline number, machine-readable
# ===========================================================================
def _jsonable(o):
    """json.dump default hook: numpy scalars and arrays -> plain Python."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return float(o) if isinstance(o, np.floating) else bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)!r}")


def _pct_block(x, pcts=(50, 75, 90)):
    """The percentile summary the stages print, as a dict. Empty input -> n=0 only."""
    x = np.asarray(x, float)
    if not x.size:
        return {"n": 0}
    out = {f"p{p}": float(np.percentile(x, p)) for p in pcts}
    out.update(n=int(x.size), mean=float(x.mean()))
    return out


def report_update(stage, payload):
    """Merge one stage's headline numbers into report.json under its own key.

    Each stage owns one key, so re-running a single stage refreshes only that block and
    the rest of the document keeps the numbers from the run that produced the caches that
    stage read. Nothing here is computed: every value is one the stage already printed.
    """
    ensure(OUT_DIR)
    doc = {}
    if REPORT_JSON.exists():
        try:
            with open(REPORT_JSON, encoding="ascii") as f:
                doc = json.load(f)
        except (ValueError, OSError) as e:  # noqa: BLE001 - a corrupt report is not fatal
            print(f"  (report.json unreadable, starting a fresh one: {e!r})")
    doc[stage] = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    doc["config"] = {
        "axis_names": list(AXIS_NAMES),
        "campaigns": {
            # "levels" is every level the campaign covers, held-out ones included, so the
            # manuscript's design table lists them; "levels_pooled" is the balanced core
            # the seed-pooled statistics were actually computed over, and "holdout" names
            # the difference. A reader of report.json can tell the two apart without
            # knowing this module.
            name: {"source": list(c["source"]), "axes": list(c["axes"]),
                   "reference": dict(c["reference"]),
                   "objective_construction": c["objective_construction"],
                   "levels": {a: list(c["expect"][a]) for a in c["axes"]},
                   "levels_pooled": {a: balanced_levels(c, a) for a in c["axes"]},
                   "holdout": {a: {lv: list(s) for lv, s in h.items()}
                               for a, h in c.get("holdout", {}).items()}}
            for name, c in CAMPAIGNS.items()
        },
        "seeds": list(SEEDS), "run_objectives": list(RUN_OBJECTIVES),
        "reference_mode": REFERENCE_MODE, "robust_quantile": ROBUST_QUANTILE,
        "core_freq": CORE_FREQ,
        "discrim_min_span": DISCRIM_MIN_SPAN,
        "exclude_saturated": EXCLUDE_SATURATED,
        "max_plans_per_run": MAX_PLANS_PER_RUN,
    }
    with open(REPORT_JSON, "w", encoding="ascii") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=_jsonable)
    print(f"  report  -> {REPORT_JSON} [{stage}]")


# ===========================================================================
# stage: extract
# ===========================================================================
def _extract_one(run, need_layer):
    """Read one pkl -> (plan shard payload, layer payload or None, invariants)."""
    res = pickle_load(run["pkl"])
    campaign, tag, seed = run["campaign"], run["tag"], run["seed"]
    camp = CAMPAIGNS[campaign]

    label = str(res.get("run_label", run["label"]))
    rc = res.get("run_config", {}) or {}
    # run_registry.jsonl always records a null seed; run_config is the trustworthy copy.
    cfg_tag = rc.get("condition_scenario")
    cfg_seed = rc.get("random_seed")
    if cfg_tag is not None and cfg_tag != tag:
        raise ExtractMismatch(f"{run['pkl']}: discovery tag {tag!r} != run_config {cfg_tag!r}")
    if cfg_seed is not None and int(cfg_seed) != seed:
        raise ExtractMismatch(f"{run['pkl']}: discovery seed {seed} != run_config {cfg_seed}")
    if label != run["label"]:
        raise ExtractMismatch(f"{run['pkl']}: run_label {label!r} != manifest {run['label']!r}")
    # The axes parsed off the label must rebuild the tag that actually selected the
    # rasters, or the axis decomposition is describing a different run than the layers.
    if tag_from_axes(run["axes"], camp) != tag:
        raise ExtractMismatch(f"{run['pkl']}: axes {run['axes']} rebuild "
                         f"{tag_from_axes(run['axes'], camp)!r}, not {tag!r}")
    if list(res["objective_names"]) != RUN_OBJECTIVES:
        raise ExtractMismatch(f"{run['pkl']}: objectives {res['objective_names']} != "
                         f"{RUN_OBJECTIVES}")

    names = list(res["objective_names"])
    bi, ci = _obj_indices(names)
    raw = np.asarray(res["objectives_raw"], float)
    nd = np.asarray(res["is_nondominated"], bool)
    dec = np.asarray(res["decisions"])
    ic = res["initial_conditions"]
    rest_idx = np.asarray(ic["restoration_eligible_indices"], np.int64)
    n_rest = rest_idx.size

    eff = effect_params_of(res["scenario_params"])
    params = {k: res["scenario_params"].get(k) for k in CONSISTENT_PARAMS}
    cost_raster = np.asarray(ic["implementation_cost"])   # native float32 on purpose
    shape = tuple(int(v) for v in ic["shape"])
    invariants = {
        "eff": eff, "params": params, "shape": shape,
        "cost_hash": _hash_array(cost_raster),
        "crs": str(ic.get("crs", "")),
        "transform": list(ic.get("transform", [])) or [1, 0, 0, 0, 1, 0],
        "n_rest": int(n_rest),
    }

    sel_bool = dec[nd][:, :n_rest] > 0.5
    raw_nd = raw[nd]
    if MAX_PLANS_PER_RUN is not None and sel_bool.shape[0] > MAX_PLANS_PER_RUN:
        keep = np.linspace(0, sel_bool.shape[0] - 1, MAX_PLANS_PER_RUN).astype(int)
        sel_bool, raw_nd = sel_bool[keep], raw_nd[keep]
    plans = [rest_idx[sel_bool[k]].astype(np.int32) for k in range(sel_bool.shape[0])]
    offsets = np.zeros(len(plans) + 1, np.int64)
    offsets[1:] = np.cumsum([p.size for p in plans])
    shard = {
        "sel_flat": (np.concatenate(plans) if plans else np.empty(0, np.int32)),
        "sel_offsets": offsets,
        "native_benefit": raw_nd[:, bi].astype(float),
        "native_cost": raw_nd[:, ci].astype(float),
        "campaign": np.array(campaign, dtype=object),
        "tag": np.array(tag, dtype=object), "seed": np.int64(seed),
        "axes": np.array(json.dumps(run["axes"]), dtype=object),
        "invariants": np.array(json.dumps(invariants), dtype=object),
    }

    layer = None
    if need_layer:
        mask = np.asarray(ic["restoration_eligible_mask"], bool)
        d = build_variant_layer(ic["abiotic_anomaly"], ic["biotic_anomaly"], mask, eff)
        layer = {"d": d.astype(np.float32), "elig": mask,
                 "elig_hash": np.array(_hash_array(mask), dtype=object),
                 "elig_count": np.int64(n_rest)}
    common = {"cost": cost_raster, "shape": shape, "eff": eff,
              "crs": invariants["crs"], "transform": invariants["transform"],
              "params": params}
    del res, dec, ic, raw
    return shard, layer, common, invariants


def cmd_extract(argv=()):
    """Cache every run's non-dominated plans and every tag's benefit raster. Resumable."""
    dry = "--dry-run" in argv
    ensure(OUT_DIR)
    runs = gather_runs()
    skipped = runs.pop("skipped")
    print("Run discovery:")
    for name, camp in CAMPAIGNS.items():
        print(f"  {name:<26} {camp['source'][0]}: {camp['source'][1]}")
    if skipped:
        # Grouped by reason, not listed one by one: most exclusions here are expected
        # (the weighting folder's global_all runs are a two_block raster pair and so are
        # not a level of the weighting axis; the registry still carries the June 2026
        # 3-objective runs under the same factorial labels), and a real problem has to
        # stay visible among them.
        by_reason = {}
        for label, camp, why in skipped:
            by_reason.setdefault((camp, why), []).append(label)
        print(f"\n  {len(skipped)} entry/entries not usable, by reason:")
        for (camp, why), labels in sorted(by_reason.items(),
                                          key=lambda kv: -len(kv[1])):
            print(f"    {len(labels):>4}  [{camp}] {why}")
            print(f"          e.g. {labels[0]}")
    if not runs:
        raise SystemExit("No usable runs found for any configured campaign.")

    design = check_design(runs)
    n_expected = sum(d["n_expected"] for d in design.values())
    print(f"\nTotal: {len(runs)} runs discovered, {n_expected} expected by the designs.")
    for name, d in design.items():
        if not d["complete"]:
            print(f"  WARNING: {name} is INCOMPLETE ({len(d['missing'])} cell(s) missing)."
                  " `interact` needs a balanced design and will refuse to run.")
    if dry:
        print("\n--dry-run: discovery and balance check only, no pkl was read.")
        return

    ensure(PLAN_SHARD_DIR)
    ensure(LAYER_SHARD_DIR)
    inv_ref = None
    elig_counts = {}
    common_written = COMMON_NPZ.exists()
    n_read = n_cached = n_unreadable = 0
    selected = {}

    for key in sorted(runs):
        run = runs[key]
        campaign, tag, seed = key
        selected[f"{campaign}|{tag}|{seed}"] = {
            "campaign": campaign, "tag": tag, "seed": seed,
            "pkl": run["pkl"], "label": run["label"], "axes": run["axes"]}
        pshard, lshard = _plan_shard(campaign, tag, seed), _layer_shard(tag)
        need_layer = not _shard_is_fresh(lshard, run["pkl"])
        if _shard_is_fresh(pshard, run["pkl"]) and not need_layer and common_written:
            n_cached += 1
            continue
        t0 = time.perf_counter()
        try:
            shard, layer, common, inv = _extract_one(run, need_layer)
        except ExtractMismatch:
            # A run that is not comparable with the others must stop the pooling, not be
            # skipped past - every downstream statistic assumes one grid, one cost raster
            # and one spillover parameterisation.
            raise
        except Exception as e:  # noqa: BLE001 - a half-written pkl is not fatal
            # The grid driver may still be writing this file; report and carry on rather
            # than losing the whole extract to one cell.
            print(f"  {tag:<26} seed{seed}  UNREADABLE, skipped ({e!r})")
            n_unreadable += 1
            continue
        n_read += 1

        # -- invariants that the whole cross-evaluation rests on ------------
        if inv_ref is None:
            inv_ref = inv
        else:
            for k, why in (("eff", "the cross-evaluation assumes one spillover "
                                   "parameterisation"),
                           ("params", "those runs are not comparable"),
                           ("shape", "plans are carried as flat indices into one grid"),
                           ("cost_hash", "cost is assumed variant-invariant")):
                if inv[k] != inv_ref[k]:
                    raise ExtractMismatch(
                        f"{run['pkl']}: {k} is {inv[k]!r}, but an earlier run had "
                        f"{inv_ref[k]!r} - {why}.")

        np.savez_compressed(pshard, **shard)
        if layer is not None:
            np.savez_compressed(lshard, **layer)
            elig_counts[tag] = int(layer["elig_count"])
        if not common_written:
            np.savez_compressed(
                COMMON_NPZ, cost=common["cost"],
                shape=np.asarray(common["shape"], np.int64),
                radius=np.int64(common["eff"]["neighbor_radius"]),
                decay=np.float64(common["eff"]["neighbor_effect_decay"]),
                eff=np.array(json.dumps(common["eff"]), dtype=object),
                crs=np.array(common["crs"], dtype=object),
                transform=np.asarray(common["transform"], float),
                elig_counts=np.array(json.dumps({}), dtype=object),
                scenario_params=np.array(json.dumps(common["params"]), dtype=object))
            common_written = True
        print(f"  {tag:<26} seed{seed}  plans={len(shard['sel_offsets']) - 1:>4}  "
              f"{'+layer' if layer is not None else '      '}  "
              f"({time.perf_counter() - t0:.0f}s)")

    print(f"\n{n_read} run(s) read, {n_cached} served from shards, "
          f"{n_unreadable} unreadable.")

    # -- refresh the eligible counts recorded in common.npz -----------------
    # Includes tags whose layer shard was served from cache this run, so common.npz
    # always describes every layer on disk and not just the ones just written.
    for p in sorted(LAYER_SHARD_DIR.glob("*.npz")):
        if p.stem not in elig_counts:
            with np.load(p, allow_pickle=True) as z:
                elig_counts[p.stem] = int(z["elig_count"])
    # Materialise and close BEFORE rewriting: an NpzFile holds the path open, and on
    # Windows savez_compressed to the same path would then fail.
    with np.load(COMMON_NPZ, allow_pickle=True) as c:
        keep = {k: c[k] for k in c.files if k != "elig_counts"}
    keep["elig_counts"] = np.array(json.dumps(elig_counts), dtype=object)
    np.savez_compressed(COMMON_NPZ, **keep)

    with open(OUT_DIR / "selected_runs.json", "w", encoding="ascii") as f:
        json.dump(selected, f, indent=1)

    n_plans = _pool_shards(selected)
    # Every run may have been served from shards, in which case the invariants were not
    # re-read; recover the scenario params from the cache so the budget line still prints.
    if inv_ref is None:
        inv_ref = {"params": json.loads(str(keep["scenario_params"])),
                   "eff": json.loads(str(keep["eff"])),
                   "shape": tuple(int(v) for v in keep["shape"]),
                   "cost_hash": "(not re-read - every run served from shards)"}
    # int(), not round(): the engine truncates (n_restoration * max_restoration_fraction),
    # which is what makes the observed plan sizes 21848 for drop_smd and 21855 elsewhere.
    budget = {t: int(n * float(inv_ref["params"]["max_restoration_fraction"]))
              for t, n in elig_counts.items()}
    print("\nEligible restoration pixels per tag (and the pixel-count budget they imply):")
    for tag in sorted(elig_counts):
        print(f"  {tag:<26} {elig_counts[tag]:>7}  budget {budget[tag]}")
    print(f"Scenario: {inv_ref['params']}")
    print(f"Spillover: radius={inv_ref['eff']['neighbor_radius']} "
          f"decay={inv_ref['eff']['neighbor_effect_decay']} "
          f"ab={inv_ref['eff']['abiotic_effect']} bi={inv_ref['eff']['biotic_effect']}")
    print(f"Cost raster identical across every run (hashed): {inv_ref['cost_hash']}")

    report_update("extract", {
        "n_runs_discovered": len(runs), "n_runs_read": n_read,
        "n_runs_from_cache": n_cached, "n_unreadable": n_unreadable,
        "n_runs_expected": n_expected, "n_plans": n_plans,
        "design": design,
        "elig_counts": elig_counts, "budget_pixels": budget,
        "n_variant_tags": len(elig_counts),
        "scenario_params": inv_ref["params"], "effect_params": inv_ref["eff"],
        "cost_raster_hash": inv_ref["cost_hash"],
        "grid_shape": list(inv_ref["shape"]),
        "skipped_entries": [[lbl, camp, why] for lbl, camp, why in skipped],
    })
    return n_plans


def _pool_shards(selected):
    """Merge the per-run plan shards into archive.npz, collapsing duplicate plans."""
    plans, campaigns, tags, seeds, ben, cost = [], [], [], [], [], []
    axes_cols = {a: [] for a in AXIS_NAMES}
    prov_of, seen = {}, {}
    n_runs = n_held = 0
    for k in sorted(selected):
        v = selected[k]
        # Held-out levels contribute a LENS, not plans: their layer shard is still built
        # (so every pooled plan can be scored under them), but their own runs' plans stay
        # out of the archive, where an under-replicated level would perturb the pooled
        # front, the core cells and the robust subset. See the CAMPAIGNS "holdout" note.
        if is_holdout_run(v["axes"], CAMPAIGNS[v["campaign"]]):
            n_held += 1
            continue
        p = _plan_shard(v["campaign"], v["tag"], v["seed"])
        if not p.exists():
            print(f"  (pool: no shard for {k} - it was unreadable at extract)")
            continue
        z = np.load(p, allow_pickle=True)
        off, flat = z["sel_offsets"], z["sel_flat"]
        b, c = z["native_benefit"], z["native_cost"]
        ax = json.loads(str(z["axes"]))
        n_runs += 1
        for i in range(len(off) - 1):
            sel = flat[off[i]:off[i + 1]]
            key = hashlib.blake2b(np.ascontiguousarray(sel).tobytes(),
                                  digest_size=16).digest()
            if key in seen:
                prov_of[str(seen[key])].append(f"{v['tag']}:{v['seed']}")
                continue
            seen[key] = len(plans)
            prov_of[str(len(plans))] = [f"{v['tag']}:{v['seed']}"]
            plans.append(sel)
            campaigns.append(v["campaign"])
            tags.append(v["tag"])
            seeds.append(v["seed"])
            ben.append(float(b[i]))
            cost.append(float(c[i]))
            for a in AXIS_NAMES:
                axes_cols[a].append(ax.get(a, NOT_APPLICABLE))

    if not plans:
        raise SystemExit("No plan shards to pool - nothing was extracted.")
    dup_counts = [len(prov_of[str(i)]) for i in range(len(plans))]
    sizes = np.array([p.size for p in plans])
    print(f"\nPooled {len(plans)} unique plans from {n_runs} runs "
          f"({sum(dup_counts) - len(plans)} duplicate(s) collapsed).")
    if n_held:
        print(f"  {n_held} run(s) on held-out levels contributed a scoring layer but no "
              "plans (CAMPAIGNS[...]['holdout']).")
    print(f"Plan size: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")
    save_archive(plans, campaigns, tags, seeds, axes_cols, ben, cost, dup_counts, prov_of)
    return len(plans)


# ===========================================================================
# stage: layers (validation gate)
# ===========================================================================
def _stratified_validation_tags(tags):
    """One tag per VALIDATE_TAG_PATTERNS, so all three axes and both objective
    constructions are exercised instead of the first few tags alphabetically."""
    out = []
    for pat in VALIDATE_TAG_PATTERNS:
        hit = next((t for t in sorted(tags) if re.search(pat, t) and t not in out), None)
        if hit:
            out.append(hit)
    return out


def cmd_layers(argv=()):
    """Validate the fast evaluator against the real engine. Aborts on disagreement."""
    from Core_optimisation.resto_anom import (
        RestorationProblem, anomaly_improvement_weight,
    )

    L = load_layers()
    shape, radius, decay = L["shape"], L["radius"], L["decay"]
    print(f"Grid {shape}, spillover radius {radius}, decay {decay}, "
          f"{len(L['tags'])} variant layers, {L['n_distinct_masks']} distinct eligible "
          "mask(s).")

    # -- 1. the local weight copy must be the engine's function -------------
    probe = np.linspace(-4, 4, 401)
    err = np.max(np.abs(anomaly_weight(probe) - anomaly_improvement_weight(probe)))
    print(f"\n[1] anomaly weight vs engine          max abs err = {err:.3e}")
    assert err == 0.0, "local anomaly_weight has drifted from resto_anom's version"

    # -- 2. degenerate inputs ----------------------------------------------
    tag0 = L["tags"][0]
    d0, cost0 = L["d"][tag0], L["cost"]
    b, c = evaluate(np.empty(0, np.int64), d0, cost0, shape, radius, decay)
    print(f"[2] empty selection                   benefit={b:.3e} cost={c:.3e}")
    assert b == 0.0 and c == 0.0

    # Highest-benefit eligible cell that is at least `radius` from every border, so the
    # hand-computed disc below needs no clipping special-case.
    elig_flat = np.flatnonzero(L["elig"][tag0].ravel())
    er, ec = np.divmod(elig_flat, shape[1])
    inner = elig_flat[(er >= radius) & (er < shape[0] - radius)
                      & (ec >= radius) & (ec < shape[1] - radius)]
    lone = int(inner[np.argmax(d0.ravel()[inner])])
    rr, cc = np.divmod(lone, shape[1])
    b_full, _ = evaluate([lone], d0, cost0, shape, radius, decay)
    b_nodecay, _ = evaluate([lone], d0, cost0, shape, radius, 0.0)
    disc = np.float64(d0[rr - radius:rr + radius + 1,
                         cc - radius:cc + radius + 1])[disc_kernel(radius)]
    hand = -(np.float64(d0[rr, cc]) + decay * (disc.sum() - np.float64(d0[rr, cc])))
    print(f"[3] single cell, decay=0              benefit={b_nodecay:.6e} "
          f"(direct only: {-np.float64(d0[rr, cc]):.6e})")
    assert abs(b_nodecay + np.float64(d0[rr, cc])) < 1e-15
    print(f"[4] single cell, hand-computed disc   fast={b_full:.6e} hand={hand:.6e} "
          f"err={abs(b_full - hand):.3e}")
    assert abs(b_full - hand) < 1e-12

    # -- 3. the real engine, on real plans ---------------------------------
    A = load_archive()
    found = load_selected_runs()
    rng = np.random.default_rng(0)
    tags_to_check = _stratified_validation_tags(L["tags"])
    per_tag = max(1, VALIDATE_N_PLANS // max(1, len(tags_to_check)))
    print(f"\n  validating against the engine on {len(tags_to_check)} stratified tag(s): "
          f"{', '.join(tags_to_check)}")
    worst_ben = worst_cost = worst_stored = worst_cache = 0.0
    n_checked = 0

    for tag in tags_to_check:
        run = next((v for (cn, t, s), v in sorted(found.items()) if t == tag), None)
        if run is None:
            continue
        res = pickle_load(run["pkl"])
        ic = res["initial_conditions"]
        # Float64 rebuild of this tag's layer, straight from the run's own rasters. Check
        # [8] scores the same plans against it, so the cost of caching d_v as float32 is
        # measured on its own rather than hidden inside the engine's looser tolerance.
        d64 = build_variant_layer(ic["abiotic_anomaly"], ic["biotic_anomaly"],
                                  np.asarray(ic["restoration_eligible_mask"], bool),
                                  L["eff"])
        problem = RestorationProblem(ic, res["scenario_params"])
        names = list(res["objective_names"])
        bi, ci = _obj_indices(names)
        nd = np.flatnonzero(np.asarray(res["is_nondominated"], bool))
        dec = np.asarray(res["decisions"])
        raw = np.asarray(res["objectives_raw"], float)
        rest_idx = np.asarray(ic["restoration_eligible_indices"], np.int64)
        n_rest = rest_idx.size
        pick = rng.choice(nd, size=min(per_tag, nd.size), replace=False)

        for i in pick:
            x = dec[i]
            eng = problem.evaluate_raw_objectives(x)
            flat = rest_idx[x[:n_rest] > 0.5]
            masks = plan_masks(flat, shape, radius)
            fb, fc = evaluate(flat, L["d"][tag], L["cost"], shape, radius, decay, masks)
            b64, _ = evaluate(flat, d64, L["cost"], shape, radius, decay, masks)
            e_ben, e_cost = float(eng[bi]), float(eng[ci])
            worst_ben = max(worst_ben, abs(fb - e_ben) / max(abs(e_ben), 1e-30))
            worst_cost = max(worst_cost, abs(fc - e_cost) / max(abs(e_cost), 1e-30))
            worst_stored = max(worst_stored,
                               abs(fb - float(raw[i, bi])) / max(abs(float(raw[i, bi])), 1e-30))
            worst_cache = max(worst_cache, abs(fb - b64) / max(abs(b64), 1e-30))
            n_checked += 1
        del res, dec, ic, problem, d64

    print(f"\n[5] benefit vs evaluate_raw_objectives  {n_checked} plans, "
          f"max rel err = {worst_ben:.3e}  (tol {VALIDATE_RTOL_BENEFIT:.0e})")
    print(f"[6] cost    vs evaluate_raw_objectives  max rel err = {worst_cost:.3e}  "
          f"(tol {VALIDATE_RTOL_COST:.0e}, expected exact)")
    print(f"[7] benefit vs stored objectives_raw    max rel err = {worst_stored:.3e}")
    print(f"[8] float32 layer cache vs float64      max rel err = {worst_cache:.3e}  "
          f"(tol {VALIDATE_RTOL_CACHE:.0e})")
    print("    [5]-[7] are bounded by the ENGINE's float32 anomaly rasters, not by this")
    print("    module; [8] is what the cached layer dtype has to meet on its own.")
    assert n_checked > 0, "no plans could be checked against the engine"
    assert worst_ben < VALIDATE_RTOL_BENEFIT, f"benefit disagrees with the engine ({worst_ben:.3e})"
    assert worst_cost < VALIDATE_RTOL_COST, f"cost disagrees with the engine ({worst_cost:.3e})"
    assert worst_stored < VALIDATE_RTOL_BENEFIT, f"benefit disagrees with stored raw ({worst_stored:.3e})"
    assert worst_cache < VALIDATE_RTOL_CACHE, (
        f"the float32 layer cache is losing precision ({worst_cache:.3e}) - it, not the "
        "engine, is the error source; revisit build_variant_layer's dtype.")
    print(f"\nVALIDATION PASSED - the fast evaluator reproduces the engine "
          f"({len(A['plans'])} plans in the archive).")
    report_update("layers", {
        "passed": True,
        "n_variant_layers": len(L["tags"]), "variant_tags": list(L["tags"]),
        "n_distinct_eligible_masks": L["n_distinct_masks"],
        "grid_shape": list(shape), "radius": int(radius), "decay": float(decay),
        "weight_vs_engine_max_abs_err": float(err),
        "empty_selection_benefit": float(b), "empty_selection_cost": float(c),
        "single_cell_nodecay_benefit": float(b_nodecay),
        "single_cell_disc_abs_err": float(abs(b_full - hand)),
        "n_plans_checked": int(n_checked),
        "validated_tags": list(tags_to_check),
        "benefit_vs_engine_max_rel_err": float(worst_ben),
        "cost_vs_engine_max_rel_err": float(worst_cost),
        "benefit_vs_stored_max_rel_err": float(worst_stored),
        "float32_cache_vs_float64_max_rel_err": float(worst_cache),
        "tol_benefit": VALIDATE_RTOL_BENEFIT, "tol_cost": VALIDATE_RTOL_COST,
        "tol_cache": VALIDATE_RTOL_CACHE,
        "n_plans_in_archive": len(A["plans"]),
    })


# ===========================================================================
# cross-evaluation core
# ===========================================================================
def cross_evaluate(plans, layers, tags=None, verbose=True, consume=False):
    """(n_plans, n_tags) benefit matrix, (n_plans,) cost, (n_plans, n_tags) overlap.

    One dilation per plan (variant-independent), then ONE gather across all variants.

    The variant rasters are stacked into a single (n_tags, n_cells) float32 matrix so a
    plan costs two fancy-index gathers rather than two per variant: at 62 variants the
    per-variant form spends most of its time missing cache on 62 separate strided reads
    of the same 21855 columns. Summation is forced to float64 so the accumulation is as
    precise as the engine's, which is what keeps `layers` inside VALIDATE_RTOL_BENEFIT.

    `consume=True` drops each raster from `layers["d"]` as it is stacked, halving peak
    memory (the stack is ~340 MB at 62 variants); the caller must not reuse that dict.
    """
    tags = list(layers["tags"] if tags is None else tags)
    shape, radius, decay = layers["shape"], layers["radius"], layers["decay"]
    n_p, n_t = len(plans), len(tags)
    n_cells = int(np.prod(shape))
    B = np.zeros((n_p, n_t))
    C = np.zeros(n_p)
    OV = np.zeros((n_p, n_t))
    cost_flat = np.asarray(layers["cost"]).ravel()

    D = np.empty((n_t, n_cells), np.float32)
    for j, t in enumerate(tags):
        D[j] = np.asarray(layers["d"][t], np.float32).ravel()
        if consume:
            layers["d"].pop(t, None)
    # Eligible masks are shared objects after the de-duplication in load_layers, so the
    # stack is over the DISTINCT masks and each variant just names a row.
    mask_rows, row_of = [], {}
    for t in tags:
        h = layers["elig_hash"][t]
        if h not in row_of:
            row_of[h] = len(mask_rows)
            mask_rows.append(layers["elig"][t].ravel())
    E = np.stack(mask_rows) if mask_rows else np.zeros((0, n_cells), bool)
    e_row = np.array([row_of[layers["elig_hash"][t]] for t in tags])

    t0 = time.perf_counter()
    for i, flat in enumerate(plans):
        sel_idx = np.asarray(flat, np.int64)
        _, nb_m = plan_masks(sel_idx, shape, radius)
        nb_idx = np.flatnonzero(nb_m.ravel())
        C[i] = cost_flat[sel_idx].sum()
        direct = D[:, sel_idx].sum(axis=1, dtype=np.float64)
        spill = D[:, nb_idx].sum(axis=1, dtype=np.float64)
        B[i] = -(direct + decay * spill)
        OV[i] = (E[:, sel_idx].mean(axis=1)[e_row] if sel_idx.size
                 else np.ones(n_t))
        if verbose and (i + 1) % 250 == 0:
            el = time.perf_counter() - t0
            print(f"    {i + 1}/{n_p} plans  ({el:.0f}s, eta {el / (i + 1) * (n_p - i - 1):.0f}s)")
    return B, C, OV, tags


def load_cross():
    """The plan x variant matrix written by `cross`."""
    p = OUT_DIR / "cross_raw.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the `cross` stage first.")
    z = np.load(p, allow_pickle=True)
    return z["B"], z["C"], z["OV"], [str(t) for t in z["tags"]]


# ===========================================================================
# stage: cross
# ===========================================================================
def cmd_cross(argv=()):
    """Score every pooled plan under every variant -> the plan x variant matrix."""
    A, L = load_archive(), load_layers()
    tags = list(L["tags"])
    print(f"Cross-evaluating {len(A['plans'])} plans x {len(tags)} variants "
          f"(1 dilation per plan)...")
    B, C, OV, tags = cross_evaluate(A["plans"], L, tags)

    # Self-consistency: a plan's benefit under its OWN variant must match the value the
    # search stored. A mismatch means the flat-index mapping or the layer cache is wrong.
    at = A["tag"]
    err_b = 0.0
    for j, t in enumerate(tags):
        m = at == t
        if not m.any():
            continue
        ref = np.abs(A["native_benefit"][m])
        err_b = max(err_b, float(np.max(np.abs(B[m, j] - A["native_benefit"][m])
                                        / np.maximum(ref, 1e-30))))
    err_c = float(np.max(np.abs(C - A["native_cost"])
                         / np.maximum(np.abs(A["native_cost"]), 1e-30)))
    print(f"  self-check vs stored objectives_raw: benefit {err_b:.3e} "
          f"(tol {VALIDATE_RTOL_BENEFIT:.0e}), cost {err_c:.3e} (tol {VALIDATE_RTOL_COST:.0e})")
    if err_b > VALIDATE_RTOL_BENEFIT or err_c > VALIDATE_RTOL_COST:
        raise SystemExit("Self-check FAILED - plans do not reproduce their own stored "
                         "objectives. Do not trust anything downstream.")

    lo = OV.min()
    print(f"  eligibility overlap: min {lo:.6f}, mean {OV.mean():.6f} "
          f"({'fine' if lo > 0.999 else 'CHECK - some plans use cells a variant drops'})")

    # Two reference fronts per variant (see REFERENCE_MODE), plus the coverage diagnostic
    # that explains why "own" is not usable on these runs.
    rows, n_unc, coverage = [], 0, {}
    print(f"\n  Reference-front coverage (plan cost spans {C.min():.0f}-{C.max():.0f}):")
    print(f"    {'variant':<26}{'n_own':>6}{'own front cost range':>24}"
          f"{'plans off the bottom':>22}")
    for j, t in enumerate(tags):
        m = at == t
        ref_pool = front_reference(B[:, j], C)
        reg_p, refb_p, _ = regret_against(ref_pool, B[:, j], C)
        if m.any():
            ref_own = front_reference(B[m, j], C[m])
            reg_o, refb_o, unc = regret_against(ref_own, B[:, j], C)
            n_unc += int(unc.sum())
            print(f"    {t:<26}{int(m.sum()):>6}"
                  f"{f'{ref_own[0].min():.0f} - {ref_own[0].max():.0f}':>24}"
                  f"{f'{int(unc.sum())} ({unc.mean():.0%})':>22}")
            coverage[t] = {"n_own": int(m.sum()),
                           "own_front_cost_min": float(ref_own[0].min()),
                           "own_front_cost_max": float(ref_own[0].max()),
                           "n_uncovered": int(unc.sum()), "frac_uncovered": float(unc.mean())}
        else:
            reg_o = refb_o = np.full(len(C), np.nan)
            print(f"    {t:<26}{0:>6}{'(no runs of its own)':>24}{'-':>22}")
            coverage[t] = {"n_own": 0}
        rows.append(pd.DataFrame({
            "plan_id": np.arange(len(C)), "native_campaign": A["campaign"],
            "native_tag": at, "native_seed": A["seed"],
            "variant": t, "benefit": B[:, j], "cost": C, "overlap": OV[:, j],
            "ref_benefit_own": refb_o, "regret_own": reg_o,
            "ref_benefit_pooled": refb_p, "regret_pooled": reg_p,
        }))
    df = pd.concat(rows, ignore_index=True)
    ensure(OUT_DIR)
    df.to_csv(OUT_DIR / "cross_matrix.csv", index=False)
    np.savez_compressed(OUT_DIR / "cross_raw.npz", B=B, C=C, OV=OV,
                        tags=np.array(tags, dtype=object))
    frac = n_unc / max(len(df), 1)
    print(f"\n  own-front reference: {n_unc} of {len(df)} plan x variant pair(s) ({frac:.0%}) "
          "fall below the whole front")
    if frac > 0.05:
        print("    -> the variants' fronts barely overlap in cost, so 'own' would hand out")
        print(f"       artificial zero regret. REFERENCE_MODE={REFERENCE_MODE!r} is in use.")
    print(f"  -> {OUT_DIR / 'cross_matrix.csv'}  ({len(df)} rows)")
    report_update("cross", {
        "n_plans": len(C), "n_variants": len(tags), "tags": list(tags),
        "n_rows": len(df),
        "selfcheck_benefit_max_rel_err": err_b, "selfcheck_cost_max_rel_err": err_c,
        "overlap_min": float(lo), "overlap_mean": float(OV.mean()),
        "cost_min": float(C.min()), "cost_max": float(C.max()),
        "own_front_coverage": coverage,
        "n_uncovered_own": int(n_unc), "frac_uncovered_own": float(frac),
    })


# ===========================================================================
# stage: discrim (the saturation guard)
# ===========================================================================
def cmd_discrim(argv=()):
    """Can each variant's objective tell two plans apart at all?

    A variant whose benefit is nearly flat over the feasible set hands every plan the
    same score, so every plan looks robust under it - which is an artefact of the
    objective, not a property of the plans. Sum-form benefit is known to flatten toward
    high benchmark quantiles (best plan / random plan was 1.13x at upper_q90), so this
    has to be measured rather than assumed.

    The criterion is the FRONT SPAN: (worst - best) / |best| over the pooled archive
    under that variant. Both ends are real region-grown plans, so it asks how far apart
    the objective can hold two plans that are actually on the table.

    A scattered-random comparator is also computed and reported, but is NOT the
    criterion - see DISCRIM_MIN_SPAN. Random plans are drawn ONCE from the intersection
    of every variant's eligible mask and reused across variants, so the dilations are
    computed once and the comparator is paired across tags. Being scattered they carry a
    far larger spillover ring than a contiguous plan of the same size, and they outscore
    the optimised archive under nearly every variant. That ratio is the price of the
    contiguity constraint, not a measure of discriminability.
    """
    A, L = load_archive(), load_layers()
    B, C, _OV, tags = load_cross()
    shape, radius, decay = L["shape"], L["radius"], L["decay"]

    sizes = np.array([p.size for p in A["plans"]])
    k = int(np.median(sizes))
    common = np.ones(int(np.prod(shape)), bool)
    for t in tags:
        common &= L["elig"][t].ravel()
    pool_idx = np.flatnonzero(common)
    print(f"Random null: {DISCRIM_N_RANDOM} plans of {k} cells drawn from the "
          f"{pool_idx.size} cells eligible under EVERY variant.")

    rng = np.random.default_rng(DISCRIM_RANDOM_SEED)
    rand_plans = [np.sort(rng.choice(pool_idx, size=k, replace=False))
                  for _ in range(DISCRIM_N_RANDOM)]
    RB, _RC, _RO, _ = cross_evaluate(rand_plans, L, tags, verbose=False)

    rows = []
    for j, t in enumerate(tags):
        best = float(B[:, j].min())          # most negative = most improvement
        worst = float(B[:, j].max())
        rand_med = float(np.median(RB[:, j]))
        span = (worst - best) / max(abs(best), 1e-30)
        ratio = abs(best) / max(abs(rand_med), 1e-30)
        rows.append({"variant": t, "best_benefit": best, "worst_benefit": worst,
                     "front_span": span, "random_median_benefit": rand_med,
                     "best_over_random_scattered": ratio,
                     "saturated": bool(span <= DISCRIM_MIN_SPAN)})
    df = pd.DataFrame(rows).sort_values("front_span")
    ensure(OUT_DIR)
    df.to_csv(OUT_DIR / "discriminability.csv", index=False)

    print(f"\n{'variant':<26}{'front span':>13}{'best/scattered':>16}   flag")
    for _, r in df.iterrows():
        print(f"{r['variant']:<26}{r['front_span']:>13.4f}"
              f"{r['best_over_random_scattered']:>16.3f}"
              f"   {'SATURATED' if r['saturated'] else ''}")
    sat = df.loc[df["saturated"], "variant"].tolist()
    print(f"\n  front span at or below {DISCRIM_MIN_SPAN}: {len(sat)} of {len(df)} "
          "variant(s).")
    if sat:
        print("  Low regret under these is uninformative - the objective barely separates")
        print("  plans there. classify excludes them from the worst-case max when")
        print(f"  EXCLUDE_SATURATED is True (currently {EXCLUDE_SATURATED}).")
        for t in sat:
            print(f"    {t}")

    # The span is expected to fall as the benchmark gets more ambitious; report that
    # gradient explicitly, since it is why the top of the scaling axis cannot be read on
    # benefit alone.
    scal_prefixes = ("global_", "upper_q25_", "upper_q50_", "upper_q75_")
    is_two_block = ~df["variant"].str.contains("_w_")
    span_by_scaling = {}
    print("\n  front span by benchmark scaling (the saturation gradient):")
    for pre in scal_prefixes:
        m = df["variant"].str.startswith(pre) & is_two_block
        if m.any():
            span_by_scaling[pre.rstrip("_")] = float(df.loc[m, "front_span"].median())
            print(f"    {pre.rstrip('_'):<12} median {span_by_scaling[pre.rstrip('_')]:.3f}"
                  f"  ({int(m.sum())} variants)")

    n_below = int((df["best_over_random_scattered"] < 1.0).sum())
    print(f"\n  scattered-random comparator: it OUTSCORES the optimised archive under "
          f"{n_below} of {len(df)} variant(s).")
    print("  That is the price of the contiguity constraint, not a discriminability")
    print("  result: spillover credits un-restored neighbours, so a scattered plan of the")
    print("  same size carries a much larger spillover ring than a region-grown one.")
    print(f"  -> {OUT_DIR / 'discriminability.csv'}")

    report_update("discrim", {
        "n_random": DISCRIM_N_RANDOM, "random_plan_size": k,
        "n_common_eligible": int(pool_idx.size),
        "min_span": DISCRIM_MIN_SPAN,
        "n_variants": len(df), "n_saturated": len(sat), "saturated": sat,
        "n_scattered_beats_archive": n_below,
        "front_span_by_scaling": span_by_scaling,
        "by_variant": {r["variant"]: {
            "best_over_random_scattered": r["best_over_random_scattered"],
            "front_span": r["front_span"], "saturated": r["saturated"]}
            for _, r in df.iterrows()},
    })


def saturated_tags():
    """Variants `discrim` flagged, or [] when it has not been run."""
    p = OUT_DIR / "discriminability.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df.loc[df["saturated"].astype(bool), "variant"].astype(str).tolist()


# ===========================================================================
# stage: noise (per axis)
# ===========================================================================
def _cores(A, shape, by="run"):
    """Cells selected by >= CORE_FREQ of a group's plans, keyed by that group.

    `by="run"`  -> {(campaign, tag, seed): idx}. One optimisation run. This is the unit
                   for anything where SEED IS THE MEASUREMENT: the seed floor in `noise`
                   and the per-run response in `interact`.
    `by="tag"`  -> {(campaign, tag): idx}. All seeds of a variant pooled into one set,
                   which is the convention the baseline analysis uses and the right unit
                   for comparing FORMULATIONS - otherwise a cross-variant number carries
                   the seed noise it is supposed to be compared against.
    """
    n_cells = int(np.prod(shape))
    keyer = (lambda i: (A["campaign"][i], A["tag"][i], int(A["seed"][i]))) if by == "run" \
        else (lambda i: (A["campaign"][i], A["tag"][i]))
    order = {}
    for i in range(len(A["plans"])):
        order.setdefault(keyer(i), []).append(i)
    return {k: _core_of(A, idx, n_cells) for k, idx in order.items()}


def _core_of(A, idx, n_cells):
    """Cells selected by >= CORE_FREQ of the plans at `idx`."""
    if not len(idx):
        return np.empty(0, np.int64)
    cnt = np.bincount(np.concatenate([A["plans"][i] for i in idx]).astype(np.int64),
                      minlength=n_cells)
    return np.flatnonzero(cnt >= CORE_FREQ * len(idx))


def cmd_noise(argv=()):
    """Seed noise floor vs each formulation axis, ranked so the axes can be compared.

    For axis `a` of campaign `c`, the comparison set is the cells that differ from the
    campaign reference ONLY in `a`; every other axis is held at its reference level, so
    the measured spread is attributable to `a` alone. The reference front is the
    leave-one-seed-out front of the reference cell, which is exactly the reference the
    seed floor uses - so the two distributions are directly comparable rather than one
    being handed a systematically stronger reference.
    """
    from scipy import stats as sps

    A, L = load_archive(), load_layers()
    B, C, _OV, tags = load_cross()
    tag_col = {t: j for j, t in enumerate(tags)}
    at, asd, acamp = A["tag"], A["seed"], A["campaign"]
    cores = _cores(A, L["shape"], by="run")      # seed floor: seed is the measurement
    cores_tag = _cores(A, L["shape"], by="tag")  # axis effects: seeds pooled per variant

    rows, reg_rows, j_rows = [], [], []
    dists = {}

    def _regret_block(ref, mask, jcol):
        r, _, _ = regret_against(ref, B[mask, jcol], C[mask])
        return r

    def _one_axis_apart(camp_name, camp, axis, ref_tag):
        """Plans whose cell differs from the campaign reference ONLY in `axis`.

        Independent of seed, so it is built once per (campaign, axis) rather than once
        per held-out seed.
        """
        m = (acamp == camp_name) & (at != ref_tag)
        for a in camp["axes"]:
            if a != axis:
                m &= A["axes"][a] == camp["reference"][a]
        # Drop held-out levels: this distribution pools each variant's own plans, so a
        # level run at fewer seeds than the rest would be compared on a different footing
        # (and against a seed floor built from a different depth). They are already
        # absent from the archive; this keeps the intent local and explicit.
        for a in camp["axes"]:
            held = holdout_levels(camp, a)
            if held:
                m &= ~np.isin(A["axes"][a], list(held))
        return m

    # ---- seed floor, per campaign reference cell --------------------------
    for camp_name, camp in CAMPAIGNS.items():
        ref_axes = dict(camp["reference"])
        ref_tag = tag_from_axes({**{a: NOT_APPLICABLE for a in AXIS_NAMES}, **ref_axes},
                                camp)
        if ref_tag not in tag_col:
            print(f"  {camp_name}: reference tag {ref_tag!r} has no layer - skipped")
            continue
        jcol = tag_col[ref_tag]
        in_ref = (acamp == camp_name) & (at == ref_tag)
        seeds_here = sorted(set(asd[in_ref].tolist()))
        within = []
        for s in seeds_here:
            hold = in_ref & (asd == s)
            base = in_ref & (asd != s)
            if not base.any() or not hold.any():
                continue
            ref = front_reference(B[base, jcol], C[base])
            w = _regret_block(ref, hold, jcol)
            within.append(w)
            k = np.flatnonzero(hold)
            reg_rows.append(pd.DataFrame({
                "campaign": camp_name, "axis": "seed", "level": ref_tag,
                "reference_variant": ref_tag, "held_out_seed": int(s),
                "plan_tag": at[k], "plan_seed": asd[k], "regret": w}))
        dists[(camp_name, "seed")] = (np.concatenate(within) if within
                                      else np.empty(0))

        # ---- one distribution per axis -----------------------------------
        for axis in camp["axes"]:
            acc = []
            other = _one_axis_apart(camp_name, camp, axis, ref_tag)
            if not other.any():
                print(f"  {camp_name}/{axis}: no cell differs from the reference on this "
                      "axis alone - skipped")
                continue
            for s in seeds_here:
                base = in_ref & (asd != s)
                if not base.any():
                    continue
                ref = front_reference(B[base, jcol], C[base])
                r = _regret_block(ref, other, jcol)
                acc.append(r)
                k = np.flatnonzero(other)
                reg_rows.append(pd.DataFrame({
                    "campaign": camp_name, "axis": axis, "level": "*",
                    "reference_variant": ref_tag, "held_out_seed": int(s),
                    "plan_tag": at[k], "plan_seed": asd[k], "regret": r}))
            dists[(camp_name, axis)] = np.concatenate(acc) if acc else np.empty(0)

    # ---- spatial floor: core Jaccard -------------------------------------
    # The two comparisons use DIFFERENT units, deliberately:
    #   seed  pairs of individual runs of the same variant - seed is the thing being
    #         measured, so it cannot be pooled away.
    #   axis  pairs of variants with all 5 seeds POOLED into one set per variant (the
    #         same convention the baseline analysis uses). Comparing single runs across
    #         variants would load the formulation number with the very seed noise it is
    #         being compared against.
    jac = {}
    axes_of_tag = {}
    for i in range(len(A["plans"])):
        axes_of_tag.setdefault((A["campaign"][i], A["tag"][i]),
                               {a: A["axes"][a][i] for a in AXIS_NAMES})
    for camp_name, camp in CAMPAIGNS.items():
        # -- seed pairs, per run, within each variant --
        seed_pairs = []
        runs_by_tag = {}
        for (cn, t, s) in cores:
            if cn == camp_name:
                runs_by_tag.setdefault(t, []).append((cn, t, s))
        for t, ks in runs_by_tag.items():
            for k1, k2 in itertools.combinations(sorted(ks), 2):
                v = jaccard(cores[k1], cores[k2])
                seed_pairs.append(v)
                j_rows.append({"campaign": camp_name, "comparison": "seed", "unit": "run",
                               "tag_a": t, "seed_a": int(k1[2]),
                               "tag_b": t, "seed_b": int(k2[2]), "jaccard": v})
        jac[(camp_name, "seed")] = np.array(seed_pairs)

        # -- seed floor ON THE SAME FOOTING as the axis rows --
        # A pooled core is built from ~5x more plans than a single run's, so a
        # pooled-vs-pooled Jaccard is not directly comparable to a run-vs-run one. Split
        # each variant's seeds into two disjoint groups, pool each, and compare: same
        # variant, same pooling, so the only thing left is the seed draw.
        half = []
        n_cells_j = int(np.prod(L["shape"]))
        g1, g2 = SEEDS[:2], SEEDS[2:]
        for t, ks in runs_by_tag.items():
            m = (acamp == camp_name) & (at == t)
            i1 = np.flatnonzero(m & np.isin(asd, g1))
            i2 = np.flatnonzero(m & np.isin(asd, g2))
            if i1.size and i2.size:
                half.append(jaccard(_core_of(A, i1, n_cells_j),
                                    _core_of(A, i2, n_cells_j)))
                j_rows.append({"campaign": camp_name, "comparison": "seed_pooled_halves",
                               "unit": f"{len(g1)} vs {len(g2)} seeds pooled",
                               "tag_a": t, "seed_a": None, "tag_b": t, "seed_b": None,
                               "jaccard": half[-1]})
        jac[(camp_name, "seed_pooled_halves")] = np.array(half)

        # -- axis pairs, seed-pooled per variant --
        tags_c = sorted(t for (cn, t) in cores_tag if cn == camp_name)
        for axis in camp["axes"]:
            cand = [t for t in tags_c
                    if is_reference_cell(axes_of_tag[(camp_name, t)], camp,
                                         except_axis=axis)]
            vals = []
            for t1, t2 in itertools.combinations(sorted(cand), 2):
                v = jaccard(cores_tag[(camp_name, t1)], cores_tag[(camp_name, t2)])
                vals.append(v)
                j_rows.append({"campaign": camp_name, "comparison": axis,
                               "unit": "variant (5 seeds pooled)",
                               "tag_a": t1, "seed_a": None,
                               "tag_b": t2, "seed_b": None, "jaccard": v})
            jac[(camp_name, axis)] = np.array(vals)

    # ---- verdicts, ranked -------------------------------------------------
    print("\n" + "=" * 86)
    print("NOISE FLOOR BY AXIS - cost-matched regret, leave-one-seed-out reference")
    print("=" * 86)
    print(f"{'campaign':<26}{'axis':<14}{'n':>7}{'p50':>9}{'p75':>9}{'p90':>9}"
          f"{'vs seed':>10}   verdict")
    stats_out = {}
    for camp_name in CAMPAIGNS:
        floor = dists.get((camp_name, "seed"))
        if floor is None:
            continue
        f75 = float(np.percentile(floor, 75)) if floor.size else 0.0
        for axis in ("seed",) + tuple(CAMPAIGNS[camp_name]["axes"]):
            x = dists.get((camp_name, axis))
            if x is None or not x.size:
                continue
            p50, p75, p90 = (float(np.percentile(x, p)) for p in (50, 75, 90))
            if axis == "seed":
                ratio, verdict, code, pval = np.nan, "(the floor)", "floor", np.nan
            else:
                ratio = (p75 / f75) if f75 > 0 else np.inf
                mw = sps.mannwhitneyu(x, floor, alternative="greater")
                pval = float(mw.pvalue)
                if pval >= 0.05 or p75 <= f75:
                    code, verdict = "not_above", "NOT above the seed floor"
                elif pval >= 0.01 or (np.isfinite(ratio) and ratio < 2.0):
                    code, verdict = "above_unclear", "above, but not clearly"
                else:
                    code, verdict = "above_clearly", "clearly above"
            rstr = "-" if axis == "seed" else (
                "inf" if not np.isfinite(ratio) else f"{ratio:.2f}x")
            print(f"{camp_name:<26}{axis:<14}{x.size:>7}{p50:>9.4f}{p75:>9.4f}"
                  f"{p90:>9.4f}{rstr:>10}   {verdict}")
            jv = jac.get((camp_name, axis), np.empty(0))
            stats_out[f"{camp_name}|{axis}"] = {
                "campaign": camp_name, "axis": axis,
                "regret": _pct_block(x), "jaccard": _pct_block(jv),
                "p75_ratio_vs_seed": (None if not np.isfinite(ratio) else float(ratio))
                                     if axis != "seed" else None,
                "mannwhitney_p": None if axis == "seed" else pval,
                "verdict_code": code, "verdict": verdict,
            }
            rows.append({"campaign": camp_name, "axis": axis, "n": int(x.size),
                         "regret_p50": p50, "regret_p75": p75, "regret_p90": p90,
                         "p75_ratio_vs_seed": (None if axis == "seed" else
                                               (None if not np.isfinite(ratio) else ratio)),
                         "mannwhitney_p": (None if axis == "seed" else pval),
                         "jaccard_n": int(jv.size),
                         "jaccard_median": (float(np.median(jv)) if jv.size else None),
                         "verdict_code": code})

    print(f"\n  core-selection agreement (Jaccard at CORE_FREQ={CORE_FREQ:.2f}, "
          "higher = same map):")
    print(f"  {'campaign':<26}{'comparison':<22}{'n':>6}{'median':>9}   unit")
    jac_out = {}
    for camp_name in CAMPAIGNS:
        for axis in ("seed", "seed_pooled_halves") + tuple(CAMPAIGNS[camp_name]["axes"]):
            jv = jac.get((camp_name, axis), np.empty(0))
            if not jv.size:
                continue
            unit = {"seed": "single run",
                    "seed_pooled_halves": "2 vs 3 seeds pooled"}.get(
                        axis, "variant, 5 seeds pooled")
            print(f"  {camp_name:<26}{axis:<22}{jv.size:>6}{np.median(jv):>9.3f}   {unit}")
            jac_out[f"{camp_name}|{axis}"] = {"campaign": camp_name, "comparison": axis,
                                              "unit": unit, **_pct_block(jv)}
    # The like-for-like read: axis rows against seed_pooled_halves, not against `seed`.
    print("  Compare each axis against `seed_pooled_halves`, not against `seed`: both are")
    print("  pooled over several seeds, so group size is held constant and the only")
    print("  difference left is the formulation. (`seed` is single runs, and is the floor")
    print("  the REGRET table above uses.)")
    for camp_name, camp in CAMPAIGNS.items():
        base = jac.get((camp_name, "seed_pooled_halves"), np.empty(0))
        if not base.size:
            continue
        b = float(np.median(base))
        for axis in camp["axes"]:
            jv = jac.get((camp_name, axis), np.empty(0))
            if not jv.size:
                continue
            v = float(np.median(jv))
            # A 10% margin, not a bare comparison: these medians rest on few pairs (as
            # low as C(4,2)=6 for scaling), so a difference of a couple of points is not
            # a finding in either direction.
            rel = v / b if b > 0 else np.inf
            if rel < 0.90:
                verdict = "RELOCATES the map"
            elif rel > 1.10:
                verdict = "agrees MORE than a re-seed does"
            else:
                verdict = "indistinguishable from a re-seed"
            jac_out[f"{camp_name}|{axis}"]["vs_seed_pooled_floor"] = rel
            jac_out[f"{camp_name}|{axis}"]["spatial_verdict"] = verdict
            print(f"    {camp_name}/{axis}: {v:.3f} vs seed-pooled floor {b:.3f} "
                  f"({rel:.2f}x, n={jv.size} vs {base.size})  -> {verdict}")
    print("=" * 86)

    ensure(OUT_DIR)
    pd.DataFrame(rows).to_csv(OUT_DIR / "noise_by_axis.csv", index=False)
    (pd.concat(reg_rows, ignore_index=True) if reg_rows else
     pd.DataFrame(columns=["campaign", "axis", "level", "reference_variant",
                           "held_out_seed", "plan_tag", "plan_seed", "regret"])
     ).to_csv(OUT_DIR / "noise_regret.csv", index=False)
    pd.DataFrame(j_rows).to_csv(OUT_DIR / "noise_jaccard.csv", index=False)
    for name in ("noise_by_axis.csv", "noise_regret.csv", "noise_jaccard.csv"):
        print(f"  -> {OUT_DIR / name}")
    report_update("noise", {
        "n_plans": len(A["plans"]), "core_freq": CORE_FREQ,
        "by_axis": stats_out, "jaccard_by_comparison": jac_out,
        "ranking": [r["axis"] for r in
                    sorted((r for r in rows if r["axis"] != "seed"),
                           key=lambda r: -r["regret_p75"])],
    })


# ===========================================================================
# stage: interact (scaling x construction, balanced two-way)
# ===========================================================================
def _two_way_anova(y, fa, fb):
    """Balanced two-way decomposition with replication. Returns a dict of SS/df/MS/F/p.

    Written out rather than pulled from statsmodels: the design is asserted balanced by
    `check_design`, and for a balanced design the sums of squares are unambiguous (no
    Type I/II/III distinction), so the four-line formula is the whole model.
    """
    from scipy import stats as sps
    y = np.asarray(y, float)
    fa = np.asarray(fa)
    fb = np.asarray(fb)
    la, lb = sorted(set(fa.tolist())), sorted(set(fb.tolist()))
    a, b = len(la), len(lb)
    n = len(y) // (a * b)
    grand = y.mean()
    cell = np.array([[y[(fa == u) & (fb == v)].mean() for v in lb] for u in la])
    ma = np.array([y[fa == u].mean() for u in la])
    mb = np.array([y[fb == v].mean() for v in lb])
    ss_a = n * b * float(((ma - grand) ** 2).sum())
    ss_b = n * a * float(((mb - grand) ** 2).sum())
    ss_ab = n * float(((cell - ma[:, None] - mb[None, :] + grand) ** 2).sum())
    ss_tot = float(((y - grand) ** 2).sum())
    ss_err = ss_tot - ss_a - ss_b - ss_ab
    df_a, df_b, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
    df_err = a * b * (n - 1)
    ms_err = ss_err / df_err if df_err > 0 else np.nan
    out = {"n_per_cell": n, "levels_a": la, "levels_b": lb,
           "ss_total": ss_tot, "ss_error": ss_err, "df_error": df_err,
           "ms_error": ms_err}
    for name, ss, df in (("a", ss_a, df_a), ("b", ss_b, df_b), ("ab", ss_ab, df_ab)):
        ms = ss / df if df > 0 else np.nan
        f = ms / ms_err if (ms_err and ms_err > 0) else np.nan
        p = float(sps.f.sf(f, df, df_err)) if np.isfinite(f) else np.nan
        out[name] = {"ss": ss, "df": df, "ms": ms, "F": f, "p": p,
                     "eta_sq": ss / ss_tot if ss_tot > 0 else np.nan}
    return out


def cmd_interact(argv=()):
    """Does leave-one-out sensitivity depend on the benchmark quantile?

    Response, per run (scaling s, construction c, seed k) - both displacements FROM the
    full-indicator plan at the same scaling, so the two axes are not structurally
    coupled the way a self-comparison would make them:

      spatial  1 - Jaccard( core(s, c, k), core pooled over the 5 seeds of (s, all) )
      benefit  median cost-matched regret of that run's plans, evaluated under variant
               f"{s}_all" - the full-indicator objective at that scaling - against the
               front pooled over the 5 seeds of (s, all) under the same variant.

    Both are evaluated under ONE variant per scaling, so the response is on one scale
    across construction levels rather than each level being normalised by its own front.

    The DECOMPOSITION runs on the drop_* rows only - a balanced 4 x 11 x 5 with seed as
    the replicate term. The `all` rows are computed and reported, but as the seed-noise
    level of the same response, NOT as a twelfth construction level: those runs helped
    build the reference they are scored against, so their response is low by construction
    and including them would inflate the construction main effect. (It would not bias the
    interaction, which is the headline term, but the main effect is quoted too.)
    """
    camp_name = "benchmark_x_construction"
    camp = CAMPAIGNS[camp_name]
    A, L = load_archive(), load_layers()
    B, C, _OV, tags = load_cross()
    tag_col = {t: j for j, t in enumerate(tags)}
    at, asd, acamp = A["tag"], A["seed"], A["campaign"]
    # Per RUN here: seed is the replicate term of the decomposition, so it must not be
    # pooled away. The reference each run is compared against IS seed-pooled (below).
    cores = _cores(A, L["shape"], by="run")
    n_cells = int(np.prod(L["shape"]))

    design = check_design(gather_runs(campaigns=[camp_name]), verbose=False)[camp_name]
    if not design["complete"]:
        raise SystemExit(
            f"{camp_name} is not a complete design ({len(design['missing'])} cell(s) "
            "missing). The two-way decomposition is a balanced-design formula; fix the "
            "grid or restrict the levels before running `interact`.")

    # Balanced core only: the decomposition below is a balanced-design formula, so a
    # held-out (under-replicated) level must not enter it.
    scalings = balanced_levels(camp, "scaling")
    constructions = camp["expect"]["construction"]
    rows = []
    for s in scalings:
        ref_tag = f"{s}_all"
        if ref_tag not in tag_col:
            raise SystemExit(f"no layer for the reference variant {ref_tag!r}")
        jcol = tag_col[ref_tag]
        ref_mask = (acamp == camp_name) & (at == ref_tag)
        ref_front = front_reference(B[ref_mask, jcol], C[ref_mask])
        ref_idx = np.flatnonzero(ref_mask)
        cnt = np.bincount(np.concatenate([A["plans"][i] for i in ref_idx]).astype(np.int64),
                          minlength=n_cells)
        ref_core = np.flatnonzero(cnt >= CORE_FREQ * ref_idx.size)
        for c in constructions:
            tag = f"{s}_{c}"
            for k in SEEDS:
                m = (acamp == camp_name) & (at == tag) & (asd == k)
                if not m.any():
                    raise SystemExit(f"no plans for {tag} seed{k} - design not balanced "
                                     "in the archive.")
                reg, _, unc = regret_against(ref_front, B[m, jcol], C[m])
                core = cores.get((camp_name, tag, k), np.empty(0, np.int64))
                rows.append({
                    "scaling": s, "construction": c, "seed": k, "tag": tag,
                    "n_plans": int(m.sum()),
                    "resp_benefit": float(np.median(reg)),
                    "frac_uncovered": float(unc.mean()),
                    "resp_spatial": 1.0 - jaccard(core, ref_core),
                    "core_size": int(core.size),
                })
    df = pd.DataFrame(rows)
    ensure(OUT_DIR)
    df.to_csv(OUT_DIR / "interaction.csv", index=False)

    # The decomposition set: drop_* only. See the docstring - the `all` runs helped build
    # the reference they are scored against, so they are the seed-noise level of this
    # response, not a construction level.
    loo = df[df["construction"] != "all"]
    base = df[df["construction"] == "all"]
    loo_constructions = [c for c in constructions if c != "all"]

    out = {}
    print(f"\nBalanced two-way decomposition, {len(scalings)} scalings x "
          f"{len(loo_constructions)} leave-one-out constructions x {len(SEEDS)} seeds "
          f"= {len(loo)} runs (the {len(base)} full-indicator runs are the seed level, "
          "not a level of the construction axis).")
    unc = loo["frac_uncovered"].mean()
    print(f"  mean fraction of plans falling below the reference front: {unc:.1%}"
          f"{'  (high - the benefit response is weakly identified)' if unc > 0.2 else ''}")
    for resp in ("resp_spatial", "resp_benefit"):
        y = loo[resp].to_numpy(float)
        res = _two_way_anova(y, loo["scaling"].to_numpy(), loo["construction"].to_numpy())
        res["seed_level_mean"] = float(base[resp].mean())
        out[resp] = res
        print(f"\n  response = {resp}   (mean {y.mean():.4f}, sd {y.std(ddof=1):.4f}; "
              f"seed level {res['seed_level_mean']:.4f})")
        if not np.isfinite(res["ms_error"]) or res["ms_error"] <= 0:
            print("    residual (seed) variance is ZERO - the response collapsed and the "
                  "F tests are void.")
            continue
        print(f"    {'term':<28}{'df':>5}{'SS':>14}{'eta^2':>9}{'F':>10}{'p':>12}")
        for key, label in (("a", "scaling"), ("b", "construction"),
                           ("ab", "scaling x construction")):
            r = res[key]
            print(f"    {label:<28}{r['df']:>5}{r['ss']:>14.5g}{r['eta_sq']:>9.3f}"
                  f"{r['F']:>10.2f}{r['p']:>12.3g}")
        print(f"    {'residual (seed)':<28}{res['df_error']:>5}{res['ss_error']:>14.5g}"
              f"{res['ss_error'] / res['ss_total']:>9.3f}")
        pv = res["ab"]["p"]
        if np.isfinite(pv) and pv < 0.05:
            print("    -> the leave-one-out displacement DOES depend on the benchmark "
                  "quantile.")
        else:
            print("    -> no evidence that the leave-one-out displacement depends on the "
                  "benchmark quantile.")

    print("\n  mean response by scaling (drop-only rows, the LOO displacement):")
    for s in scalings:
        print(f"    {s:<12} spatial {loo.loc[loo['scaling'] == s, 'resp_spatial'].mean():.4f}"
              f"   benefit {loo.loc[loo['scaling'] == s, 'resp_benefit'].mean():.4f}"
              f"   (seed level: spatial "
              f"{base.loc[base['scaling'] == s, 'resp_spatial'].mean():.4f})")
    print(f"  -> {OUT_DIR / 'interaction.csv'}")

    report_update("interact", {
        "campaign": camp_name, "n_runs": len(df), "n_runs_in_anova": len(loo),
        "scalings": scalings, "anova_constructions": loo_constructions,
        "constructions": constructions, "seeds": list(SEEDS),
        "mean_frac_uncovered": float(unc),
        "anova": out,
        "loo_mean_by_scaling": {
            s: {"spatial": float(loo.loc[loo["scaling"] == s, "resp_spatial"].mean()),
                "benefit": float(loo.loc[loo["scaling"] == s, "resp_benefit"].mean()),
                "seed_level_spatial": float(base.loc[base["scaling"] == s,
                                                     "resp_spatial"].mean())}
            for s in scalings},
    })


# ===========================================================================
# stage: classify
# ===========================================================================
def _axes_departed(tag, tag_axes):
    """Which of its campaign's axes this variant departs from the reference on."""
    if tag not in tag_axes:
        return ""
    camp_name, axes = tag_axes[tag]
    camp = CAMPAIGNS[camp_name]
    return "+".join(a for a in camp["axes"] if axes[a] != camp["reference"][a]) or "reference"


def _composition_by_axis(A, robust):
    """Archive/robust shares per LEVEL of each campaign axis, others at reference.

    `composition` answers "is the robust subset one family?" at the grain of a whole
    campaign. This answers it one axis at a time: inside a campaign, hold every OTHER
    axis of that campaign at its reference level and split the remaining plans by the
    level of the axis in question. On the crossed benchmark grid that reads `scaling`
    at construction "all" and `construction` at scaling "global" - the only slices
    where a level difference is attributable to that axis alone.

    Two share conventions per level, because they answer different questions:
      share_full / share_robust    denominators are the POOLED archive and the whole
                                   robust subset, i.e. the same convention as
                                   `composition`, so the rows are read against each
                                   other and against the campaign rows.
      *_slice                      denominators are the held slice, i.e. how the axis
                                   splits its own runs. Levels of one axis sum to 1.
    """
    out = {}
    n_full = len(robust)
    n_rob = max(int(robust.sum()), 1)
    for camp_name, camp in CAMPAIGNS.items():
        in_camp = A["campaign"] == camp_name
        if not in_camp.any():
            continue
        per_axis = {}
        for axis in camp["axes"]:
            held = in_camp.copy()
            for other in camp["axes"]:
                if other != axis:
                    held &= A["axes"][other] == camp["reference"][other]
            if not held.any():
                continue
            n_held, n_held_rob = int(held.sum()), int((held & robust).sum())
            levels = {}
            for lvl in sorted(set(A["axes"][axis][held].tolist())):
                m = held & (A["axes"][axis] == lvl)
                nl, nlr = int(m.sum()), int((m & robust).sum())
                sf, sr = nl / n_full, nlr / n_rob
                levels[str(lvl)] = {
                    "n_full": nl, "n_robust": nlr,
                    "share_full": sf, "share_robust": sr,
                    "share_full_slice": nl / max(n_held, 1),
                    "share_robust_slice": nlr / max(n_held_rob, 1),
                    "enrichment": sr / sf if sf else np.nan,
                    "is_reference": bool(lvl == camp["reference"][axis]),
                }
            per_axis[axis] = {
                "held_at": {o: camp["reference"][o]
                            for o in camp["axes"] if o != axis},
                "n_full": n_held, "n_robust": n_held_rob,
                "reference": camp["reference"][axis],
                "levels": levels,
            }
        if per_axis:
            out[camp_name] = per_axis
    return out


def cmd_classify(argv=()):
    """Per-plan deviation summary and the robust subset."""
    A = load_archive()
    B, C, _OV, tags = load_cross()
    at = A["tag"]
    tag_axes = tag_axes_table()

    sat = saturated_tags() if EXCLUDE_SATURATED else []
    keep = [j for j, t in enumerate(tags) if t not in sat]
    dropped_sat = [t for t in tags if t in sat]
    # A variant with no finished runs has no reference front of its own, so it cannot
    # contribute a deviation column; drop it rather than poisoning the row-wise max.
    keep = [j for j in keep if (at == tags[j]).any()]
    dropped_norun = [t for t in tags if t not in sat and not (at == t).any()]
    var_tags = [tags[j] for j in keep]
    print(f"Worst-case deviation over {len(var_tags)} variant(s) of "
          f"{len(tags)} evaluated.")
    if dropped_sat:
        print(f"  excluded as SATURATED (discrim): {', '.join(dropped_sat)}")
    if dropped_norun:
        print(f"  excluded (no runs of their own): {', '.join(dropped_norun)}")
    if not var_tags:
        raise SystemExit("No variant qualifies - nothing to classify.")

    R = np.zeros((len(C), len(keep)))
    R_own = np.zeros((len(C), len(keep)))
    for k, j in enumerate(keep):
        m = at == tags[j]
        R[:, k], _, _ = regret_against(front_reference(B[:, j], C), B[:, j], C)
        R_own[:, k], _, _ = regret_against(front_reference(B[m, j], C[m]), B[:, j], C)
    if REFERENCE_MODE == "own":
        R, R_own = R_own, R
    print(f"  reference: REFERENCE_MODE={REFERENCE_MODE!r}")

    max_reg = np.nanmax(R, axis=1)
    mean_reg = np.nanmean(R, axis=1)
    worst = [var_tags[i] for i in np.nanargmax(R, axis=1)]
    worst_axes = [_axes_departed(t, tag_axes) for t in worst]
    # How much the conclusion depends on that choice.
    alt_max = np.nanmax(R_own, axis=1)
    rho = pd.Series(max_reg).corr(pd.Series(alt_max), method="spearman")
    print(f"  worst-case regret under the other reference: Spearman rho = {rho:.3f} "
          f"({'ranking is robust to the choice' if rho > 0.8 else 'RANKING DEPENDS ON THE CHOICE'})")

    cutoff = float(np.quantile(max_reg, ROBUST_QUANTILE))
    robust = max_reg <= cutoff
    print(f"  worst-case regret: median {np.median(max_reg):.4f}, "
          f"p90 {np.quantile(max_reg, 0.9):.4f}, max {max_reg.max():.4f}")
    print(f"  robust subset = bottom {ROBUST_QUANTILE:.0%} of worst-case regret "
          f"(cutoff {cutoff:.4f}) -> {int(robust.sum())} of {len(max_reg)} plans")

    # Confound guard 1: if the robust subset is just the cheap tail, the frequency-map gap
    # in `maps` is a budget artefact rather than a robustness result.
    print("\n  cost distribution (is the robust subset just the cheap plans?)")
    for name, m in (("full archive", np.ones(len(C), bool)), ("robust subset", robust)):
        c = C[m]
        print(f"    {name:<14} median {np.median(c):.4g}  "
              f"IQR [{np.percentile(c, 25):.4g}, {np.percentile(c, 75):.4g}]")
    med_ratio = np.median(C[robust]) / max(np.median(C), 1e-30)
    if abs(med_ratio - 1.0) > 0.10:
        print(f"    WARNING: robust median cost is {med_ratio:.2f}x the archive median - "
              "read the frequency gap as partly a cost effect.")

    # Confound guard 2: the archive mixes two objective CONSTRUCTIONS (see CAMPAIGNS).
    # A robust subset drawn overwhelmingly from one of them is a family artefact, not a
    # robustness result, so the composition is reported next to the cost check.
    print("\n  composition (is the robust subset one family?)")
    comp = {}
    for camp_name in CAMPAIGNS:
        m = A["campaign"] == camp_name
        if not m.any():
            continue
        share_full = float(m.mean())
        share_rob = float((m & robust).sum() / max(robust.sum(), 1))
        comp[camp_name] = {"n_full": int(m.sum()), "share_full": share_full,
                           "n_robust": int((m & robust).sum()), "share_robust": share_rob,
                           "enrichment": share_rob / share_full if share_full else np.nan}
        flag = "" if abs(share_rob - share_full) <= 0.15 else "   <- OVER/UNDER-REPRESENTED"
        print(f"    {camp_name:<26} {share_full:>6.1%} of archive -> "
              f"{share_rob:>6.1%} of robust{flag}")

    # ...and the same question one axis at a time, other axes held at reference.
    comp_axis = _composition_by_axis(A, robust)
    for camp_name, per_axis in comp_axis.items():
        for axis, blk in per_axis.items():
            held = ", ".join(f"{o}={l}" for o, l in blk["held_at"].items()) or "nothing"
            print(f"\n  composition by {axis} ({camp_name}, holding {held}; "
                  f"{blk['n_full']} plans, {blk['n_robust']} robust)")
            for lvl, v in blk["levels"].items():
                mark = "  (reference)" if v["is_reference"] else ""
                print(f"    {lvl:<26} {v['share_full']:>6.1%} of archive -> "
                      f"{v['share_robust']:>6.1%} of robust   "
                      f"[{v['share_full_slice']:>5.1%} -> "
                      f"{v['share_robust_slice']:>5.1%} within axis]{mark}")

    print("\n  worst variant, by how often it is the binding case:")
    for t, n in pd.Series(worst).value_counts().head(12).items():
        print(f"    {t:<26} {n:>5}  ({n / len(worst):.1%})  [{_axes_departed(t, tag_axes)}]")
    print("\n  binding axis:")
    for t, n in pd.Series(worst_axes).value_counts().items():
        print(f"    {t:<26} {n:>5}  ({n / len(worst_axes):.1%})")

    df = pd.DataFrame({
        "plan_id": np.arange(len(C)), "native_campaign": A["campaign"],
        "native_tag": at, "native_seed": A["seed"],
        "n_cells": [pl.size for pl in A["plans"]], "dup_count": A["dup_count"],
        "cost": C, "benefit_native": A["native_benefit"],
        "max_regret": max_reg, "mean_regret": mean_reg,
        "worst_variant": worst, "worst_axes": worst_axes,
        "is_robust": robust, "max_regret_alt_reference": alt_max,
    })
    for a in AXIS_NAMES:
        df[f"axis_{a}"] = A["axes"][a]
    for k, t in enumerate(var_tags):
        df[f"regret__{t}"] = R[:, k]
    ensure(OUT_DIR)
    df.to_csv(OUT_DIR / "plan_summary.csv", index=False)
    print(f"\n  -> {OUT_DIR / 'plan_summary.csv'}")
    n_cells_plan = np.asarray([pl.size for pl in A["plans"]])
    report_update("classify", {
        "reference_mode": REFERENCE_MODE, "robust_quantile": ROBUST_QUANTILE,
        "n_plans": len(C), "variants": list(var_tags),
        "excluded_saturated": dropped_sat, "excluded_no_runs": dropped_norun,
        "spearman_rho_alt_reference": float(rho),
        "max_regret_median": float(np.median(max_reg)),
        "max_regret_p90": float(np.quantile(max_reg, 0.9)),
        "max_regret_max": float(max_reg.max()),
        "cutoff": cutoff, "n_robust": int(robust.sum()),
        "cost_full": {"median": float(np.median(C)),
                      "q25": float(np.percentile(C, 25)),
                      "q75": float(np.percentile(C, 75))},
        "cost_robust": {"median": float(np.median(C[robust])),
                        "q25": float(np.percentile(C[robust], 25)),
                        "q75": float(np.percentile(C[robust], 75))},
        "median_cost_ratio": float(med_ratio),
        "cost_confound_warning": bool(abs(med_ratio - 1.0) > 0.10),
        "composition": comp,
        "composition_by_axis": comp_axis,
        "composition_warning": bool(any(
            abs(v["share_robust"] - v["share_full"]) > 0.15 for v in comp.values())),
        "n_cells_full_median": float(np.median(n_cells_plan)),
        "n_cells_robust_median": float(np.median(n_cells_plan[robust])),
        "worst_variant_counts": {str(t): int(n)
                                 for t, n in pd.Series(worst).value_counts().items()},
        "worst_axis_counts": {str(t): int(n)
                              for t, n in pd.Series(worst_axes).value_counts().items()},
        "robust_by_native_tag": {str(t): {"n_robust": int(robust[at == t].sum()),
                                          "n_plans": int((at == t).sum())}
                                 for t in sorted(set(at.tolist()))},
        "regret_by_variant": {t: {"median": float(np.median(R[:, k])),
                                  "p90": float(np.quantile(R[:, k], 0.9)),
                                  "max": float(R[:, k].max()),
                                  "axes": _axes_departed(t, tag_axes)}
                              for k, t in enumerate(var_tags)},
    })


# ===========================================================================
# stage: maps (the result)
# ===========================================================================
def _frequency(plans, idx, n_cells):
    """Per-cell selection frequency (fraction of the given plans selecting each cell)."""
    if len(idx) == 0:
        return np.zeros(n_cells)
    counts = np.bincount(np.concatenate([plans[i] for i in idx]).astype(np.int64),
                         minlength=n_cells)
    return counts.astype(float) / len(idx)


def cmd_maps(argv=()):
    """Selection frequency: full archive vs robust subset. The gap is the result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualisations import crop_to_eligible

    A, L = load_archive(), load_layers()
    p = OUT_DIR / "plan_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the `classify` stage first.")
    # low_memory=False: the axis_* columns mix level names with the NOT_APPLICABLE
    # sentinel, which trips pandas' chunked dtype inference.
    summary = pd.read_csv(p, low_memory=False)
    robust = summary["is_robust"].to_numpy(bool)

    shape = L["shape"]
    n_cells = int(np.prod(shape))
    all_idx = np.arange(len(A["plans"]))
    freq_full = _frequency(A["plans"], all_idx, n_cells)
    freq_rob = _frequency(A["plans"], all_idx[robust], n_cells)
    delta = freq_rob - freq_full

    # Eligible footprint = union over variants, so no variant's dropped cells go missing.
    elig = np.zeros(n_cells, bool)
    for t in L["tags"]:
        elig |= L["elig"][t].ravel()
    elig_idx = np.flatnonzero(elig)

    core_full = np.flatnonzero(freq_full >= CORE_FREQ)
    core_rob = np.flatnonzero(freq_rob >= CORE_FREQ)
    gained = np.setdiff1d(core_rob, core_full)
    lost = np.setdiff1d(core_full, core_rob)
    print(f"Full archive: {len(all_idx)} plans | robust subset: {int(robust.sum())} plans")
    print(f"  cells ever selected      full {int((freq_full > 0).sum())}  "
          f"robust {int((freq_rob > 0).sum())}")
    print(f"  core (freq >= {CORE_FREQ:.2f})       full {core_full.size}  robust {core_rob.size}")
    print(f"  core Jaccard             {jaccard(core_full, core_rob):.4f}")
    print(f"  core cells gained/lost   +{gained.size} / -{lost.size}")
    print(f"  max |delta| frequency    {np.abs(delta).max():.4f}")

    # How much spatial consensus there is at all, so an empty core is read as a property
    # of the archive rather than as a broken threshold. Each plan holds ~5% of the
    # eligible landscape, so the mean frequency is ~0.05 by construction; what matters is
    # how far the upper tail rises above that.
    f_elig = freq_full[elig_idx]
    qs = [50, 75, 90, 99, 99.9, 100]
    print("  selection frequency over the eligible footprint (full archive): "
          + "  ".join(f"p{q:g}={np.percentile(f_elig, q):.3f}" for q in qs))
    for thr in (0.5, 0.75, 0.9):
        print(f"    cells at freq >= {thr:.2f}: full {int((freq_full >= thr).sum()):>7}"
              f"   robust {int((freq_rob >= thr).sum()):>7}")
    if core_full.size == 0:
        print(f"  NOTE: no cell reaches CORE_FREQ={CORE_FREQ:.2f} across the pooled")
        print("  archive, so 'core Jaccard' above is vacuous. The archive pools plans")
        print("  from every variant of every axis; consensus that strong is not present")
        print("  at this budget. Read the frequency surface and the delta map, and pick a")
        print("  CORE_FREQ from the percentiles above if a thresholded core is wanted.")

    ensure(OUT_DIR)
    rows, cols = np.unravel_index(elig_idx, shape)
    pd.DataFrame({"row": rows, "col": cols, "flat_index": elig_idx,
                  "freq_full": freq_full[elig_idx], "freq_robust": freq_rob[elig_idx],
                  "delta": delta[elig_idx]}).to_csv(
        OUT_DIR / "selection_frequency.csv", index=False)

    def panel(vec, ax, title, cmap, vmin, vmax):
        m = np.full(n_cells, np.nan)
        m[elig_idx] = vec[elig_idx]
        img = crop_to_eligible(m.reshape(shape), elig_idx, shape)
        h = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return h

    ensure(FIG_DIR)
    # The two frequency panels share one scale so they are directly comparable; the delta
    # panel gets a symmetric diverging scale about zero so sign reads off the hue.
    vmax = max(freq_full.max(), freq_rob.max())
    dmax = float(np.abs(delta[elig_idx]).max()) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    h0 = panel(freq_full, axes[0], f"full archive (n={len(all_idx)})", "YlOrRd", 0, vmax)
    panel(freq_rob, axes[1], f"robust subset (n={int(robust.sum())})", "YlOrRd", 0, vmax)
    h2 = panel(delta, axes[2], "delta (robust - full)", "RdBu_r", -dmax, dmax)
    fig.colorbar(h0, ax=axes[:2], location="bottom", fraction=0.045, pad=0.03,
                 aspect=45, label="selection frequency (shared scale)")
    fig.colorbar(h2, ax=axes[2], location="bottom", fraction=0.045, pad=0.03,
                 aspect=22, label="change in frequency")
    fig.suptitle("Where to restore: full archive vs the robustness-filtered subset", y=0.97)
    out_png = FIG_DIR / "selection_frequency_full_vs_robust.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_png}")

    # GeoTIFFs, for overlay in QGIS and for the manuscript figure (the R layer redraws
    # these rather than embedding the PNG above, which stays the analyst-facing diagnostic).
    tifs = {}
    try:
        import rasterio
        from rasterio.transform import Affine

        def _write_tif(vec, name):
            m = np.full(n_cells, np.nan, dtype=np.float32)
            m[elig_idx] = vec[elig_idx]
            path = OUT_DIR / name
            with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                               count=1, dtype="float32", crs=L["crs"] or None,
                               transform=Affine(*list(L["transform"])[:6]),
                               nodata=np.nan) as dst:
                dst.write(m.reshape(shape), 1)
            print(f"  -> {path}")
            return str(path)

        for vec, name in ((freq_full, "freq_full.tif"), (freq_rob, "freq_robust.tif"),
                          (delta, "delta_frequency.tif")):
            tifs[name] = _write_tif(vec, name)
    except Exception as e:  # noqa: BLE001 - the PNG and CSV are the deliverables
        print(f"  (GeoTIFF skipped: {e!r})")

    print(f"  -> {OUT_DIR / 'selection_frequency.csv'}")
    report_update("maps", {
        "core_freq": CORE_FREQ,
        "n_plans_full": len(all_idx), "n_plans_robust": int(robust.sum()),
        "cells_selected_full": int((freq_full > 0).sum()),
        "cells_selected_robust": int((freq_rob > 0).sum()),
        "core_full": int(core_full.size), "core_robust": int(core_rob.size),
        "core_jaccard": float(jaccard(core_full, core_rob)),
        "core_gained": int(gained.size), "core_lost": int(lost.size),
        "core_is_empty": bool(core_full.size == 0),
        "freq_percentiles_eligible": {f"p{q:g}": float(np.percentile(f_elig, q))
                                      for q in qs},
        "cells_by_freq_threshold": {
            f"{thr:.2f}": {"full": int((freq_full >= thr).sum()),
                           "robust": int((freq_rob >= thr).sum())}
            for thr in (0.5, 0.75, 0.9)},
        "max_abs_delta": float(np.abs(delta).max()),
        "n_eligible_cells": int(elig_idx.size),
        "figure_png": str(out_png), "rasters": tifs,
    })


# ===========================================================================
# dispatch
# ===========================================================================
def cmd_all(argv=()):
    for name in ("extract", "layers", "cross", "discrim", "noise", "interact",
                 "classify", "maps"):
        print(f"\n{'=' * 74}\n== {name}\n{'=' * 74}")
        COMMANDS[name](argv)


COMMANDS = {
    "extract": cmd_extract, "layers": cmd_layers, "cross": cmd_cross,
    "discrim": cmd_discrim, "noise": cmd_noise, "interact": cmd_interact,
    "classify": cmd_classify, "maps": cmd_maps, "all": cmd_all,
}


def main(argv):
    stage = argv[1] if len(argv) > 1 else None
    if stage not in COMMANDS:
        print(__doc__)
        print(f"Stages: {', '.join(COMMANDS)}")
        return
    COMMANDS[stage](argv[2:])


if __name__ == "__main__":
    main(sys.argv)
