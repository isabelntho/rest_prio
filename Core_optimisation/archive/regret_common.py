"""Helpers shared by the three regret legs (scenario_regret, allocation_regret,
robustness_synthesis): the stage-keyed report file and the robust-subset overlap stats.

Each leg owns one report.json with one key per stage plus a `config` key, so re-running
a single stage refreshes only that block - the convention uncertainty_analysis.py set
and paper2/_results_uncertainty.qmd reads.
"""
import json
import time

import numpy as np
from scipy.stats import hypergeom

from Core_optimisation.paths import ensure
from Core_optimisation.uncertainty_analysis import _jsonable


def write_report(report_json, stage, payload, config):
    """Merge one stage's headline numbers into report_json under its own key."""
    ensure(report_json.parent)
    doc = {}
    if report_json.exists():
        try:
            with open(report_json, encoding="ascii") as f:
                doc = json.load(f)
        except (ValueError, OSError) as e:
            print(f"  (report.json unreadable, starting fresh: {e!r})")
    doc[stage] = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    doc["config"] = config
    with open(report_json, "w", encoding="ascii") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=_jsonable)
    print(f"  report  -> {report_json} [{stage}]")


def overlap_stats(a, b, n):
    """How much two boolean plan subsets share, against what independence would give."""
    na, nb = int(a.sum()), int(b.sum())
    inter = int((a & b).sum())
    union = int((a | b).sum())
    exp = na * nb / n
    return {"n_a": na, "n_b": nb, "intersection": inter,
            "jaccard": inter / max(union, 1), "expected_by_chance": float(exp),
            "ratio_vs_chance": inter / max(exp, 1e-9),
            "hypergeom_p": float(hypergeom.sf(inter - 1, n, na, nb))}


def model_robust_mask(plan_summary_csv, n_plans):
    """The section-3.5 robust subset as a boolean over archive row order."""
    import pandas as pd

    ps = pd.read_csv(plan_summary_csv, usecols=["plan_id", "is_robust"])
    assert np.array_equal(ps["plan_id"].to_numpy(), np.arange(n_plans)), \
        "plan_summary.csv plan_id is not the archive row order"
    return (ps["is_robust"].astype(str) == "True").to_numpy()
