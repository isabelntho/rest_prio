"""Spatial-assumption sweep: 12 settings, 26 GA runs, each with a simple-ranking baseline.

    pixi run python -m Core_optimisation.spatial_sweep [--workers 12] [--pixel-workers 3]
    pixi run python -m Core_optimisation.spatial_sweep --smoke        # 10 gens, pop 20, 1 seed
    pixi run python -m Core_optimisation.spatial_sweep --only A,B     # subset of settings
    pixi run python -m Core_optimisation.spatial_sweep --resume       # skip runs already ok
    pixi run python -m Core_optimisation.spatial_sweep report         # summary.csv + overlays

Questions
  Q1  Does spatial interaction make the GA necessary?     A (no spatial effects) vs B (baseline)
  Q2  Which spillover assumptions change the result?      C-H vs B
  Q3  Which patch rules change the result?                I-L vs B

Everything not listed in a setting is the nsga2_2obj preset (CH, restoration_benefit + cost,
NSGA-II pop 100) with: 150 generations and NO hypervolume early stopping (every run gets the same
budget), 5 % budget, pixel tolerance 0.01, warm start off, no min patch unless stated.

Output layout (outputs/spatial_sweep/, or outputs/spatial_sweep_smoke/):
  manifest.csv          one row per run: setting, parameters, status, minutes, peak memory
  driver.log            written by the launcher (see README in the plan) if redirected
  logs/<label>.log      each run's own line-buffered log
  runs/<label>/         that run's results_files/, summary_files/, r_inputs/, run_registry.jsonl,
                        ga_front.csv, ga_summary.json and simple_ranking/ (ranking_front.csv, plans.npz)
  summary.csv, overlays/    from `report`

The min-patch settings (K, L) run under region_grow because nothing repairs a patch-size floor on
the scattered path (the engine disables it there). See SETTINGS.

This module has NO import-time side effects: Windows 'spawn' workers re-import it.
"""
import argparse
import copy
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
from scipy import ndimage

from .data_loader import load_initial_conditions
from .grid_parallel import _invoke
from .paths import OUTPUTS
from .optimization_engine import RestorationProblem, initialize_patch_approach
from .run import _build_run_config, _grid_cfg, _load_preset, _resolve_runs

PRESET = "nsga2_2obj"
N_GENERATIONS = 150
POP_SIZE = 100
PIXEL_TOLERANCE = 0.01
MAX_FRACTION = 0.05
SMOKE_GENERATIONS = 10
SMOKE_POP = 20

# id, question, slug (used in the label), description, decision unit, radius, decay,
# spillover_to_restored, min patch size, sampling strategy (pixel mode only), seeds.
# cost = rough relative run time, used only to launch the slowest runs first.
_S = (101, 102)
SETTINGS = [
    dict(id="A", q=1, slug="pix-nospill", unit="pixel", tile=1, radius=0, decay=0.0, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=2,
         desc="No spatial effects: individual pixels, no spillover, no min patch"),
    dict(id="B", q=1, slug="base-2x2", unit="tile", tile=2, radius=3, decay=0.2, to_restored=False,
         min_patch=1, strategy="scattered", seeds=(101, 102, 103, 104), cost=1,
         desc="Baseline: 2x2 tiles, spillover radius 3 decay 0.2 to un-restored cells"),
    dict(id="C", q=2, slug="nospill", unit="tile", tile=2, radius=0, decay=0.0, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=1, desc="Spillover off"),
    dict(id="D", q=2, slug="r1", unit="tile", tile=2, radius=1, decay=0.2, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=1, desc="Spillover radius 1"),
    dict(id="E", q=2, slug="r5", unit="tile", tile=2, radius=5, decay=0.2, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=1, desc="Spillover radius 5"),
    dict(id="F", q=2, slug="decay0p1", unit="tile", tile=2, radius=3, decay=0.1, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=1, desc="Weaker spillover: decay 0.1"),
    dict(id="G", q=2, slug="decay0p5", unit="tile", tile=2, radius=3, decay=0.5, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=1, desc="Stronger spillover: decay 0.5"),
    dict(id="H", q=2, slug="spill-restored", unit="tile", tile=2, radius=3, decay=0.2, to_restored=True,
         min_patch=1, strategy="scattered", seeds=_S, cost=1,
         desc="Restored cells also benefit from restored neighbours (rewards clustering)"),
    dict(id="I", q=3, slug="pix", unit="pixel", tile=1, radius=3, decay=0.2, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=2, desc="Individual pixels instead of 2x2 tiles"),
    dict(id="J", q=3, slug="5x5", unit="tile", tile=5, radius=3, decay=0.2, to_restored=False,
         min_patch=1, strategy="scattered", seeds=_S, cost=1, desc="5x5 tiles"),
    dict(id="K", q=3, slug="pix-min9", unit="pixel", tile=1, radius=3, decay=0.2, to_restored=False,
         min_patch=9, strategy="region_grow", seeds=_S, cost=3,
         desc="Individual pixels, patches must be at least 9 pixels (region_grow operators)"),
    dict(id="L", q=3, slug="pix-min9-spill-restored", unit="pixel", tile=1, radius=3, decay=0.2,
         to_restored=True, min_patch=9, strategy="region_grow", seeds=_S, cost=3,
         desc="As K, with restored cells also benefiting from restored neighbours"),
]
QUESTIONS = {1: "Q1 spatial interaction vs simple ranking", 2: "Q2 spillover assumptions",
             3: "Q3 patch rules"}
MANIFEST_COLS = ["id", "question", "description", "unit", "tile", "radius", "decay", "to_restored",
                 "min_patch", "strategy", "seed", "label", "status", "minutes", "peak_rss_gb", "error"]


# ---------------------------------------------------------------------------
# task construction
# ---------------------------------------------------------------------------
def _label(s, seed):
    return f"{s['id']}_{s['slug']}_seed{seed}"


def _scenario_params(base, s, smoke):
    sp = copy.deepcopy(base)
    sp.update({
        "max_restoration_fraction": MAX_FRACTION,
        "neighbor_radius": s["radius"],
        "neighbor_effect_decay": s["decay"],
        "spillover_to_restored": s["to_restored"],
        "min_patch_size": s["min_patch"],
        "sampling_strategy": s["strategy"],
    })
    if s["strategy"] == "region_grow":
        # Neutral growth keeps the operators unbiased like the scattered runs' neutral repair;
        # seeds_min=1 was the best knob in the region-parameter sweep.
        sp["region_growth_bias"] = "neutral"
        sp["region_seeds_min"] = 1
    if smoke:
        sp["hv_warmup_samples"] = 20
    return sp


def _run_cfg(s, smoke, smoke_pop=SMOKE_POP):
    """The preset, overlaid with the sweep's shared settings and this setting's unit."""
    cfg = _load_preset(PRESET)
    n_gen = SMOKE_GENERATIONS if smoke else N_GENERATIONS
    cfg.update({
        "N_GENERATIONS": n_gen,
        "POP_SIZE": smoke_pop if smoke else POP_SIZE,
        "HV_PATIENCE": n_gen + 1,
        "WARM_SEEDING": False,
        "PIXEL_TOLERANCE": PIXEL_TOLERANCE,
        "USE_PATCH_APPROACH": s["unit"] == "tile",
        "PATCH_SIZE": s["tile"] if s["unit"] == "tile" else cfg["PATCH_SIZE"],
        "PATCH_CONSTRAINT_TYPE": "pixel_count",
        "SAVE_SNAPSHOTS": False,
        "CAPTURE_REPAIR_DIAG": False,
        "SCENARIO_MODE": "custom",
    })
    return cfg


def build_tasks(root, only=None, smoke=False, smoke_pop=SMOKE_POP):
    tasks = []
    for s in SETTINGS:
        if only and s["id"] not in only:
            continue
        cfg = _run_cfg(s, smoke, smoke_pop)
        sp = _scenario_params(cfg["custom_scenario_params"], s, smoke)
        cfg["custom_scenario_params"] = sp
        cfg["_run_config"] = _build_run_config(cfg)
        eco = _resolve_runs(cfg)[0][1]
        seeds = s["seeds"][:1] if smoke else s["seeds"]
        for seed in seeds:
            label = _label(s, seed)
            grid = _grid_cfg(cfg, eco, None)
            grid.update({"verbose": True, "output_dir": os.path.join(root, "runs", label)})
            tasks.append(dict(
                label=label, setting=s, seed=seed, tag=cfg["CONDITION_SCENARIO"], cfg=grid,
                scenario_params=sp, run_dir=os.path.join(root, "runs", label),
                log_dir=os.path.join(root, "logs"), unit=s["unit"], cost=s["cost"],
                run_config={**cfg["_run_config"], "condition_scenario": cfg["CONDITION_SCENARIO"],
                            "random_seed": seed, "sweep_id": s["id"], "sweep_question": s["q"],
                            "sweep_description": s["desc"], "spillover_to_restored": s["to_restored"],
                            "min_patch_size": s["min_patch"], "neighbor_radius": s["radius"],
                            "neighbor_effect_decay": s["decay"]},
            ))
    # longest first, so the tail of the schedule is short runs
    tasks.sort(key=lambda t: (-t["cost"], t["label"]))
    return tasks


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def _peak_rss_gb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2 ** 30, 2)
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def _mem(tag):
    """Log this process's memory at a phase boundary (peak = high-water mark so far)."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", info.rss)
        print(f"[mem] {tag}: rss {info.rss / 2 ** 30:.2f} GB, peak {peak / 2 ** 30:.2f} GB", flush=True)
    except Exception:  # noqa: BLE001 - diagnostics only
        pass


def _check_setting(problem, ic, task):
    """Fail in seconds, not after hours, if the engine is not running what the table says."""
    s, sp = task["setting"], task["scenario_params"]
    ep = problem.effect_params
    got = (ep["neighbor_radius"], ep["neighbor_effect_decay"],
           bool(ep.get("spillover_to_restored", False)))
    want = (s["radius"], s["decay"], s["to_restored"])
    assert got == want, f"{task['label']}: spillover (radius, decay, to_restored) {got} != {want}"
    assert problem.min_patch_size == s["min_patch"], (
        f"{task['label']}: min_patch_size {problem.min_patch_size} != {s['min_patch']} "
        "(the engine clamps or disables it - check sampling_strategy and the budget)")
    want_px = int(MAX_FRACTION * int(ic["n_restoration_pixels"]))
    assert problem.max_action_pixels == want_px, (
        f"{task['label']}: budget {problem.max_action_pixels} != {want_px}")
    if s["unit"] == "tile":
        sizes = [len(v) for v in ic["patch_mappings"]["restoration_patches"]["patch_to_pixels"].values()]
        assert ic["patch_size"] == s["tile"] and max(sizes) == s["tile"] ** 2, (
            f"{task['label']}: tiles are not {s['tile']}x{s['tile']} (patch_size={ic['patch_size']}, "
            f"largest tile {max(sizes)} px)")


def _save_ga_outputs(res, task, minutes):
    """Small per-run files so `report` never has to unpickle a multi-GB results file."""
    F = np.asarray(res["objectives_raw"], float)
    nd = np.asarray(res["is_nondominated"], bool)
    names = res["objective_names"]
    bi, ci = names.index("restoration_benefit"), names.index("implementation_cost")
    out = task["run_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ga_front.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benefit_raw", "cost", "nondominated"])
        for row, flag in zip(F, nd):
            w.writerow([repr(float(row[bi])), repr(float(row[ci])), int(flag)])
    ai = res["algorithm_info"]
    with open(os.path.join(out, "ga_summary.json"), "w") as f:
        json.dump({"label": task["label"], "n_solutions": int(res["n_solutions"]),
                   "n_nondominated": int(res["n_nondominated_solutions"]),
                   "actual_generations": ai["actual_generations"],
                   "converged_early": bool(ai["converged_early"]),
                   "max_action_pixels": int(res["problem_info"]["max_action_pixels"]),
                   "minutes": round(minutes, 2)}, f, indent=1)


def run_task(task):
    """One GA run + its ranking baseline, in this process. Returns a manifest update tuple."""
    from .simple_ranking import build_ranking_front

    s, cfg, label = task["setting"], task["cfg"], task["label"]
    t0 = time.perf_counter()
    os.makedirs(task["log_dir"], exist_ok=True)
    # line-buffered so a multi-hour run can be watched while it runs
    logf = open(os.path.join(task["log_dir"], f"{label}.log"), "w", buffering=1,
                encoding="ascii", errors="replace")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = logf
    try:
        try:
            ic = load_initial_conditions(
                ".", objectives=cfg["objectives"], region=cfg["region"], ecosystem=cfg["ecosystem"],
                sample_fraction=cfg["sample_fraction"], sample_seed=cfg["sample_seed"],
                aggregation_factor=cfg["aggregation_factor"], condition_scenario=task["tag"])
            _mem("data loaded")
            if s["unit"] == "tile":
                # Done here rather than inside the run so the tile geometry can be checked
                # first; run_optimization_instance skips this step when it is already done.
                ic = initialize_patch_approach(ic, patch_size=s["tile"])
            problem = RestorationProblem(ic, task["scenario_params"],
                                         pixel_tolerance=cfg["pixel_tolerance"])
            _check_setting(problem, ic, task)
            print(f"[setting ok] {label}: {s['desc']}")
            _mem("problem built")

            res = _invoke(ic, task["scenario_params"], label, task["run_config"], task["seed"], cfg)
            _mem("run + save finished")
            if res is None:
                return (label, False, (time.perf_counter() - t0) / 60, _peak_rss_gb(),
                        "run returned no front")
            _save_ga_outputs(res, task, (time.perf_counter() - t0) / 60)
            del res

            print("[ranking] building simple-ranking front")
            build_ranking_front(
                ic, task["scenario_params"], s["unit"] == "tile", s["tile"], cfg["pixel_tolerance"],
                os.path.join(task["run_dir"], "simple_ranking"), problem=problem)
            _mem("ranking done")
            return (label, True, (time.perf_counter() - t0) / 60, _peak_rss_gb(), None)
        except Exception as e:  # noqa: BLE001 - report, never crash the pool
            import traceback
            traceback.print_exc()
            return (label, False, (time.perf_counter() - t0) / 60, _peak_rss_gb(),
                    f"{type(e).__name__}: {e}")
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


# ---------------------------------------------------------------------------
# manifest + scheduling
# ---------------------------------------------------------------------------
def _manifest_rows(tasks):
    rows = []
    for t in tasks:
        s = t["setting"]
        rows.append(dict(id=s["id"], question=QUESTIONS[s["q"]], description=s["desc"], unit=s["unit"],
                         tile=s["tile"], radius=s["radius"], decay=s["decay"],
                         to_restored=s["to_restored"], min_patch=s["min_patch"],
                         strategy=s["strategy"] if s["unit"] == "pixel" else "tile",
                         seed=t["seed"], label=t["label"], status="pending", minutes="",
                         peak_rss_gb="", error=""))
    return rows


def _write_manifest(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _read_manifest(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["label"]: r for r in csv.DictReader(f)}


def schedule(tasks, workers, pixel_workers, on_start, on_done, gb_of=None, budget_gb=None):
    """Run tasks on a process pool. Memory, not cores, limits how many fit:
    - at most `pixel_workers` pixel-mode runs at once (their populations are ~3M variables wide);
    - if `gb_of` and `budget_gb` are given, a run only starts while the summed reservations of
      the running runs plus its own stay within the budget (one run is always allowed)."""
    pending, running = list(tasks), {}
    # One run per worker process: its peak memory is then that run's own, and the memory is
    # returned to the OS between runs instead of staying resident in a reused worker.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        while pending or running:
            n_pix = sum(1 for t in running.values() if t["unit"] == "pixel")
            reserved = sum(gb_of(t) for t in running.values()) if gb_of else 0.0
            i = 0
            while len(running) < workers and i < len(pending):
                t = pending[i]
                if t["unit"] == "pixel" and n_pix >= pixel_workers:
                    i += 1
                    continue
                if gb_of and budget_gb and running and reserved + gb_of(t) > budget_gb:
                    i += 1
                    continue
                pending.pop(i)
                if gb_of:
                    reserved += gb_of(t)
                running[ex.submit(run_task, t)] = t
                n_pix += t["unit"] == "pixel"
                on_start(t)
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                t = running.pop(fut)
                try:
                    on_done(t, fut.result())
                except Exception as e:  # noqa: BLE001 - a worker died (e.g. out of memory)
                    on_done(t, (t["label"], False, 0.0, None, f"worker crashed: {e!r}"))
                    if type(e).__name__ == "BrokenProcessPool":
                        for r in list(running.values()) + pending:
                            on_done(r, (r["label"], False, 0.0, None,
                                        "not run: process pool broke (re-launch with --resume)"))
                        return


def _stamp():
    return time.strftime("%H:%M:%S")


def cmd_run(args):
    root = str(OUTPUTS / ("spatial_sweep_smoke" if args.smoke else "spatial_sweep"))
    only = {x.strip().upper() for x in args.only.split(",")} if args.only else None
    tasks = build_tasks(root, only, args.smoke, args.smoke_pop)
    if not tasks:
        raise SystemExit(f"no settings match --only {args.only!r}")
    os.makedirs(root, exist_ok=True)

    path = os.path.join(root, "manifest.csv")
    prior = _read_manifest(path)
    fresh = {r["label"]: r for r in _manifest_rows(tasks)}
    rows = {lbl: prior.get(lbl, row) for lbl, row in fresh.items()}
    todo = [t for t in tasks if not (args.resume and rows[t["label"]]["status"] == "ok")]
    for t in todo:
        rows[t["label"]] = fresh[t["label"]]          # reset to pending
    # rows for settings outside --only are kept as they were
    ordered = [rows[t["label"]] for t in tasks] + [r for l, r in prior.items() if l not in rows]
    _write_manifest(path, ordered)

    def gb_of(t):
        return args.pixel_gb if t["unit"] == "pixel" else args.tile_gb

    print(f"{_stamp()} {len(todo)} of {len(tasks)} runs to launch; workers={args.workers} "
          f"(<= {args.pixel_workers} pixel-mode at once); memory reservation "
          f"{args.pixel_gb:g} GB/pixel run, {args.tile_gb:g} GB/tile run, budget {args.mem_budget_gb:g} GB; "
          f"output {root}", flush=True)

    def on_start(t):
        rows[t["label"]]["status"] = "running"
        _write_manifest(path, ordered)
        print(f"{_stamp()} start  {t['label']}", flush=True)

    def on_done(t, r):
        label, ok, minutes, rss, err = r
        row = rows[label]
        row.update(status="ok" if ok else "failed", minutes=f"{minutes:.1f}",
                   peak_rss_gb="" if rss is None else rss, error=(err or "")[:300])
        _write_manifest(path, ordered)
        print(f"{_stamp()} {'ok    ' if ok else 'FAILED'} {label} ({minutes:.1f} min"
              f"{'' if rss is None else f', peak {rss} GB'}){'' if ok else ': ' + str(err)}", flush=True)

    t0 = time.perf_counter()
    schedule(todo, args.workers, args.pixel_workers, on_start, on_done,
             gb_of=gb_of, budget_gb=args.mem_budget_gb)
    n_ok = sum(1 for t in todo if rows[t["label"]]["status"] == "ok")
    print(f"{_stamp()} done: {n_ok}/{len(todo)} ok in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)
    if args.smoke:
        _smoke_projection(root, todo, args.smoke_pop)
    return 0 if n_ok == len(todo) else 1


def _smoke_projection(root, tasks, smoke_pop=SMOKE_POP):
    """Per-setting seconds/generation from the smoke logs, scaled to the real run."""
    scale = (POP_SIZE / smoke_pop) * (N_GENERATIONS / SMOKE_GENERATIONS)
    print("\nSmoke timing (evaluation cost scales with population, so x pop ratio; assumes the same "
          "concurrency as the real launch):")
    print(f"  {'label':38s} {'gen10 min':>9s} {'proj. 150 gens @ pop100 (h)':>28s}")
    for t in sorted(tasks, key=lambda t: t["label"]):
        log = os.path.join(root, "logs", f"{t['label']}.log")
        mins = None
        if os.path.exists(log):
            for m in re.finditer(r"Generation 10/\d+.*?Elapsed: ([\d.]+)min", open(log).read()):
                mins = float(m.group(1))
        print(f"  {t['label']:38s} {'?' if mins is None else f'{mins:.1f}':>9s} "
              f"{'?' if mins is None else f'{mins * scale / 60:.1f}':>28s}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
# Settings whose pooled fronts get cross-evaluated against B (Q2, the spillover question) -
# B itself is the reference every one of them is compared to, both directions.
CROSS_EVAL_IDS = {"B", "C", "D", "E", "F", "G", "H"}


def _load_front(run_dir):
    import pandas as pd
    ga = pd.read_csv(os.path.join(run_dir, "ga_front.csv"))
    rk = pd.read_csv(os.path.join(run_dir, "simple_ranking", "ranking_front.csv"))
    return ga, rk


def _export_dir(root, label):
    """The single timestamped r_inputs export directory a finished run wrote."""
    base = os.path.join(root, "runs", label, "r_inputs")
    subs = [os.path.join(base, d) for d in (os.listdir(base) if os.path.isdir(base) else [])]
    subs = [d for d in subs if os.path.isdir(d)]
    if len(subs) != 1:
        raise FileNotFoundError(f"expected one r_inputs export under {base}, found {len(subs)}")
    return subs[0]


def _plans_for_ids(export_dir, want_ids):
    """{solution_id: sorted int64 flat pixel index array} for restore actions, streamed from
    pixel_selection.csv (CH-scale exports are hundreds of MB - too big to load whole). Also
    returns the raster shape, read from this export's own metadata.json."""
    meta = json.load(open(os.path.join(export_dir, "metadata.json")))
    shape = tuple(meta["raster_info"]["shape"])
    tr = meta["raster_info"]["transform"]
    x0, dx, y0, dy = tr[2], tr[0], tr[5], tr[4]
    width = shape[1]
    want = {int(w) for w in want_ids}
    buf = {}
    with open(os.path.join(export_dir, "pixel_selection.csv"), newline="") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if row[1] != "restore":
                continue
            sid = int(row[0])
            if sid not in want:
                continue
            col = int(round((float(row[2]) - x0 - dx / 2.0) / dx))
            rr = int(round((float(row[3]) - y0 - dy / 2.0) / dy))
            buf.setdefault(sid, []).append(rr * width + col)
    return {s: np.array(sorted(v), dtype=np.int64) for s, v in buf.items()}, shape


def _patch_geometry(idx, shape):
    """(n_patches, mean_patch_size_ha) of one plan - CH pixels are 100 m, so 1 px = 1 ha."""
    board = np.zeros(shape, dtype=bool)
    board.flat[idx] = True
    n = int(ndimage.label(board)[1])
    return n, (len(idx) / n if n else 0.0)


def _gap_stats(ref_pts, pts):
    """Cost-matched benefit shortfall of `pts` against the front through `ref_pts`: (mean, max)."""
    from .uncertainty_analysis import front_reference, regret_against
    ref = front_reference(ref_pts[:, 0], ref_pts[:, 1])
    reg, _, unc = regret_against(ref, pts[:, 0], pts[:, 1])
    vals = reg[~unc]
    return (float(np.mean(vals)), float(np.max(vals))) if vals.size else (float("nan"), float("nan"))


def _write_extra_analysis(root, summary_pooled, pooled_ids):
    """patch_geometry.csv (all settings), cross_eval.csv + overlap_with_b.csv (vs B).

    Pixel selections are read from each LABEL's pixel_selection.csv exactly once (one
    `_plans_for_ids` call per file, restricted to that file's pooled-non-dominated ids), then
    reused for whichever of patch geometry / cross-eval / overlap that setting needs.
    """
    import pandas as pd
    from .uncertainty_analysis import jaccard

    geom_rows, cross_rows, overlap_rows = [], [], []
    cross_settings = {s["id"]: s for s in SETTINGS if s["id"] in CROSS_EVAL_IDS}
    state = {"ic": None, "problems": {}}

    def problem_for(sid):
        if state["ic"] is None:
            print("  loading initial_conditions for cross-evaluation ...", flush=True)
            ic = load_initial_conditions(
                ".", objectives=["restoration_benefit", "cost"], region="CH", ecosystem="all",
                sample_fraction=None, sample_seed=42, aggregation_factor=None,
                condition_scenario="global_all")
            state["ic"] = ic
            # pixel_selection.csv gives RASTER-flat indices; evaluate_raw_objectives needs a
            # position in x_restore (restoration_eligible_indices order) - map one to the other.
            n_rest = int(ic["n_restoration_pixels"])
            lut = np.full(int(np.prod(ic["shape"])), -1, dtype=np.int64)
            lut[np.asarray(ic["restoration_eligible_indices"], np.int64)] = np.arange(n_rest)
            state["lut"] = lut
        if sid not in state["problems"]:
            s = cross_settings[sid]
            sp = {"max_restoration_fraction": MAX_FRACTION, "abiotic_effect": 0.01,
                  "biotic_effect": 0.01, "normalize_objectives": True, "min_patch_size": 1,
                  "sampling_strategy": "scattered", "neighbor_radius": s["radius"],
                  "neighbor_effect_decay": s["decay"], "spillover_to_restored": s["to_restored"]}
            state["problems"][sid] = RestorationProblem(state["ic"], sp)
        return state["problems"][sid]

    def score(problem, raster_idx):
        n_rest = int(state["ic"]["n_restoration_pixels"])
        n_conv = int(state["ic"]["n_conversion_pixels"])
        pos = state["lut"][raster_idx]
        assert (pos >= 0).all(), "a selected pixel is not restoration-eligible in this ic"
        x = np.zeros(n_rest + n_conv, dtype=int)
        x[pos] = 1
        raw = problem.evaluate_raw_objectives(x)
        bi = problem.objective_names.index("restoration_benefit")
        ci = problem.objective_names.index("implementation_cost")
        return float(raw[bi]), float(raw[ci])

    b_plans, b_repr = None, None   # b_repr = (key, idx, cost) - B's median-cost pooled-ND plan
    for sid in sorted(summary_pooled):
        ids = pooled_ids.get(sid, [])
        if not ids:
            continue
        G = summary_pooled[sid][0]
        known = dict(zip(ids, G))   # (label, solution_id) -> (benefit_raw, cost), for the sanity check

        by_label = {}
        for label, s in ids:
            by_label.setdefault(label, []).append(s)
        plans, shape = {}, None
        for label, want in by_label.items():
            got, shape = _plans_for_ids(_export_dir(root, label), want)
            for local_sid, idx in got.items():
                plans[(label, local_sid)] = idx
        if not plans:
            print(f"  WARNING: no plans loaded for {sid}, skipping")
            continue
        if sid == "B":
            b_plans = plans

        ns, sizes = zip(*(_patch_geometry(idx, shape) for idx in plans.values()))
        geom_rows.append(dict(id=sid, n_plans=len(plans), n_patches_mean=float(np.mean(ns)),
                              mean_patch_size_ha_mean=float(np.mean(sizes))))

        costs = G[:, 1]
        j = int(np.argmin(np.abs(costs - np.median(costs))))
        rep_key, rep_cost = ids[j], float(costs[j])
        rep_idx = plans[rep_key]
        if sid == "B":
            b_repr = (rep_key, rep_idx, rep_cost)
        elif b_repr is not None:
            overlap_rows.append(dict(id=sid, jaccard=jaccard(b_repr[1], rep_idx),
                                     cost_this=rep_cost, cost_b=b_repr[2]))

        if sid in cross_settings and sid != "B" and b_plans is not None:
            problem_b, problem_x = problem_for("B"), problem_for(sid)

            def cross_rows_for(items, native_problem, cross_problem, direction):
                for key, idx in items:
                    b_nat, c_nat = score(native_problem, idx)
                    b_cr, c_cr = score(cross_problem, idx)
                    if key in known:
                        kb = float(known[key][0])
                        rel = abs(b_nat - kb) / max(abs(kb), 1e-9)
                        assert rel < 1e-3, (
                            f"cross_eval own-model mismatch for {key}: {b_nat} vs ga_front.csv's "
                            f"{kb} (rel {rel:.2e})")
                    cross_rows.append(dict(setting=sid, direction=direction, label=key[0],
                                           solution_id=key[1], benefit_native=b_nat,
                                           cost_native=c_nat, benefit_cross=b_cr, cost_cross=c_cr))

            cross_rows_for(plans.items(), problem_x, problem_b, f"{sid}_plan_under_B_model")
            cross_rows_for(b_plans.items(), problem_b, problem_x, f"B_plan_under_{sid}_model")

    pd.DataFrame(geom_rows).to_csv(os.path.join(root, "patch_geometry.csv"), index=False)
    print(f"wrote {os.path.join(root, 'patch_geometry.csv')}")
    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(os.path.join(root, "cross_eval.csv"), index=False)
        print(f"wrote {os.path.join(root, 'cross_eval.csv')}")
    if overlap_rows:
        pd.DataFrame(overlap_rows).to_csv(os.path.join(root, "overlap_with_b.csv"), index=False)
        print(f"wrote {os.path.join(root, 'overlap_with_b.csv')}")


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from .uncertainty_analysis import nondominated_2d

    root = str(OUTPUTS / ("spatial_sweep_smoke" if args.smoke else "spatial_sweep"))
    manifest = _read_manifest(os.path.join(root, "manifest.csv"))
    ok = [r for r in manifest.values() if r["status"] == "ok"]
    if not ok:
        raise SystemExit(f"no finished runs in {root}/manifest.csv")

    rows, pooled = [], {}
    for r in ok:
        run_dir = os.path.join(root, "runs", r["label"])
        try:
            ga, rk = _load_front(run_dir)
        except FileNotFoundError as e:
            print(f"skip {r['label']}: {e}")
            continue
        nd = ga["nondominated"] == 1
        g = ga.loc[nd, ["benefit_raw", "cost"]].to_numpy()
        g_ids = [(r["label"], int(sid)) for sid in ga.index[nd]]
        k = rk[["benefit_raw", "cost"]].to_numpy()
        summ = json.load(open(os.path.join(run_dir, "ga_summary.json")))
        rk_mean, rk_max = _gap_stats(g, k)
        ga_mean, ga_max = _gap_stats(k, g)
        rows.append(dict(level="run", id=r["id"], label=r["label"], seed=r["seed"],
                         description=r["description"], n_ga_nd=len(g), n_rank_nd=int(rk["nondominated"].sum()),
                         generations=summ["actual_generations"], minutes=r["minutes"],
                         ranking_shortfall_vs_ga=rk_mean, ranking_shortfall_vs_ga_max=rk_max,
                         ga_shortfall_vs_ranking=ga_mean, ga_shortfall_vs_ranking_max=ga_max))
        p = pooled.setdefault(r["id"], dict(description=r["description"], ga=[], rk=[], ga_ids=[]))
        p["ga"].append(g)
        p["rk"].append(k)
        p["ga_ids"].extend(g_ids)

    summary_pooled, pooled_ids = {}, {}
    for sid, p in sorted(pooled.items()):
        # seeds are pooled per setting: a seed is a replicate, not a separate scenario
        G, K = np.vstack(p["ga"]), np.vstack(p["rk"])
        mask = nondominated_2d(G)
        G, ids = G[mask], [x for x, m in zip(p["ga_ids"], mask) if m]
        K = K[nondominated_2d(K)]
        summary_pooled[sid] = (G, K, p["description"])
        pooled_ids[sid] = ids
        rk_mean, rk_max = _gap_stats(G, K)
        ga_mean, ga_max = _gap_stats(K, G)
        rows.append(dict(level="setting_pooled", id=sid, label="", seed="", description=p["description"],
                         n_ga_nd=len(G), n_rank_nd=len(K), generations="", minutes="",
                         ranking_shortfall_vs_ga=rk_mean, ranking_shortfall_vs_ga_max=rk_max,
                         ga_shortfall_vs_ranking=ga_mean, ga_shortfall_vs_ranking_max=ga_max))
    pd.DataFrame(rows).to_csv(os.path.join(root, "summary.csv"), index=False)
    print(f"wrote {os.path.join(root, 'summary.csv')}")

    _write_extra_analysis(root, summary_pooled, pooled_ids)

    os.makedirs(os.path.join(root, "overlays"), exist_ok=True)
    for q, title in QUESTIONS.items():
        ids = [s["id"] for s in SETTINGS if s["q"] == q and s["id"] in summary_pooled]
        if q > 1 and "B" in summary_pooled:
            ids = ["B"] + [i for i in ids if i != "B"]
        if not ids:
            continue
        ncol = min(3, len(ids))
        nrow = int(np.ceil(len(ids) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
        for ax, sid in zip(axes.ravel(), ids):
            G, K, desc = summary_pooled[sid]
            ax.plot(G[:, 1], -G[:, 0], "o", ms=3, color="#1f77b4", label="GA (pooled seeds)")
            o = np.argsort(K[:, 1])
            ax.plot(K[o, 1], -K[o, 0], "-s", ms=4, color="#ff7f0e", label="simple ranking")
            ax.set_title(f"{sid}: {desc[:44]}", fontsize=8)
            ax.set_xlabel("cost")
            ax.set_ylabel("benefit")
        for ax in axes.ravel()[len(ids):]:
            ax.axis("off")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(root, "overlays", f"Q{q}.png"), dpi=130)
        plt.close(fig)
    print(f"wrote overlays to {os.path.join(root, 'overlays')}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m Core_optimisation.spatial_sweep")
    p.add_argument("command", nargs="?", default="run", choices=["run", "report"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--pixel-workers", type=int, default=2,
                   help="max concurrent pixel-mode runs (memory-bound)")
    p.add_argument("--mem-budget-gb", type=float, default=42.0,
                   help="total memory the running runs may reserve (64 GB machine, ~46 GB free "
                        "with nothing else heavy running)")
    p.add_argument("--pixel-gb", type=float, default=17.0,
                   help="memory reserved per pixel-mode run (measured peak 16.3 GB at pop 100, "
                        "in the GA loop: populations of ~3M int64 variables)")
    p.add_argument("--tile-gb", type=float, default=9.0,
                   help="memory reserved per tile-mode run (measured peak 8.4 GB for 2x2 at pop 100)")
    p.add_argument("--smoke-pop", type=int, default=SMOKE_POP,
                   help="population size in --smoke mode (100 = the real size, for a memory probe)")
    p.add_argument("--only", help="comma-separated setting ids, e.g. A,B")
    p.add_argument("--resume", action="store_true", help="skip runs the manifest marks ok")
    p.add_argument("--smoke", action="store_true",
                   help=f"{SMOKE_GENERATIONS} gens, pop {SMOKE_POP}, first seed only, separate folder")
    args = p.parse_args(argv)
    return cmd_report(args) if args.command == "report" else cmd_run(args)


# REQUIRED on Windows: 'spawn' re-imports this module in every worker.
if __name__ == "__main__":
    sys.exit(main())
