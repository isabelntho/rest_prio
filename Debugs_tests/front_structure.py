"""Front-structure diagnostic for one optimisation export.

    pixi run python Debugs_tests/front_structure.py <run-dir-or-label> [--sample N]

Reads an outputs/r_inputs/<run> directory and reports whether the run produced a real
Pareto front or a 1-D chain of near-identical plans:

  objectives     pairwise Pearson/Spearman + PC1 share of variance. PC1 near 1.0 means
                 the population lies on a single curve, on which EVERY smooth metric
                 correlates with every other by construction - so do not read the
                 correlations as evidence about which objective to use.
  core           pixels selected by all solutions, as a share of one plan
  jaccard        overlap between plans; the per-objective extremes are always included
  compactness    components / LPI / mesh / edge density - DESCRIPTIVE fragmentation
                 diagnostics, not evidence about which clustering metric to optimise
  frozen         objective extremes identical at generation 0 and the final generation,
                 i.e. a seed the search never beat

Values in objectives.csv are stored as pymoo minimises them, so maximised objectives
(restoration_benefit, spatial_clustering) appear negated. Correlations are reported on
the stored values.
"""
import os
import sys
import json
import csv
import array
import argparse

import numpy as np
from scipy import ndimage

from _common import REPO_ROOT

R_INPUTS_DIR = os.path.join(REPO_ROOT, "outputs", "r_inputs")


def resolve_run(name):
    """Accept a full path, a directory name under outputs/r_inputs, or 'newest'."""
    if os.path.isdir(name):
        return name
    cand = os.path.join(R_INPUTS_DIR, name)
    if os.path.isdir(cand):
        return cand
    if name == "newest":
        dirs = [os.path.join(R_INPUTS_DIR, d) for d in os.listdir(R_INPUTS_DIR)]
        dirs = [d for d in dirs if os.path.isdir(d)]
        if dirs:
            return max(dirs, key=os.path.getmtime)
    raise SystemExit(f"No such run: {name} (looked in {R_INPUTS_DIR})")


def read_objectives(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "objectives.csv"))))
    names = [k for k in rows[0] if k not in ("solution_id", "is_nondominated")]
    ids = np.array([int(r["solution_id"]) for r in rows])
    F = np.array([[float(r[n]) for n in names] for r in rows])
    nd = np.array([int(r.get("is_nondominated", 1)) for r in rows], dtype=bool)
    return ids, names, F, nd


def stream_selection(path, transform, width, want=None):
    """{solution_id: sorted int32 linear pixel indices} for restore actions.

    `want` limits parsing to a set of solution ids. Streamed with csv.reader because the
    CH export is ~400 MB / 12.5M rows.
    """
    x0, dx, y0, dy = transform[2], transform[0], transform[5], transform[4]
    out = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if row[1] != "restore":
                continue
            sid = int(row[0])
            if want is not None and sid not in want:
                continue
            col = int(round((float(row[2]) - x0 - dx / 2.0) / dx))
            rr = int(round((float(row[3]) - y0 - dy / 2.0) / dy))
            out.setdefault(sid, array.array("i")).append(rr * width + col)
    return {s: np.unique(np.frombuffer(a, dtype=np.int32)) for s, a in out.items()}


def spearman(a, b):
    def rank(v):
        o = np.argsort(v, kind="stable")
        rk = np.empty(len(v), dtype=float)
        rk[o] = np.arange(len(v))
        return rk
    return pearson(rank(a), rank(b))


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def compactness(lin, shape, patch_size):
    """Fragmentation panel for one plan, plus inter-patch adjacency and its ceiling."""
    h, w = shape
    g = np.zeros(h * w, dtype=bool)
    g[lin] = True
    g = g.reshape(h, w)
    n_px = int(g.sum())

    lab, n_comp = ndimage.label(g)                       # 4-connectivity
    sizes = np.bincount(lab.ravel())[1:].astype(np.float64)
    perim = (np.count_nonzero(g[:, :-1] != g[:, 1:])
             + np.count_nonzero(g[:-1, :] != g[1:, :]))

    # Patch ids are (row // PS, col // PS), so for orthogonal neighbours the ids differ
    # exactly when the shared coordinate crosses a patch boundary.
    cols_differ = (np.arange(w - 1) // patch_size) != (np.arange(1, w) // patch_size)
    rows_differ = (np.arange(h - 1) // patch_size) != (np.arange(1, h) // patch_size)
    inter = (np.count_nonzero((g[:, :-1] & g[:, 1:]) & cols_differ[None, :])
             + np.count_nonzero((g[:-1, :] & g[1:, :]) & rows_differ[:, None]))

    rows, cols = np.divmod(lin, w)
    n_patch = len(np.unique((rows // patch_size) * w + (cols // patch_size)))
    # A PS x PS tiling shares PS pixel-edges per internal patch side, 2 internal sides
    # per patch away from the boundary -> 2 * PS per patch, less a perimeter term.
    ceiling = 2 * patch_size * n_patch - 2 * patch_size * np.sqrt(n_patch)

    return dict(n_px=n_px, components=n_comp, lpi=float(sizes.max() / n_px),
                mesh=float((sizes ** 2).sum() / n_px), edge_density=perim / n_px,
                mean_comp=n_px / n_comp, inter_patch_adj=inter, n_patches=n_patch,
                inter_patch_ceiling=ceiling,
                inter_patch_frac=inter / ceiling if ceiling > 0 else float("nan"))


def report_objectives(names, F):
    print("== objectives ==")
    print("  %-22s %-22s %9s %9s" % ("", "", "pearson", "spearman"))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print("  %-22s %-22s %+9.4f %+9.4f"
                  % (names[i], names[j], pearson(F[:, i], F[:, j]),
                     spearman(F[:, i], F[:, j])))
    z = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1.0)
    ev = np.linalg.svd(z, compute_uv=False) ** 2
    share = ev / ev.sum()
    print("  variance by PC: " + "  ".join("%.4f" % s for s in share))
    if share[0] > 0.95:
        print("  -> PC1 %.1f%%: the population lies on a ~1-D chain. Correlations "
              "between ANY two" % (100 * share[0]))
        print("     smooth metrics are near-guaranteed here and say nothing about the "
              "metrics themselves.")


def report_frozen(run_dir, names):
    path = os.path.join(run_dir, "population_stats.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return
    first, last = rows[0], rows[-1]
    print("\n== objective extremes, generation %s vs %s =="
          % (first["generation"], last["generation"]))
    frozen = []
    for n in names:
        for kind in ("min", "max"):
            key = "%s_%s" % (n, kind)
            if key not in first:
                continue
            a, b = float(first[key]), float(last[key])
            tag = ""
            if abs(a - b) <= 1e-9 * max(1.0, abs(a)):
                tag = "   FROZEN"
                frozen.append(key)
            print("  %-28s %+12.6f -> %+12.6f%s" % (key, a, b, tag))
    if frozen:
        print("  -> %d extreme(s) never moved: held by an initial seed the search never"
              % len(frozen))
        print("     beat. Selection pressure on those objectives is absent, so any "
              "experiment")
        print("     about them is uninformative until the operators are fixed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="outputs/r_inputs dir, its name, or 'newest'")
    ap.add_argument("--sample", type=int, default=24,
                    help="solutions used for jaccard/compactness (0 = all)")
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    shape = tuple(meta["raster_info"]["shape"])
    transform = meta["raster_info"]["transform"]
    patch_size = int(meta["run_config"].get("patch_size") or 1)
    print("run: %s" % os.path.basename(run_dir))
    print("grid %dx%d  patch_size %d  solutions %s  reported non-dominated %s\n"
          % (shape[0], shape[1], patch_size, meta.get("n_solutions"),
             meta.get("n_nondominated_solutions")))

    ids, names, F, _nd = read_objectives(run_dir)
    report_objectives(names, F)
    report_frozen(run_dir, names)

    # Always include each objective's best and worst plan - the extremes are what the
    # acceptance criteria are stated on - then fill up to --sample with a spread.
    keep = set()
    for j in range(len(names)):
        keep.add(int(ids[int(np.argmin(F[:, j]))]))
        keep.add(int(ids[int(np.argmax(F[:, j]))]))
    if args.sample and len(ids) > args.sample:
        for s in ids[np.linspace(0, len(ids) - 1, args.sample).astype(int)]:
            keep.add(int(s))
        want = keep
    else:
        want = set(int(s) for s in ids)

    sel_path = os.path.join(run_dir, "pixel_selection.csv")
    print("\nreading %s (%.0f MB)..."
          % (os.path.basename(sel_path), os.path.getsize(sel_path) / 1e6))
    # The core needs every solution, so parse all of them and subset afterwards.
    sel = stream_selection(sel_path, transform, shape[1])

    counts = np.zeros(shape[0] * shape[1], dtype=np.int16)
    for lin in sel.values():
        counts[lin] += 1
    n_sol = len(sel)
    core = int(np.count_nonzero(counts == n_sol))
    per_plan = int(np.median([len(v) for v in sel.values()]))
    print("\n== shared core ==")
    print("  %d pixels selected by all %d solutions = %.1f%% of a plan (%d px)"
          % (core, n_sol, 100.0 * core / per_plan, per_plan))
    touched = int(np.count_nonzero(counts))
    print("  %d distinct pixels ever selected = %.2fx one plan" % (touched, touched / per_plan))

    sub = sorted(s for s in want if s in sel)
    print("\n== jaccard over %d solutions ==" % len(sub))
    pairs = []
    for i, a in enumerate(sub):
        for b in sub[i + 1:]:
            inter = np.intersect1d(sel[a], sel[b], assume_unique=True).size
            pairs.append((inter / (len(sel[a]) + len(sel[b]) - inter), a, b))
    if pairs:
        pairs.sort()
        med = pairs[len(pairs) // 2]
        print("  min    %.4f  (sol %d vs %d)" % pairs[0])
        print("  median %.4f  (sol %d vs %d)" % med)
        print("  max    %.4f  (sol %d vs %d)" % pairs[-1])
        print("  -> most distant pair on the front shares %.1f%% of its pixels"
              % (100 * pairs[0][0]))

    print("\n== compactness (descriptive; NOT evidence for a metric choice) ==")
    print("  %-6s %10s %8s %9s %9s %9s %10s %7s"
          % ("sol", "components", "lpi", "mesh", "edge_dens", "mean_comp",
             "inter_adj", "% ceil"))
    panel = {}
    for s in sub:
        m = compactness(sel[s], shape, patch_size)
        panel[s] = m
        print("  %-6d %10d %8.4f %9.1f %9.4f %9.2f %10d %6.1f%%"
              % (s, m["components"], m["lpi"], m["mesh"], m["edge_density"],
                 m["mean_comp"], m["inter_patch_adj"], 100 * m["inter_patch_frac"]))
    lpis = np.array([m["lpi"] for m in panel.values()])
    comps = np.array([m["components"] for m in panel.values()])
    print("  -> largest contiguous area holds %.1f-%.1f%% of restored land across "
          "%d-%d components" % (100 * lpis.min(), 100 * lpis.max(), comps.min(), comps.max()))


if __name__ == "__main__":
    sys.exit(main())
