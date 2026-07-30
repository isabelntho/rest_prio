"""Objective-redundancy analyses: is a spatial objective actually a separate axis?

Subcommands (pixi run python Debugs_tests/redundancy_checks.py <cmd>):

  clustering   Is spatial clustering a redundant objective? Measures, on the exact input
               layers: (1) Moran's I of the cost layer vs the restoration-potential layer
               (how spatially clumped each is), and (2) the cherry-pick gap = how much
               objective value contiguity costs, plus how clustered each objective's OWN
               optimum already is. No optimisation is re-run.

  adjacency    Are plain adjacency and inter_patch_adjacency effectively the same axis? On
               the stored inter_patch_adjacency patch run, recomputes plain adjacency from
               each solution's pixel mask, checks internal edges ~= 4*n_patches, and reports
               the correlations of plain/inter with cost.
"""
import os
import sys
import pickle
import numpy as np

from _common import REPO_ROOT, RESULTS_DIR, load_ic


# ===========================================================================
# clustering  (was clustering_redundancy.py)
# ===========================================================================
# Loader short-names (see data_loader all_objectives): 'cost' -> implementation_cost.
# spatial_clustering needs no raster, so it is not requested here.
CLUST_OBJ = ["restoration_potential", "cost"]
CLUST_MAX_FRAC = 0.05
CLUST_N_TRIALS = 20      # random-seed contiguous grows per objective; best is kept
CLUST_SEED = 101


def _morans_i(z_full, nbr):
    """Rook-contiguity Moran's I over eligible pixels. z_full is centred values."""
    valid = nbr >= 0
    zi = z_full[:, None]                      # (n,1) broadcast against 4 neighbours
    zj = np.where(valid, z_full[nbr], 0.0)    # neighbour values, 0 where no neighbour
    cross = float(np.sum(np.where(valid, zi * zj, 0.0)))
    s0 = float(valid.sum())                   # sum of binary weights
    denom = float(np.sum(z_full * z_full))
    n = z_full.size
    return (n / s0) * (cross / denom)


def _cluster_metrics_ndimage(sel, nbr, elig, shape):
    """How clustered is a selection? Returns (adjacency, n_components).

    adjacency  = shared orthogonal edges between selected pixels.
    n_components = number of disconnected 4-connected clusters (fewer = more clumped).
    """
    from scipy import ndimage
    sel_nb = np.where(nbr >= 0, sel[np.clip(nbr, 0, None)], False)
    sel_nb &= (nbr >= 0)
    adjacency = int(np.sum(sel_nb[sel]) // 2)   # undirected shared edges
    mask2d = np.zeros(shape, dtype=bool)
    idx = elig[sel]
    r, c = np.divmod(idx, shape[1])
    mask2d[r, c] = True
    _, n_comp = ndimage.label(mask2d)
    return adjacency, int(n_comp)


def _best_contiguous(scores, layer, nbr, n_rest, k, n_trials, rng):
    """Grow a single contiguous blob toward HIGH score; return the lowest layer-sum
    achieved over n_trials random seeds (heuristic best-case contiguous plan)."""
    from Core_optimisation.spatial_operations import grow_region_plan
    best = np.inf
    for _ in range(n_trials):
        sel = grow_region_plan(nbr, n_rest, k, n_seeds=1, scores=scores,
                               mode="scored", rng=rng)
        s = float(layer[sel].sum())
        if s < best:
            best = s
    return best


def cmd_clustering():
    from Core_optimisation.spatial_operations import build_restoration_neighbor_table, grow_region_plan

    print("Loading initial conditions (region=Bern, ecosystem=all, global_all)...")
    ic = load_ic(CLUST_OBJ)

    elig = np.asarray(ic["restoration_eligible_indices"], dtype=np.int64)
    n_rest = elig.size
    k = max(int(CLUST_MAX_FRAC * n_rest), 1)

    rp = np.asarray(ic["restoration_potential_1d"], dtype=np.float64)
    cost2d = np.asarray(ic["implementation_cost"], dtype=np.float64)
    cost = cost2d.ravel()[elig]

    print(f"eligible pixels n_rest={n_rest}   budget k={k} ({CLUST_MAX_FRAC*100:.0f}%)")
    print(f"rp:   mean={rp.mean():.4f}  sd={rp.std():.4f}  (lower = more degraded = higher potential)")
    print(f"cost: mean={cost.mean():.1f}  sd={cost.std():.1f}")

    nbr, rows, cols = build_restoration_neighbor_table(ic)

    # ---- 1. Moran's I (spatial autocorrelation) --------------------------------
    I_rp = _morans_i(rp - rp.mean(), nbr)
    I_cost = _morans_i(cost - cost.mean(), nbr)
    print("\n=== 1. Moran's I (rook, over eligible pixels) ===")
    print(f"  restoration_potential : I = {I_rp:+.3f}")
    print(f"  implementation_cost   : I = {I_cost:+.3f}")
    print("  (near 0 = no spatial pattern; near 1 = strongly clumped)")

    # ---- 2. Cherry-pick gap ----------------------------------------------------
    rng = np.random.default_rng(CLUST_SEED)

    scat_rp = float(np.sort(rp)[:k].sum())          # N most-degraded (min rp)
    scat_cost = float(np.sort(cost)[:k].sum())      # N cheapest (min cost)

    rand_rp = k * float(rp.mean())
    rand_cost = k * float(cost.mean())

    cont_rp = _best_contiguous(-rp, rp, nbr, n_rest, k, CLUST_N_TRIALS, rng)     # grow to low rp
    cont_cost = _best_contiguous(-cost, cost, nbr, n_rest, k, CLUST_N_TRIALS, rng)  # grow to low cost

    gap_rp = (cont_rp - scat_rp) / (rand_rp - scat_rp)
    gap_cost = (cont_cost - scat_cost) / (rand_cost - scat_cost)

    print("\n=== 2. Cherry-pick gap (fixed budget = %d pixels) ===" % k)
    print("  restoration_potential (objective sum, lower=better):")
    print(f"    scattered optimum : {scat_rp:12.1f}")
    print(f"    best contiguous   : {cont_rp:12.1f}")
    print(f"    random baseline   : {rand_rp:12.1f}")
    print(f"    GAP               : {gap_rp*100:6.1f}%  (0=contiguity free, 100=all cherry-pick lost)")
    print("  implementation_cost (lower=better):")
    print(f"    scattered optimum : {scat_cost:12.1f}")
    print(f"    best contiguous   : {cont_cost:12.1f}")
    print(f"    random baseline   : {rand_cost:12.1f}")
    print(f"    GAP               : {gap_cost*100:6.1f}%  (0=contiguity free, 100=all cherry-pick lost)")

    # ---- 3. How clustered is each objective's OWN optimum? ---------------------
    shape = tuple(ic["shape"])
    best_rp_sel = np.zeros(n_rest, dtype=bool)
    best_rp_sel[np.argsort(rp)[:k]] = True             # N most-degraded pixels
    best_cost_sel = np.zeros(n_rest, dtype=bool)
    best_cost_sel[np.argsort(cost)[:k]] = True         # N cheapest pixels
    rand_sel = np.zeros(n_rest, dtype=bool)
    rand_sel[rng.choice(n_rest, size=k, replace=False)] = True
    blob_sel = grow_region_plan(nbr, n_rest, k, n_seeds=1,
                                scores=np.zeros(n_rest), mode="neutral", rng=rng)

    max_adj = 2 * k  # loose upper bound on shared edges for k pixels
    print("\n=== 3. Each canonical selection scored on ALL axes (budget k=%d) ===" % k)
    print(f"  {'selection':<24}{'rp_sum':>11}{'cost_sum':>11}{'adjacency':>11}{'%adj':>7}{'comps':>9}")
    for name, sel in [("best-potential set", best_rp_sel),
                      ("cheapest-cost set", best_cost_sel),
                      ("random scatter", rand_sel),
                      ("contiguous blob (ref)", blob_sel)]:
        adj, ncomp = _cluster_metrics_ndimage(sel, nbr, elig, shape)
        rp_s = float(rp[sel].sum())
        cost_s = float(cost[sel].sum())
        print(f"  {name:<24}{rp_s:>11.0f}{cost_s:>11.0f}{adj:>11}{adj/max_adj*100:>6.1f}%{ncomp:>9}")
    print("  (rp_sum and cost_sum: lower = better on that objective)")

    print("\n=== Reading ===")
    print("  Moran's I: how spatially clumped each input layer is.")
    print("  Cherry-pick gap: how much objective value contiguity costs.")
    print("  Section 3: whether an objective's OWN optimum is already clustered")
    print("             (high adjacency / few components) = redundant with the")
    print("             clustering objective; scattered (low adj / many comps) =")
    print("             genuinely independent of it.")


# ===========================================================================
# adjacency  (was adjacency_metric_equivalence.py)
# ===========================================================================
ADJ_PKL = os.path.join(RESULTS_DIR, "res_20260722_1147_inter_patch_adjacency.pkl")


def _plain_adjacency(mask2d):
    """Orthogonal shared edges among selected pixels (compactness)."""
    h = np.sum(mask2d[:, :-1] & mask2d[:, 1:])
    v = np.sum(mask2d[:-1, :] & mask2d[1:, :])
    return int(h + v)


def cmd_adjacency():
    from visualisations import _convert_patch_decisions_to_pixel_matrix

    with open(ADJ_PKL, "rb") as f:
        res = pickle.load(f)

    names = list(res["objective_names"])
    raw = np.asarray(res["objectives_raw"], dtype=np.float64)
    dec = np.asarray(res["decisions"])
    ic = res["initial_conditions"]
    shape = tuple(ic["shape"])
    elig = np.asarray(ic["restoration_eligible_indices"], dtype=np.int64)
    n_rest = elig.size
    print(f"solutions: {dec.shape[0]}   patch-decision vars: {dec.shape[1]}   objectives: {names}")

    ci = names.index("spatial_clustering")
    ki = names.index("implementation_cost") if "implementation_cost" in names else names.index("cost")
    # spatial_clustering objective is negated (minimiser maximises inter-patch contact).
    inter_stored = -raw[:, ci]
    cost = raw[:, ki]

    pix = _convert_patch_decisions_to_pixel_matrix(dec, ic, results=res)
    if pix is None:
        print("ERROR: could not convert patch decisions to pixels.")
        return
    pix = np.asarray(pix)[:, :n_rest].astype(bool)
    n_patches = dec[:, :dec.shape[1]].sum(axis=1).astype(float)  # selected patches per solution

    rows, cols = np.divmod(elig, shape[1])
    plain = np.zeros(dec.shape[0])
    for i in range(dec.shape[0]):
        m = np.zeros(shape, dtype=bool)
        sel = pix[i]
        m[rows[sel], cols[sel]] = True
        plain[i] = _plain_adjacency(m)

    internal = plain - inter_stored  # should be ~ 4 * n_patches

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    print("\n=== ranges across solutions ===")
    print(f"  plain adjacency        : {plain.min():.0f} .. {plain.max():.0f}")
    print(f"  inter_patch adjacency  : {inter_stored.min():.0f} .. {inter_stored.max():.0f}")
    print(f"  cost                   : {cost.min():.1f} .. {cost.max():.1f}")
    print(f"  n_patches selected     : {n_patches.min():.0f} .. {n_patches.max():.0f}"
          f"  (CV={n_patches.std()/n_patches.mean()*100:.1f}%)")

    print("\n=== affine-equivalence check: internal edges vs 4*n_patches ===")
    print(f"  internal = plain - inter : mean={internal.mean():.0f}  sd={internal.std():.0f}")
    print(f"  internal / n_patches     : mean={np.mean(internal/np.maximum(n_patches,1)):.3f}  (expect ~4.0)")

    print("\n=== correlations across solutions ===")
    print(f"  corr(plain, inter_patch)       = {corr(plain, inter_stored):+.4f}")
    print(f"  corr(plain adjacency,  cost)   = {corr(plain, cost):+.4f}")
    print(f"  corr(inter_patch adj., cost)   = {corr(inter_stored, cost):+.4f}")
    print("\n  If corr(plain,inter) ~ 1 and the two cost-correlations match, the metric")
    print("  switch moved along the SAME axis and could not decouple clustering from cost.")


# ===========================================================================
COMMANDS = {
    "clustering": cmd_clustering,
    "adjacency": cmd_adjacency,
}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    if cmd not in COMMANDS:
        print(__doc__)
        print(f"Subcommands: {', '.join(COMMANDS)}")
        return
    COMMANDS[cmd]()


if __name__ == "__main__":
    main(sys.argv)
