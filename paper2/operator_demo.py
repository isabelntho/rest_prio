"""Toy-landscape walkthrough of the region-based search operators.

Builds a small synthetic landscape (a 40x40 grid of eligible cells instead of
the ~437k of the Bern run) and pushes one population through the REAL operator
classes from Core_optimisation.spatial_operations, unmodified:

    SpatialCoverageSampling -> RegionSwapCrossover -> RegionEvolveMutation
                            -> MinPatchSizeRepair

That is the production `region_evolve` order: pymoo's Mating applies crossover
first, then mutation, and the repair runs afterwards on the offspring.

The point of the toy is legibility. At 100 m over Kanton Bern one decision
variable is one of ~437k pixels and nothing at the level of an individual gene
is visible; here a cell in the map IS a bit in the genotype, so the figures can
show what each operator does to the decision vector.

Nothing here loads data. The module is imported by the figure chunks in
_methods.qmd and can also be run standalone:

    pixi run python paper2/operator_demo.py --preview OUTDIR

REPRODUCIBILITY. Every operator in spatial_operations.py builds its generator as
`np.random.default_rng(np.random.randint(0, 2**31 - 1))`, i.e. it seeds itself
from the LEGACY GLOBAL numpy RNG. Fixing a figure therefore means calling
np.random.seed(...) immediately before each operator call - passing a Generator
in has no effect. That is what _seeded() below is for.
"""
import os
import sys

import numpy as np

# paper2/operator_demo.py -> repo root is one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Core_optimisation.spatial_operations import (  # noqa: E402
    MinPatchSizeRepair,
    RegionEvolveMutation,
    RegionSwapCrossover,
    SpatialCoverageSampling,
    _label_components,
    build_restoration_neighbor_table,
)

# --------------------------------------------------------------------------
# toy configuration
# --------------------------------------------------------------------------
# Grid small enough that a cell is a readable mark at print width. The operator
# settings below are the production ones from run_custom_nsga2.py wherever they
# are scale-free (move probabilities, growth bias, score temperature, budget
# tolerance) and scaled down where they are counted in pixels (seed lattice,
# seed count, mutation edits), so the moves stay visible against a budget of
# ~110 rather than ~21,900 pixels.
GRID_H = 40
GRID_W = 40
BUDGET_FRACTION = 0.08     # baseline run uses 0.05
MIN_PATCH_SIZE = 8         # baseline run uses S = 2
SEED_GRID = 8              # baseline 16
REGION_SEEDS = 6           # baseline 25
REGION_SEEDS_MIN = 2       # baseline 5
MUT_EDIT_FRACTION = 0.10   # n_edits as a share of the budget
PIXEL_TOLERANCE = 0.10     # as in the baseline run
SCORE_TEMPERATURE = 1.0    # as in the baseline run
GROWTH_BIAS = "scored"     # as in the baseline run

SEED = 11
N_POP = 24

# Move order and the production probabilities, quoted so the figure titles match
# RegionEvolveMutation's own defaults (spatial_operations.py).
MOVE_PROBS = {"relocate": 0.40, "spawn": 0.20, "delete": 0.10,
              "grow": 0.15, "shrink": 0.15}
MOVE_ORDER = ["relocate", "spawn", "delete", "grow", "shrink"]

# --------------------------------------------------------------------------
# palette (Okabe-Ito derived; safe in greyscale and for common colour vision
# deficiencies, and consistent with the diverging/sequential ramps used by the
# diagnostics in Debugs_tests)
# --------------------------------------------------------------------------
C_ELIGIBLE = "#e9e9e9"
C_SELECTED = "#0072B2"   # blue
C_KEPT = "#9ecae1"       # muted blue: selected before AND after
C_ADDED = "#009E73"      # green
C_REMOVED = "#D55E00"    # vermillion
C_PARENT_A = "#0072B2"
C_PARENT_B = "#E69F00"   # orange
C_BOTH = "#56B4E9"       # light blue: region present in both parents
C_REGROWN = "#009E73"    # added by the crossover's own trim/grow step
C_GRID = "#ffffff"
SEQUENTIAL = "YlOrRd"


# --------------------------------------------------------------------------
# toy landscape
# --------------------------------------------------------------------------
def _z(v):
    """Standardise; returns zeros if the field is constant."""
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def _gauss(rr, cc, r0, c0, sr, sc):
    return np.exp(-((((rr - r0) / sr) ** 2) + (((cc - c0) / sc) ** 2)))


def make_toy_landscape(H=GRID_H, W=GRID_W):
    """Synthetic landscape standing in for the Bern eligible pool.

    Returns (ic, scores, potential, cost) where `ic` is the minimal
    initial-conditions dict the region operators need: they read only 'shape'
    and 'restoration_eligible_indices' (everything else goes through
    build_restoration_neighbor_table, which caches its table on the dict).

    Every cell of the grid is eligible: the toy is about what the operators do
    to a plan, and an ineligible mask only adds holes the reader has to parse
    around. The operators themselves are indifferent - they address cells
    through the neighbour table either way.

    Potential and cost are smooth but only PARTLY overlapping, so the blended
    score has genuine trade-off structure rather than a single optimum. The
    blend z(potential) - z(cost) mirrors the real region_scores construction in
    resto_anom.py.
    """
    rr, cc = np.mgrid[0:H, 0:W].astype(float)

    potential = (1.15 * _gauss(rr, cc, 0.24 * H, 0.18 * W, 0.17 * H, 0.15 * W)
                 + 1.00 * _gauss(rr, cc, 0.72 * H, 0.68 * W, 0.20 * H, 0.18 * W)
                 + 0.55 * _gauss(rr, cc, 0.56 * H, 0.12 * W, 0.12 * H, 0.10 * W))
    cost = (0.90 * _gauss(rr, cc, 0.16 * H, 0.30 * W, 0.20 * H, 0.28 * W)
            + 0.85 * _gauss(rr, cc, 0.84 * H, 0.30 * W, 0.18 * H, 0.22 * W)
            + 0.55 * (cc / max(W - 1, 1)))

    idx = np.arange(H * W)
    ic = {"shape": (H, W), "restoration_eligible_indices": idx}
    pot = potential.ravel()[idx]
    cst = cost.ravel()[idx]
    scores = _z(pot) - _z(cst)
    return ic, scores, pot, cst


class ToyProblem(object):
    """The three attributes the region operators read off `problem`.

    RestorationProblem itself is not needed: none of the four operators touches
    anything else on it (see their _do methods in spatial_operations.py).
    """

    def __init__(self, n_rest, k):
        self.n_var = int(n_rest)
        self.n_restoration_pixels = int(n_rest)
        self.max_action_pixels = int(k)


def make_problem(ic, frac=BUDGET_FRACTION):
    n_rest = int(np.asarray(ic["restoration_eligible_indices"]).size)
    return ToyProblem(n_rest, max(int(frac * n_rest), 1))


def to_map(vals, ic, fill=np.nan):
    """Scatter a length-n_rest vector back onto the (H, W) grid."""
    arr = np.full(tuple(ic["shape"]), fill, dtype=float)
    arr.flat[np.asarray(ic["restoration_eligible_indices"])] = vals
    return arr


def _seeded(seed):
    """Seed the legacy global RNG that every operator draws its generator from."""
    np.random.seed(int(seed) % (2 ** 31 - 1))


def _components(sel, ic):
    nbr, rows, cols = build_restoration_neighbor_table(ic)
    return _label_components(np.asarray(sel, bool), tuple(ic["shape"]), rows, cols)


def _min_comp(sel, ic):
    comps = _components(sel, ic)
    return min((int(c.size) for c in comps), default=0)


# --------------------------------------------------------------------------
# the stage-by-stage run
# --------------------------------------------------------------------------
def run_stages(seed=SEED, n_pop=N_POP):
    """Drive the real operators in production order and return every stage.

    The returned dict carries the whole populations (so the score-temperature
    figure can pool over them) plus the specific individuals the walkthrough
    figures show.
    """
    ic, scores, pot, cst = make_toy_landscape()
    prob = make_problem(ic)
    k = prob.max_action_pixels
    n_edits = max(2, int(round(MUT_EDIT_FRACTION * k)))

    # --- 1. initial population -------------------------------------------
    sampler = SpatialCoverageSampling(
        ic, k, scores, region_seeds=REGION_SEEDS,
        region_seeds_min=REGION_SEEDS_MIN, growth_bias="neutral",
        seed_grid=SEED_GRID)
    _seeded(seed)
    X0 = sampler._do(prob, n_pop)

    # --- 2. crossover -----------------------------------------------------
    # pymoo hands the crossover a (n_parents, n_matings, n_var) block.
    cx = RegionSwapCrossover(ic, k, scores, growth_bias=GROWTH_BIAS,
                             pixel_tolerance=PIXEL_TOLERANCE,
                             score_temperature=SCORE_TEMPERATURE)
    n_mat = n_pop // 2
    P = np.stack([X0[:n_mat], X0[n_mat:2 * n_mat]])
    _seeded(seed + 1)
    Q = cx._do(prob, P)
    Xc = np.concatenate([Q[0], Q[1]], axis=0)

    # Show the mating whose child inherits visibly from BOTH parents - the
    # point of the operator. Score = how balanced the child's inheritance is.
    balance = []
    for m in range(n_mat):
        a = P[0, m].astype(bool)
        b = P[1, m].astype(bool)
        ch = Q[0, m].astype(bool)
        na, nb = int((ch & a & ~b).sum()), int((ch & b & ~a).sum())
        balance.append(min(na, nb) - 0.02 * abs(na - nb))
    m_star = int(np.argmax(balance))

    # --- 3. mutation, one move at a time ----------------------------------
    # move_probs is a constructor argument, so a one-hot dict isolates a move
    # without patching anything. All five start from the SAME individual so the
    # panels are directly comparable.
    # Start the move panels from a member of the INITIAL population: it sits
    # exactly on the budget, so a spawned region fits inside the tolerance band
    # instead of being trimmed straight back off by the operator's own budget
    # enforcement (a post-crossover individual can already sit at the top of
    # the band, where spawn is a no-op).
    base_ind = _pick_multi_region(X0, ic)
    moves = {}
    for j, mv in enumerate(MOVE_ORDER):
        mut = RegionEvolveMutation(
            ic, k, scores, move_probs={mv: 1.0}, n_edits=n_edits,
            growth_bias=GROWTH_BIAS, pixel_tolerance=PIXEL_TOLERANCE,
            score_temperature=SCORE_TEMPERATURE)
        after = None
        for attempt in range(12):
            _seeded(seed + 100 * (j + 1) + attempt)
            cand = mut._do(prob, base_ind.reshape(1, -1).copy())[0]
            if not np.array_equal(cand, base_ind):
                after = cand
                break
        if after is None:                       # move was a no-op every time
            after = base_ind.copy()
        moves[mv] = after

    # --- 4. the production chain, for the repair panel ---------------------
    # Repair only has work to do on what the variation operators actually
    # produce, so run the real mutation (full move distribution) over the whole
    # offspring population and then repair it, rather than hand-fragmenting an
    # individual.
    mut_prod = RegionEvolveMutation(
        ic, k, scores, n_edits=n_edits, growth_bias=GROWTH_BIAS,
        pixel_tolerance=PIXEL_TOLERANCE, score_temperature=SCORE_TEMPERATURE)
    _seeded(seed + 2)
    Xm = mut_prod._do(prob, Xc.copy())

    rep = MinPatchSizeRepair(
        ic, k, MIN_PATCH_SIZE, scores=scores, pixel_tolerance=PIXEL_TOLERANCE,
        growth_bias=GROWTH_BIAS, score_temperature=SCORE_TEMPERATURE)
    _seeded(seed + 3)
    Xr = rep._do(prob, Xm.copy())

    # Pick the individual that shows BOTH halves of the repair: sub-S fragments
    # dropped and the freed budget regrown onto the survivors. Requiring more
    # than one surviving component keeps the survivors visible - an individual
    # that collapses to a single region shows the deletion but not the
    # "regrowth is seeded only from survivors" part.
    i_rep, best_rank = 0, None
    for i in range(len(Xm)):
        before = np.asarray(Xm[i][:prob.n_restoration_pixels], bool)
        after = np.asarray(Xr[i][:prob.n_restoration_pixels], bool)
        comps_before = _components(before, ic)
        n_small = sum(1 for c in comps_before if c.size < MIN_PATCH_SIZE)
        if n_small == 0 or len(_components(after, ic)) < 2:
            continue
        rank = (int((after & ~before).sum()) > 0, n_small)
        if best_rank is None or rank > best_rank:
            i_rep, best_rank = i, rank

    return dict(
        ic=ic, prob=prob, scores=scores, potential=pot, cost=cst,
        k=k, n_edits=n_edits, n_rest=prob.n_restoration_pixels,
        X0=X0, Xc=Xc, Xm=Xm, Xr=Xr,
        parent_a=P[0, m_star], parent_b=P[1, m_star], child=Q[0, m_star],
        base_ind=base_ind, moves=moves,
        repair_before=Xm[i_rep], repair_after=Xr[i_rep],
        repair_min_before=_min_comp(Xm[i_rep], ic),
        repair_min_after=_min_comp(Xr[i_rep], ic),
    )


def _pick_multi_region(X, ic, target_n=4):
    """An individual with enough components that delete/relocate have a choice.

    Prefers `target_n` regions (several to pick from, still legible) and, among
    those, the most evenly sized set. The operator chooses which component to
    act on at random, so an individual with one dominant region and a few
    stragglers would make most draws invisible.
    """
    best, best_rank = 0, None
    for i in range(len(X)):
        comps = _components(X[i], ic)
        if not comps:
            continue
        sizes = [int(c.size) for c in comps]
        rank = (-abs(len(comps) - target_n), min(sizes))
        if best_rank is None or rank > best_rank:
            best, best_rank = i, rank
    return X[best].copy()


def growth_frequency(temperature, seed=SEED, n_ind=40):
    """Selection frequency of a population grown at a given score temperature.

    Used for the score-temperature figure: identical settings apart from the
    Gumbel temperature, so the difference in the frequency surface is
    attributable to the ordering alone.
    """
    ic, scores, _, _ = make_toy_landscape()
    prob = make_problem(ic)
    k = prob.max_action_pixels
    mut = RegionEvolveMutation(
        ic, k, scores, move_probs={"relocate": 1.0},
        n_edits=max(2, int(round(MUT_EDIT_FRACTION * k))),
        growth_bias="scored", pixel_tolerance=PIXEL_TOLERANCE,
        score_temperature=float(temperature))
    sampler = SpatialCoverageSampling(
        ic, k, scores, region_seeds=REGION_SEEDS,
        region_seeds_min=REGION_SEEDS_MIN, growth_bias="neutral",
        seed_grid=SEED_GRID)
    _seeded(seed)
    X = sampler._do(prob, n_ind)
    # Several score-guided relocations, so the ordering has time to express
    # itself; that is the regime the prose describes.
    for r in range(6):
        _seeded(seed + 1000 * (r + 1) + int(temperature * 97))
        X = mut._do(prob, X)
    return ic, X[:, :prob.n_restoration_pixels].astype(bool).mean(axis=0)


# ==========================================================================
# panels
# ==========================================================================
def _categorical(ax, cat, colors, ic, gridlines=True):
    """imshow an integer category grid with an explicit colour per category."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors), 1.0), cmap.N)
    ax.imshow(cat, cmap=cmap, norm=norm, interpolation="nearest")
    _frame(ax, ic, gridlines)


def _frame(ax, ic, gridlines=True):
    H, W = tuple(ic["shape"])
    ax.set_xticks([])
    ax.set_yticks([])
    if gridlines and max(H, W) <= 60:
        # One line per cell: the figure's whole point is that a cell is a gene.
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color=C_GRID, linewidth=0.25, alpha=0.55)
        ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("0.6")


def _base_categories(ic):
    """0 = unselected; every cell of the toy grid is eligible."""
    return np.zeros(tuple(ic["shape"]), dtype=int)


def _to_grid_bool(vec, ic):
    n_rest = int(np.asarray(ic["restoration_eligible_indices"]).size)
    g = np.zeros(tuple(ic["shape"]), dtype=bool)
    g.flat[np.asarray(ic["restoration_eligible_indices"])] = \
        np.asarray(vec[:n_rest], dtype=bool)
    return g


def panel_state(ax, sel, ic, title=None, seed_lattice=False, color=C_SELECTED):
    """A single plan: selected cells against the eligible backdrop."""
    cat = _base_categories(ic)
    cat[_to_grid_bool(sel, ic)] = 1
    _categorical(ax, cat, [C_ELIGIBLE, color], ic)
    if seed_lattice:
        _draw_lattice(ax, ic)
    if title:
        ax.set_title(title, fontsize=8.5)


def panel_delta(ax, before, after, ic, title=None):
    """What one operator did: cells kept, added and removed."""
    b = _to_grid_bool(before, ic)
    a = _to_grid_bool(after, ic)
    cat = _base_categories(ic)
    cat[b & a] = 1
    cat[a & ~b] = 2
    cat[b & ~a] = 3
    _categorical(ax, cat, [C_ELIGIBLE, C_KEPT, C_ADDED, C_REMOVED], ic)
    if title:
        ax.set_title(title, fontsize=8.5)


def panel_child(ax, child, parent_a, parent_b, ic, title=None):
    """A crossover child, with each selected cell attributed to its source."""
    ch = _to_grid_bool(child, ic)
    pa = _to_grid_bool(parent_a, ic)
    pb = _to_grid_bool(parent_b, ic)
    cat = _base_categories(ic)
    cat[ch & pa & ~pb] = 1
    cat[ch & pb & ~pa] = 2
    cat[ch & pa & pb] = 3
    cat[ch & ~pa & ~pb] = 4
    _categorical(ax, cat, [C_ELIGIBLE, C_PARENT_A, C_PARENT_B,
                           C_BOTH, C_REGROWN], ic)
    if title:
        ax.set_title(title, fontsize=8.5)


def _draw_lattice(ax, ic, n=SEED_GRID):
    H, W = tuple(ic["shape"])
    for i in range(1, n):
        ax.axvline(i * W / n - 0.5, color="0.45", lw=0.4, ls=":", zorder=4)
        ax.axhline(i * H / n - 0.5, color="0.45", lw=0.4, ls=":", zorder=4)


def _legend(fig, entries, ncol, y=0.0, fontsize=7.5):
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(facecolor=c, edgecolor="0.6", linewidth=0.4, label=l)
                        for l, c in entries],
               loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=fontsize, frameon=False, handlelength=1.3,
               handleheight=1.0, columnspacing=1.4)


# ==========================================================================
# figures
# ==========================================================================
def _crossover_legend(child, parent_a, parent_b, n_rest):
    """Legend entries for the categories that actually occur in this mating.

    A mating in which the inherited regions happen to be disjoint has no
    "in both parents" cells; listing the colour anyway would promise the reader
    marks that are not on the page.
    """
    ch = np.asarray(child[:n_rest], bool)
    pa = np.asarray(parent_a[:n_rest], bool)
    pb = np.asarray(parent_b[:n_rest], bool)
    entries = [("not selected", C_ELIGIBLE),
               ("selected (parent A)", C_PARENT_A),
               ("selected (parent B)", C_PARENT_B)]
    if (ch & pa & pb).any():
        entries.append(("in both parents", C_BOTH))
    if (ch & ~pa & ~pb).any():
        entries.append(("grown by the trim/grow step", C_REGROWN))
    return entries


LEGEND_DELTA = [
    ("not selected", C_ELIGIBLE),
    ("selected, unchanged by the move", C_KEPT),
    ("added by the move", C_ADDED),
    ("removed by the move", C_REMOVED),
]


def figure_operators(stages=None, figsize=(5.6, 2.55)):
    """Figure: one crossover mating - two parents and their child.

    The two parents are unmutated members of the initial population, so they
    double as the sampling illustration - they differ in both the number of
    regions and their placement, which is what SpatialCoverageSampling varies -
    and no separate row of initial plans is needed.
    """
    import matplotlib.pyplot as plt

    d = stages if stages is not None else run_stages()
    ic = d["ic"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # The seed lattice is drawn on the parents because that is the construct
    # that placed their seeds; the child owes nothing to it.
    panel_state(axes[0], d["parent_a"], ic, color=C_PARENT_A, seed_lattice=True,
                title="(a) parent A\ninitial plan, %d regions"
                      % len(_components(d["parent_a"], ic)))
    panel_state(axes[1], d["parent_b"], ic, color=C_PARENT_B, seed_lattice=True,
                title="(b) parent B\ninitial plan, %d regions"
                      % len(_components(d["parent_b"], ic)))
    panel_child(axes[2], d["child"], d["parent_a"], d["parent_b"], ic,
                title="(c) child\nwhole regions inherited")

    # Margins are set explicitly rather than by tight_layout/bbox_inches: knitr
    # writes the figure at exactly the declared size, so the layout has to fit
    # inside the canvas on its own.
    fig.subplots_adjust(left=0.012, right=0.988, top=0.815, bottom=0.135,
                        wspace=0.06)
    _legend(fig, _crossover_legend(d["child"], d["parent_a"], d["parent_b"],
                                   d["n_rest"]), ncol=3, y=-0.012)
    return fig


def figure_mutation_moves(stages=None, figsize=(5.6, 4.5)):
    """Figure: the five region-level moves, all from the same starting plan.

    Panel (a) is that starting plan drawn on the same synthetic landscape as
    figure_operators, so the moves can be read against a state rather than
    reconstructed from the unchanged cells of the delta panels.
    """
    import matplotlib.pyplot as plt

    d = stages if stages is not None else run_stages()
    ic = d["ic"]
    base = d["base_ind"]
    n_base = len(_components(base, ic))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    flat = axes.ravel()

    # Drawn in the same pale blue the delta panels use for unchanged cells, so
    # one colour means "selected before the move" throughout the figure.
    panel_state(flat[0], base, ic, color=C_KEPT,
                title="(a) starting plan\n%d regions, %d cells"
                      % (n_base, int(base[:d["n_rest"]].sum())))

    for ax, letter, mv in zip(flat[1:], "bcdef", MOVE_ORDER):
        after = d["moves"][mv]
        panel_delta(ax, base, after, ic,
                    title="(%s) %s ($p$ = %.2f)" % (letter, mv, MOVE_PROBS[mv]))

    fig.subplots_adjust(left=0.012, right=0.988, top=0.905, bottom=0.115,
                        wspace=0.06, hspace=0.26)
    _legend(fig, LEGEND_DELTA, ncol=2, y=-0.008)
    return fig


def figure_score_temperature(figsize=(4.9, 2.9), temperatures=(0.0, SCORE_TEMPERATURE)):
    """Figure: selection frequency under deterministic vs perturbed ordering."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(temperatures), figsize=figsize)
    im = None
    for ax, T in zip(np.atleast_1d(axes), temperatures):
        ic, freq = growth_frequency(T)
        cat = _base_categories(ic)
        _categorical(ax, cat, [C_ELIGIBLE], ic, gridlines=False)
        arr = to_map(np.where(freq > 0, freq * 100.0, np.nan), ic)
        im = ax.imshow(arr, cmap=SEQUENTIAL, vmin=0, vmax=100, interpolation="nearest")
        ax.set_title("$T$ = %.1f" % T, fontsize=8.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.30, wspace=0.06)
    cax = fig.add_axes([0.27, 0.185, 0.46, 0.042])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("population selection frequency (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    return fig


# ==========================================================================
# self-checks
# ==========================================================================
def self_check(stages=None):
    """Assert that each stage did what the methods text says it does.

    Returns a list of (name, ok, detail). Nothing here is cosmetic: if one of
    these fails the figure is telling the reader something untrue.
    """
    d = stages if stages is not None else run_stages()
    ic, k = d["ic"], d["k"]
    lo, hi = int(k * (1 - PIXEL_TOLERANCE)), int(k * (1 + PIXEL_TOLERANCE))
    out = []

    def add(name, ok, detail):
        out.append((name, bool(ok), detail))

    # budget: sampling is exact, the later stages sit inside the tolerance band
    n0 = d["X0"][:, :d["n_rest"]].sum(axis=1)
    add("sampling hits the budget exactly", bool(np.all(n0 == k)),
        "counts %d..%d, budget %d" % (n0.min(), n0.max(), k))
    for nm in ("Xc", "Xm", "Xr"):
        c = d[nm][:, :d["n_rest"]].sum(axis=1)
        add("%s inside the budget band" % nm, bool(np.all((c >= lo) & (c <= hi))),
            "counts %d..%d, band [%d, %d]" % (c.min(), c.max(), lo, hi))

    # the contiguity floor
    mc = [_min_comp(x, ic) for x in d["Xr"]]
    add("every repaired plan meets the patch floor", min(mc) >= MIN_PATCH_SIZE,
        "smallest component over the population %d, floor %d" % (min(mc), MIN_PATCH_SIZE))
    mc_pre = [_min_comp(x, ic) for x in d["Xm"]]
    add("the floor was actually binding before repair", min(mc_pre) < MIN_PATCH_SIZE,
        "smallest component before repair %d" % min(mc_pre))

    # the crossover transmits whole regions rather than mixing pixels
    ch = np.asarray(d["child"][:d["n_rest"]], bool)
    pa = np.asarray(d["parent_a"][:d["n_rest"]], bool)
    pb = np.asarray(d["parent_b"][:d["n_rest"]], bool)
    from_a, from_b = int((ch & pa).sum()), int((ch & pb).sum())
    add("child inherits from both parents", from_a > 0 and from_b > 0,
        "%d cells from A, %d from B, %d grown by the trim/grow step"
        % (from_a, from_b, int((ch & ~pa & ~pb).sum())))

    # each isolated move does its own job
    base = d["base_ind"]
    nb, cb = len(_components(base, ic)), int(base[:d["n_rest"]].sum())
    expect = {
        "relocate": ("region count and size held, cells moved",
                     lambda n, c, x: n == nb and abs(c - cb) <= 2 and not np.array_equal(x, base)),
        "spawn": ("a region added", lambda n, c, x: n > nb),
        "delete": ("a region removed", lambda n, c, x: n < nb),
        "grow": ("selection larger", lambda n, c, x: c > cb),
        "shrink": ("selection smaller", lambda n, c, x: c < cb),
    }
    for mv in MOVE_ORDER:
        x = d["moves"][mv]
        n, c = len(_components(x, ic)), int(x[:d["n_rest"]].sum())
        label, test = expect[mv]
        add("move '%s': %s" % (mv, label), test(n, c, x),
            "%d -> %d regions, %d -> %d cells, %d genes flipped"
            % (nb, n, cb, c, int((x != base).sum())))

    # the legacy-global-RNG seeding really does pin the figure
    d2 = run_stages()
    same = all(np.array_equal(d[nm], d2[nm]) for nm in ("X0", "Xc", "Xm", "Xr"))
    add("run is reproducible", same, "re-ran run_stages() and compared every stage")
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", metavar="OUTDIR", default=None,
                    help="write the figures as PNGs to OUTDIR")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)

    d = run_stages()
    print("toy landscape: %dx%d grid, %d eligible cells (genes), budget %d (%.0f%%), "
          "S = %d, n_edits = %d"
          % (GRID_H, GRID_W, d["n_rest"], d["k"], 100 * BUDGET_FRACTION,
             MIN_PATCH_SIZE, d["n_edits"]))

    if not args.no_check:
        rows = self_check(d)
        width = max(len(n) for n, _, _ in rows)
        for name, ok, detail in rows:
            print("  [%s] %-*s  %s" % ("ok" if ok else "FAIL", width, name, detail))
        n_bad = sum(1 for _, ok, _ in rows if not ok)
        print("  %d of %d checks failed" % (n_bad, len(rows)))

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.preview, exist_ok=True)
        for name, builder in (("fig_operators", lambda: figure_operators(d)),
                              ("fig_mutation_moves", lambda: figure_mutation_moves(d)),
                              ("fig_score_temperature", figure_score_temperature)):
            fig = builder()
            path = os.path.join(args.preview, name + ".png")
            # No bbox_inches="tight": knitr saves the figure at its declared
            # size, so the preview must show the same clipping the manuscript
            # would get.
            fig.savefig(path, dpi=args.dpi, facecolor="white")
            plt.close(fig)
            print("  -> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
