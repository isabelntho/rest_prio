# =============================================================================
# ECOSYSTEM CONDITION ANOMALIES CALCULATION
# =============================================================================
# Calculates abiotic and biotic anomaly rasters for the optimisation, writing to
# data/anomaly_scenarios/ (KB) or data/CH_wide/ (CH) - see OUTPUT_DIR below.
#
# TWO scenario families, controlled independently by the GENERATE_* switches:
#
# 1. CONDITION_SCENARIOS - vary WHICH indicators enter the composite.
#      Benchmark:     global | upper_q<NN> (q25/q50/q75/q90) | zones
#      Indicator LOO: all | drop one abiotic (3) | drop one biotic (8)
#      The global and zones benchmarks are crossed with every LOO construction;
#      the upper_q75 benchmark is crossed with the agricultural-relevant LOO
#      constructions (smd/sbd/soc/uzl/cdi/swf_h/swf_t/ndvi) for the Block 3
#      factorial. upper_q90 uses all indicators only (see the q25/q50 block for
#      why the high quantiles are not crossed).
#      Output: abiotic_{tag}.tif + biotic_{tag}.tif, one per ECT category.
#
# 2. WEIGHT_SCHEMES - vary HOW MUCH each indicator counts, indicator set fixed.
#      w_flat | w_cat | w_<indicator>, at the WEIGHT_BENCHMARKS scalings.
#      Output: ONE all-indicator weighted composite written to BOTH the abiotic_
#      and biotic_ filenames, so the optimisation's 1:1 abiotic+biotic sum carries
#      the weight vector without any change to the objective code. See the
#      INDICATOR WEIGHTING SCENARIOS block for the full rationale.
#
# Both families share the same pixel footprint, so plans are comparable across
# every tag (the weighted composite is explicitly masked to the equal-weight
# category intersection - see "Footprint pinning" below).

# =============================================================================
# REGION CONFIGURATION
# =============================================================================
# region controls which EC data and spatial mask are used:
#   "KB" -> Kanton Bern subregion: individual aligned EC rasters + Bern mask
#           (the original setup.R behaviour)
#   "CH" -> whole Switzerland: the EC_stack.tif raster stack + CH-wide mask.
#           The CH stack has no 'cdi' layer; it simply drops out downstream.
region <- "KB"

# Repo-root-relative: OUTPUT_DIR and DIAG_DIR below are root-relative too, so this
# script is run with the working directory set to the repository root.
source("data/setup.R")

# Load required packages
required_packages <- c("dplyr", "terra", "sf", "readxl", "classInt", "tidyr")
for (pkg in required_packages) {
    library(pkg, character.only = TRUE)
}

# ---------------------------------------------------------------------------
# Region overrides. For region == "CH" the individual aligned rasters and the
# Kanton Bern mask defined in setup.R are replaced by the CH-wide EC stack and
# a whole-Switzerland land-use / boundary mask. Everything downstream is
# unchanged: the anomaly, benchmark and diagnostic code all consume ec_data,
# LU, kb and mask_by_ecosystem exactly as before. 'cdi' is absent from the CH
# stack, so any scenario referencing it drops out naturally.
# ---------------------------------------------------------------------------
if (region == "CH") {
    EC_STACK_PATH <- "Y:/CH_Kanton_Bern/03_Workspaces/03_Habitat_condition/Restoration_potential/Data/EC_stack.tif"
    CH_LULC_PATH  <- "W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/LULC/LULC_2018_agg.tif"
    CH_BOUND_PATH <- "W:/EU_BioES_SELINA/WP3/4. Spatially_Explicit_EC/Data/CH_shps/swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET.shp"

    cat("Region = CH: using EC_stack and whole-Switzerland mask\n")

    ec_template <- rast(EC_STACK_PATH)

    # CH-wide land use, resampled onto the EC stack grid so mask_by_ecosystem's
    # mask(r, lu_mask) aligns (replaces the Bern-cropped LU from setup.R).
    LU <- resample(rast(CH_LULC_PATH), ec_template[[1]], method = "near")

    # Whole-Switzerland boundary (all cantons, no Bern filter) replaces 'kb'.
    kb <- st_read(CH_BOUND_PATH, quiet = TRUE)
    kb <- st_transform(kb, crs(ec_template))

    # Load the EC stack as a named list keyed by variable code (no 'cdi'),
    # matching the shape load_ec_data() returns for the KB region.
    load_ec_data <- function() {
        r <- rast(EC_STACK_PATH)
        cat(sprintf("Loaded EC stack: %d layers (%s)\n",
                    nlyr(r), paste(names(r), collapse = ", ")))
        # as.list() drops layer names; restore them so downstream code can
        # index ec_data[[var_code]] exactly as it does for the KB rasters.
        lst <- as.list(r)
        names(lst) <- names(r)
        lst
    }
}

# =============================================================================
# CONFIGURATION
# =============================================================================

ECOSYSTEM_TYPES <- c("forest", "agricultural", "grassland")

ECT_CATEGORIES <- list(
    "abiotic" = c("smd", "sbd", "soc"),
    "biotic"  = c("uzl", "tsd", "can", "cdi", "swf_h", "swf_t", "lai", "ndvi"),
    "landscape" = c("snh", "frag", "tcd")
)

# Region-aware output directory. The KB (Kanton Bern) and CH (whole Switzerland)
# rasters have different extents and MUST NOT overwrite each other: the Python
# optimisation picks the folder from its own REGION setting
# (data_loader.load_initial_conditions: 'CH_wide' if region == 'CH' else
# 'anomaly_scenarios'), so a CH run writing into anomaly_scenarios would silently
# feed CH-extent rasters to every Bern run.
OUTPUT_DIR <- if (region == "CH") "data/CH_wide" else "data/anomaly_scenarios"

# Indicator directionality:
#   positive  - higher raw value = better condition (default; z-score as-is)
#   negative  - lower raw value = better condition (smd, sbd); raster is negated
#               before computing the z-score so that degraded pixels still get
#               negative anomalies
#   unimodal  - optimal value lies at `optimum`; both extremes are degraded.
#               Raster is transformed to -|x - optimum| before z-scoring, so
#               pixels near the optimum receive positive anomalies.
INDICATOR_DIRECTIONALITY <- list(
    smd   = list(type = "positive"),
    sbd   = list(type = "negative"),
    soc   = list(type = "positive"),
    uzl   = list(type = "positive"),
    tsd   = list(type = "unimodal", optimum = 50),
    can   = list(type = "positive"),
    cdi   = list(type = "positive"),
    swf_h = list(type = "positive"),
    swf_t = list(type = "positive"),
    lai   = list(type = "positive"),
    ndvi  = list(type = "positive")
)

# 13 condition scenarios (separate axes)
CONDITION_SCENARIOS <- list(
    list(tag = "global_all",        benchmark = "global",    exclude_vars = character(0)),
    list(tag = "global_drop_smd",   benchmark = "global",    exclude_vars = "smd"),
    list(tag = "global_drop_sbd",   benchmark = "global",    exclude_vars = "sbd"),
    list(tag = "global_drop_soc",   benchmark = "global",    exclude_vars = "soc"),
    list(tag = "global_drop_uzl",   benchmark = "global",    exclude_vars = "uzl"),
    list(tag = "global_drop_tsd",   benchmark = "global",    exclude_vars = "tsd"),
    list(tag = "global_drop_can",   benchmark = "global",    exclude_vars = "can"),
    list(tag = "global_drop_cdi",   benchmark = "global",    exclude_vars = "cdi"),
    list(tag = "global_drop_swf_h", benchmark = "global",    exclude_vars = "swf_h"),
    list(tag = "global_drop_swf_t", benchmark = "global",    exclude_vars = "swf_t"),
    list(tag = "global_drop_lai",   benchmark = "global",    exclude_vars = "lai"),
    list(tag = "global_drop_ndvi",  benchmark = "global",    exclude_vars = "ndvi"),
    list(tag = "upper_q75_all",     benchmark = "upper_q75", exclude_vars = character(0)),
    list(tag = "zones_all",         benchmark = "zones",     exclude_vars = character(0)),

    # -- Benchmark-quantile sweep (all-indicator construction only) --------------
    # One ORDERED axis: how ambitious is the reference condition. "global" uses every
    # pixel, so it is effectively the q0 end of this same family, giving the series
    # global < q25 < q50 < q75 < q90. Read as a dose-response (how fast does the plan
    # change as the bar rises), not as unordered categories - the objective is
    # rescaled differently at each level, so HV is NOT comparable across them.
    # The all-indicator construction is what makes the quantile axis interpretable on
    # its own; the LOO crossings below are a separate, deliberate addition.
    list(tag = "upper_q25_all",     benchmark = "upper_q25", exclude_vars = character(0)),
    list(tag = "upper_q50_all",     benchmark = "upper_q50", exclude_vars = character(0)),
    list(tag = "upper_q90_all",     benchmark = "upper_q90", exclude_vars = character(0)),

    # -- q25 / q50 x indicator-LOO -----------------------------------------------
    # Crosses the benchmark-quantile axis with the indicator-LOO axis, so the
    # interaction is estimable: does dropping an indicator matter MORE or LESS
    # depending on how ambitious the reference is? That question is the reason to
    # cross them - the benchmark axis works by re-weighting the indicators against
    # each other (abiotic q75 = 2.58x global, biotic = 11.51x), so the two axes are
    # not obviously independent.
    #
    # NOTE: the 8 indicators here are the agricultural set; tsd/can/lai (forest-only)
    # are added for q25/q50/q75 in the "COMPLETE the LOO" block further down, which
    # is what makes the construction axis balanced across scaling levels.
    #
    # Restricted to global + q25 + q50 ON PURPOSE. Higher quantiles inflate the
    # anomalies until anomaly_improvement_weight saturates, flattening
    # restoration_benefit (best-plan / random-plan falls 6.04x at global -> 1.43x at
    # q75 -> 1.13x at q90). A LOO effect measured at q75/q90 would be read off an
    # objective that barely discriminates, so those crossings are omitted rather
    # than generated and then distrusted. global_drop_* already exist above.
    # Same 8 agricultural indicators as the q75 block below.
    list(tag = "upper_q25_drop_smd",   benchmark = "upper_q25", exclude_vars = "smd"),
    list(tag = "upper_q25_drop_sbd",   benchmark = "upper_q25", exclude_vars = "sbd"),
    list(tag = "upper_q25_drop_soc",   benchmark = "upper_q25", exclude_vars = "soc"),
    list(tag = "upper_q25_drop_uzl",   benchmark = "upper_q25", exclude_vars = "uzl"),
    list(tag = "upper_q25_drop_cdi",   benchmark = "upper_q25", exclude_vars = "cdi"),
    list(tag = "upper_q25_drop_swf_h", benchmark = "upper_q25", exclude_vars = "swf_h"),
    list(tag = "upper_q25_drop_swf_t", benchmark = "upper_q25", exclude_vars = "swf_t"),
    list(tag = "upper_q25_drop_ndvi",  benchmark = "upper_q25", exclude_vars = "ndvi"),
    list(tag = "upper_q50_drop_smd",   benchmark = "upper_q50", exclude_vars = "smd"),
    list(tag = "upper_q50_drop_sbd",   benchmark = "upper_q50", exclude_vars = "sbd"),
    list(tag = "upper_q50_drop_soc",   benchmark = "upper_q50", exclude_vars = "soc"),
    list(tag = "upper_q50_drop_uzl",   benchmark = "upper_q50", exclude_vars = "uzl"),
    list(tag = "upper_q50_drop_cdi",   benchmark = "upper_q50", exclude_vars = "cdi"),
    list(tag = "upper_q50_drop_swf_h", benchmark = "upper_q50", exclude_vars = "swf_h"),
    list(tag = "upper_q50_drop_swf_t", benchmark = "upper_q50", exclude_vars = "swf_t"),
    list(tag = "upper_q50_drop_ndvi",  benchmark = "upper_q50", exclude_vars = "ndvi"),

    # -- q75 x indicator-LOO (agricultural-relevant) - for the Block 3 factorial --
    # The scaling x construction factorial needs each LOO construction at BOTH the
    # global (anomaly) and upper_q75 references. The global LOO rasters already
    # exist above; these add the matching upper_q75 versions. LOO is restricted to
    # the agricultural EC indicators (setup.r): smd/sbd/soc + uzl/cdi/swf_h/swf_t/ndvi.
    list(tag = "upper_q75_drop_smd",   benchmark = "upper_q75", exclude_vars = "smd"),
    list(tag = "upper_q75_drop_sbd",   benchmark = "upper_q75", exclude_vars = "sbd"),
    list(tag = "upper_q75_drop_soc",   benchmark = "upper_q75", exclude_vars = "soc"),
    list(tag = "upper_q75_drop_uzl",   benchmark = "upper_q75", exclude_vars = "uzl"),
    list(tag = "upper_q75_drop_cdi",   benchmark = "upper_q75", exclude_vars = "cdi"),
    list(tag = "upper_q75_drop_swf_h", benchmark = "upper_q75", exclude_vars = "swf_h"),
    list(tag = "upper_q75_drop_swf_t", benchmark = "upper_q75", exclude_vars = "swf_t"),
    list(tag = "upper_q75_drop_ndvi",  benchmark = "upper_q75", exclude_vars = "ndvi"),

    # -- Forest-only indicators, to COMPLETE the LOO at every quantile ------------
    # tsd / can / lai exist only on forest pixels (see the per-ecosystem sets in
    # setup.R: forest biotic = tsd/can/lai, ag = uzl/cdi/swf_h/swf_t/ndvi, grassland
    # = uzl/swf_h/swf_t/ndvi). The q75 LOO family above was built for the
    # agricultural Block 3 factorial and skipped them, but the optimisation runs
    # with ECOSYSTEM_TO_RUN = "combined" (no ecosystem masking), so forest pixels
    # ARE in the decision space and these three do affect their biotic anomaly.
    # Adding them brings q25/q50/q75 up to the same 11 constructions as global, so
    # the scaling x construction design is BALANCED - a crossed factorial needs the
    # same construction levels at every scaling level.
    list(tag = "upper_q25_drop_tsd",   benchmark = "upper_q25", exclude_vars = "tsd"),
    list(tag = "upper_q25_drop_can",   benchmark = "upper_q25", exclude_vars = "can"),
    list(tag = "upper_q25_drop_lai",   benchmark = "upper_q25", exclude_vars = "lai"),
    list(tag = "upper_q50_drop_tsd",   benchmark = "upper_q50", exclude_vars = "tsd"),
    list(tag = "upper_q50_drop_can",   benchmark = "upper_q50", exclude_vars = "can"),
    list(tag = "upper_q50_drop_lai",   benchmark = "upper_q50", exclude_vars = "lai"),
    list(tag = "upper_q75_drop_tsd",   benchmark = "upper_q75", exclude_vars = "tsd"),
    list(tag = "upper_q75_drop_can",   benchmark = "upper_q75", exclude_vars = "can"),
    list(tag = "upper_q75_drop_lai",   benchmark = "upper_q75", exclude_vars = "lai"),

    # -- zones x indicator-LOO ---------------------------------------------------
    # Same reason the quantile axis was crossed with LOO (see the q25/q50 block):
    # the benchmark does not just shift the anomalies, it re-weights the indicators
    # against each other, so a LOO effect measured at one benchmark need not hold at
    # another. Zone-wise standardisation re-weights them by SPATIAL STRUCTURE rather
    # than by level - an indicator whose variation is mostly between production
    # regions or altitude bands loses most of its variance once each zone is centred
    # on its own mean, while a spatially unstructured indicator keeps it. Which
    # indicators those are is not knowable from the global composites, which is
    # exactly why the crossing has to be generated rather than reasoned about.
    #
    # Crossed with ALL 11 indicators, matching the global family (not the 8-indicator
    # agricultural subset used for the q75 Block 3 factorial). The optimisation runs
    # with ECOSYSTEM_TO_RUN = "combined", so forest pixels are in the decision space
    # and tsd/can/lai do affect their biotic anomaly.
    #
    # The saturation argument that stopped the q75/q90 crossings does NOT apply here:
    # that failure mode is high quantiles inflating anomalies until
    # anomaly_improvement_weight saturates and restoration_benefit flattens
    # (6.04x best/random at global -> 1.13x at q90). Zone-wise anomalies are centred
    # near 0 by construction, not inflated. Worth confirming against the objective on
    # zones_all before reading anything into these 11.
    list(tag = "zones_drop_smd",   benchmark = "zones", exclude_vars = "smd"),
    list(tag = "zones_drop_sbd",   benchmark = "zones", exclude_vars = "sbd"),
    list(tag = "zones_drop_soc",   benchmark = "zones", exclude_vars = "soc"),
    list(tag = "zones_drop_uzl",   benchmark = "zones", exclude_vars = "uzl"),
    list(tag = "zones_drop_tsd",   benchmark = "zones", exclude_vars = "tsd"),
    list(tag = "zones_drop_can",   benchmark = "zones", exclude_vars = "can"),
    list(tag = "zones_drop_cdi",   benchmark = "zones", exclude_vars = "cdi"),
    list(tag = "zones_drop_swf_h", benchmark = "zones", exclude_vars = "swf_h"),
    list(tag = "zones_drop_swf_t", benchmark = "zones", exclude_vars = "swf_t"),
    list(tag = "zones_drop_lai",   benchmark = "zones", exclude_vars = "lai"),
    list(tag = "zones_drop_ndvi",  benchmark = "zones", exclude_vars = "ndvi")
)

# =============================================================================
# INDICATOR WEIGHTING SCENARIOS (vertex set)
# =============================================================================
# The CONDITION_SCENARIOS above vary WHICH indicators enter the composite. These
# vary HOW MUCH each one counts, holding the indicator set fixed.
#
# Two weighting assumptions are baked into the scenarios above and are tested here:
#   1. every indicator inside an ECT category counts equally;
#   2. the abiotic block (3 indicators) and the biotic block (5-8, depending on
#      ecosystem) count equally as blocks, because the optimisation sums the two
#      category rasters 1:1. An abiotic indicator therefore carries ~2-3x the
#      influence of a biotic one.
#
# Construction: unlike CONDITION_SCENARIOS, a weighting scenario produces ONE
# composite over ALL abiotic+biotic indicators, written to BOTH the abiotic_ and
# biotic_ filenames. The optimisation's 1:1 abiotic+biotic sum is left untouched,
# so the objective becomes 2 * effect * w(C) instead of
# effect * w(A) + effect * w(B) - the same total effect magnitude, but now the
# abiotic-vs-biotic block balance is a property of the weight vector rather than
# a hardcoded 1:1. That is what makes "w_flat" and "w_cat" a meaningful contrast.
#
# Weights are resolved PER ECOSYSTEM over the indicators available there
# (setup.R's ec_categories), so an indicator only influences the ecosystems it
# exists in. Note forest has 3 abiotic + 3 biotic indicators, so w_cat and w_flat
# are identical there by construction - a useful self-check.
#
# Schemes:
#   "flat"       - w_i = 1/n for every available indicator
#   "cat"        - 0.5/n_abiotic on each abiotic, 0.5/n_biotic on each biotic
#                  (reproduces the equal-block weighting of the existing rasters)
#   "focal:<k>"  - k gets 2/n (double its equal share); the remaining 1 - 2/n is
#                  spread over the other n-1 indicators IN PROPORTION to their
#                  flat baseline shares, i.e. they all shrink by the same factor
#                  and keep their relative sizes.

# =============================================================================
# WHICH SCENARIO FAMILIES TO (RE)GENERATE
# =============================================================================
# The condition scenarios and their diagnostics are already on disk and existing
# optimisation results depend on them bit-for-bit. Regenerating them is a
# deliberate act, not a side effect of adding a new scenario family, so both are
# OFF by default - flip to TRUE only when the condition rasters really must be
# rebuilt.
GENERATE_CONDITION_SCENARIOS <- FALSE
GENERATE_WEIGHT_SCENARIOS    <- TRUE
GENERATE_BASELINE_DIAGNOSTICS <- FALSE

# Export the PER-INDICATOR anomaly layers (the z-scores that the composites average
# over). Nothing in the optimisation reads these: they exist so the weight-simplex
# screening diagnostic (Debugs_tests/weight_simplex_screen.py) can rebuild any weighted
# composite in python without re-deriving the z-scoring, and so the indicator
# correlation matrix can be computed at all - neither is possible from the composited
# abiotic_/biotic_ rasters alone.
#
# SAFE TO RUN: this block writes ONLY into OUTPUT_DIR/indicators/, a directory no other
# code reads or writes. It cannot touch the condition or weighting composites.
# But the OTHER flags above are not: set GENERATE_WEIGHT_SCENARIOS <- FALSE before
# running this on its own, or the weighting rasters that the 65 completed weighting runs
# depend on bit-for-bit get rewritten as a side effect.
GENERATE_INDICATOR_LAYERS <- FALSE

# Rasterise the EXTENDED weighting campaign: explicit weight vectors sampled further out
# on the simplex than the 13 local vertices reach (they span L1 0 to 0.150 of a simplex
# that reaches 0.65). Vectors come from a CSV rather than from WEIGHT_SCHEMES because an
# arbitrary Dirichlet draw has no rule that names it - "flat"/"cat"/"focal:<k>" are rules,
# and this campaign needs vectors. Written by
#   pixi run python Debugs_tests/weight_simplex_screen.py --emit-vectors
#
# SAFE TO RUN: any tag whose rasters already exist is SKIPPED, not rewritten (see
# SIMPLEX_SKIP_EXISTING). That matters because the CSV includes the campaign's own
# reference level global_w_flat, whose rasters the 65 completed vertex runs depend on
# bit-for-bit - the reference is re-RUN in the new batch, never re-generated.
GENERATE_SIMPLEX_WEIGHT_SCENARIOS <- FALSE
WEIGHT_VECTOR_CSV <- "data/diagnostics/weight_vectors_simplex.csv"
SIMPLEX_WEIGHT_BENCHMARK <- "global"
SIMPLEX_SKIP_EXISTING <- TRUE

# Restrict condition-scenario generation to these tags; NULL = every tag in
# CONDITION_SCENARIOS. Set this whenever ADDING scenarios: the generation loop writes
# with overwrite = TRUE and has no per-file existence check, so an unfiltered run
# rebuilds every existing raster - including the ones current optimisation results
# depend on bit-for-bit. Filtering is the safe way to add levels to an existing family.
CONDITION_TAG_FILTER <- paste0(
    "zones_drop_",
    c("smd", "sbd", "soc", "uzl", "tsd", "can", "cdi", "swf_h", "swf_t", "lai", "ndvi")
)

# Benchmarks the weighting vertex set is generated at. The grid runs the weighting
# axis at "global" only; add "upper_q75" here to cross it with the scaling axis.
WEIGHT_BENCHMARKS <- c("global")

# Indicators that get a focal (up-weighted) vertex: the abiotic + biotic union.
WEIGHT_FOCAL_VARS <- unlist(ECT_CATEGORIES[c("abiotic", "biotic")], use.names = FALSE)

# tag suffix -> scheme string. Tags follow the existing "{benchmark}_{construction}"
# grammar (e.g. "global_w_flat"), which is what the downstream R analysis parses.
WEIGHT_SCHEMES <- c(
    setNames("flat", "w_flat"),
    setNames("cat",  "w_cat"),
    setNames(paste0("focal:", WEIGHT_FOCAL_VARS), paste0("w_", WEIGHT_FOCAL_VARS))
)

# Load data
cat("Loading EC data...\n")
ec_data <- load_ec_data()

DEM_PATH <- "Z:/people/inicholson/WP2/NCP_models/Data/DEM_mean_LV95.tif"
PROD_REGIONS_PATH <- "Z:/people/inicholson/WP2/NCP_models/Data/PRODUCTION_REGIONS/PRODREG.shp"

# =============================================================================
# ANOMALY CALCULATION FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# Zone raster (production region x altitude class) - built once and cached.
# ---------------------------------------------------------------------------
.zone_raster_cache <- NULL

#' Build a zone raster combining production regions and altitude classes.
#'
#' Zones are the Cartesian product of production-region labels (from
#' PROD_REGIONS_PATH) and three altitude bands derived from DEM_PATH:
#'   class 1 :    0 - 600 m
#'   class 2 :  600 - 1200 m
#'   class 3 : 1200 m +
#'
#' The first attribute column of PROD_REGIONS_PATH is used as the region label.
#'
#' @param template_rast SpatRaster - defines target CRS, extent, and resolution.
#' @return SpatRaster with integer zone IDs.
build_zone_raster <- function(template_rast) {
    if (!is.null(.zone_raster_cache) &&
            compareGeom(.zone_raster_cache, template_rast, stopOnError = FALSE)) {
        return(.zone_raster_cache)
    }

    cat("    Building zone raster (production regions x altitude classes)...\n")

    # --- Altitude classes (3 bands) ---
    dem      <- rast(DEM_PATH)
    dem_proj <- project(dem, template_rast, method = "bilinear")
    alt_class <- classify(dem_proj, rbind(
        c(-Inf,  600, 1),
        c( 600, 1200, 2),
        c(1200,  Inf, 3)
    ), include.lowest = TRUE)
    names(alt_class) <- "alt_class"

    # --- Production region IDs ---
    prod_regions <- st_read(PROD_REGIONS_PATH, quiet = TRUE)
    prod_regions <- st_transform(prod_regions, crs(template_rast))
    # Use the third attribute column as the region label
    label_col <- setdiff(names(prod_regions), attr(prod_regions, "sf_column"))[3]
    cat(sprintf("    Production region label column: '%s'\n", label_col))
    prod_regions$zone_prod_id <- as.integer(as.factor(prod_regions[[label_col]]))
    prod_rast <- rasterize(vect(prod_regions), template_rast,
                           field = "zone_prod_id", background = NA)
    names(prod_rast) <- "prod_id"

    # --- Combine: sequential IDs from 1 to (n_prod * 3) ---
    zone_rast        <- (prod_rast - 1L) * 3L + alt_class
    names(zone_rast) <- "zone_id"

    n_zones <- length(na.omit(unique(values(zone_rast))))
    cat(sprintf("    Zone raster built: %d unique zones\n", n_zones))

    .zone_raster_cache <<- zone_rast
    return(zone_rast)
}

#' Compute benchmark statistics (mean, sd) for a single masked variable raster.
#' @param r_var SpatRaster - ecosystem-masked variable layer
#' @param method "global", "upper_q<NN>" (any quantile, e.g. upper_q25/q50/q75/q90),
#'   or "zones"
#' @return list(mean, sd) - scalars for global/upper_q<NN>; SpatRasters for zones
get_benchmark_stats <- function(r_var, method) {
    if (method == "global") {
        list(
            mean = global(r_var, "mean", na.rm = TRUE)[[1]],
            sd   = global(r_var, "sd",   na.rm = TRUE)[[1]]
        )
    } else if (grepl("^upper_q[0-9]+$", method)) {
        # Reference set = the top (100 - NN)% of pixels, e.g. upper_q75 -> top quartile,
        # upper_q90 -> top decile. r_var has already been directionality-transformed by
        # the caller, so "upper" always means "better condition" regardless of the
        # indicator's raw polarity. upper_q75 is just one value of p here, so the
        # existing rasters are reproduced bit-for-bit.
        p     <- as.numeric(sub("^upper_q", "", method)) / 100
        qv    <- global(r_var, function(x) quantile(x, p, na.rm = TRUE))[[1]]
        r_ref <- r_var
        r_ref[r_ref < qv] <- NA
        list(
            mean = global(r_ref, "mean", na.rm = TRUE)[[1]],
            sd   = global(r_ref, "sd",   na.rm = TRUE)[[1]]
        )
    } else if (method == "zones") {
        zone_rast <- build_zone_raster(r_var)
        if (!compareGeom(r_var, zone_rast, stopOnError = FALSE)) {
            zone_rast <- resample(zone_rast, r_var, method = "near")
        }
        # Per-zone mean and sd (columns: [zone_id, stat_value])
        zone_mean_df <- zonal(r_var, zone_rast, fun = "mean", na.rm = TRUE)
        zone_sd_df   <- zonal(r_var, zone_rast, fun = "sd",   na.rm = TRUE)
        # Map per-zone stats back to pixels via exact-value reclassification
        mean_rast <- classify(zone_rast, as.matrix(zone_mean_df[, c(1, 2)]))
        sd_rast   <- classify(zone_rast, as.matrix(zone_sd_df[,   c(1, 2)]))
        list(mean = mean_rast, sd = sd_rast)
    } else {
        stop(sprintf(paste0("Unknown benchmark method: '%s'. Use 'global', ",
                            "'upper_q<NN>' (e.g. upper_q75), or 'zones'."), method))
    }
}

#' Calculate standardized anomalies for a given ecosystem and variable set.
#' @param ecosystem_name Name of ecosystem type
#' @param variable_codes Vector of variable codes to process
#' @param benchmark "global", "upper_q<NN>" (e.g. upper_q75), or "zones"
#' @return SpatRaster with anomaly layers, or NULL
calculate_anomalies_by_ecosystem <- function(ecosystem_name, variable_codes,
                                             benchmark = "global") {
    cat(sprintf("  Processing %s ecosystem...\n", ecosystem_name))

    available_rasters <- ec_data[variable_codes]
    available_rasters <- available_rasters[!sapply(available_rasters, is.null)]

    if (length(available_rasters) == 0) {
        cat(sprintf("  Warning: No data available for %s ecosystem\n", ecosystem_name))
        return(NULL)
    }

    r_stack  <- terra::rast(available_rasters)
    r_masked <- mask_by_ecosystem(r_stack, ecosystem_name)

    anomaly_stack <- NULL

    for (var_code in names(r_masked)) {
        r_var <- r_masked[[var_code]]

        # Apply directionality transformation before computing benchmark stats
        dir_cfg <- INDICATOR_DIRECTIONALITY[[var_code]]
        dir_type <- if (!is.null(dir_cfg)) dir_cfg$type else "positive"
        if (dir_type == "negative") {
            r_var <- -r_var
        } else if (dir_type == "unimodal") {
            r_var <- -abs(r_var - dir_cfg$optimum)
        }

        stats <- get_benchmark_stats(r_var, benchmark)

        # For scalar benchmarks (global, upper_q75) guard against degenerate sd.
        # For the zones benchmark stats$sd is a SpatRaster; pixels with sd == 0
        # or NA will produce NA anomalies naturally - no scalar guard needed.
        if (!inherits(stats$sd, "SpatRaster")) {
            # Announce the skip rather than dropping the indicator silently: a dropped
            # indicator changes the COMPOSITION of the composite, which would confound a
            # benchmark comparison with an indicator-set change. Most likely at high
            # quantiles, where the reference set is small (q90 = top decile).
            if (is.na(stats$sd) || stats$sd <= 0) {
                cat(sprintf("    %s [%s]: SKIPPED - reference sd is %s at benchmark %s\n",
                            var_code, dir_type,
                            if (is.na(stats$sd)) "NA" else "<= 0", benchmark))
                next
            }
            cat(sprintf("    %s [%s]: ref_mean=%.3f, ref_sd=%.3f\n",
                        var_code, dir_type, stats$mean, stats$sd))
        } else {
            cat(sprintf("    %s [%s]: zone-wise benchmark (SpatRaster)\n", var_code, dir_type))
        }

        anomaly_layer <- (r_var - stats$mean) / stats$sd

        if (!inherits(anomaly_layer, "SpatRaster")) {
            cat(sprintf("  Warning: Invalid anomaly layer for %s\n", var_code))
            next
        }

        names(anomaly_layer) <- paste0(var_code, "_anom")
        anomaly_stack <- if (is.null(anomaly_stack)) anomaly_layer else c(anomaly_stack, anomaly_layer)
    }

    return(anomaly_stack)
}

#' Calculate mean anomaly for one ECT category, with optional indicator exclusion.
#' @param anomaly_stack SpatRaster with individual variable anomalies
#' @param ect_category "abiotic", "biotic", or "landscape"
#' @param exclude_vars Character vector of variable codes to drop from the mean
#' @return SpatRaster with mean anomaly, or NULL
calculate_ect_mean_anomaly <- function(anomaly_stack, ect_category,
                                       exclude_vars = character(0)) {
    if (is.null(anomaly_stack) || nlyr(anomaly_stack) == 0) return(NULL)

    target_variables <- setdiff(ECT_CATEGORIES[[ect_category]], exclude_vars)
    if (length(target_variables) == 0) {
        cat(sprintf("    All variables excluded for %s -- skipping\n", ect_category))
        return(NULL)
    }

    layer_names  <- names(anomaly_stack)
    valid_layers <- unlist(lapply(target_variables, function(var) {
        layer_names[grepl(paste0("^", var, "_anom"), layer_names)]
    }))
    valid_layers <- valid_layers[valid_layers %in% layer_names]

    if (length(valid_layers) == 0) {
        cat(sprintf("    No valid layers for %s (looking for: %s)\n",
                    ect_category, paste(target_variables, collapse = ", ")))
        return(NULL)
    }

    cat(sprintf("    %s: averaging %d layer(s): %s\n",
                ect_category, length(valid_layers), paste(valid_layers, collapse = ", ")))

    mean_anomaly <- if (length(valid_layers) == 1) {
        anomaly_stack[[valid_layers]]
    } else {
        mean(anomaly_stack[[valid_layers]], na.rm = TRUE)
    }

    names(mean_anomaly) <- paste0(ect_category, "_anomaly")
    return(mean_anomaly)
}

# =============================================================================
# INDICATOR WEIGHTING FUNCTIONS
# =============================================================================

#' Build an indicator weight vector for one ecosystem's available indicators.
#'
#' Weights always sum to 1 over `var_codes`. See WEIGHT_SCHEMES above for the
#' scheme definitions.
#'
#' @param var_codes Character vector of indicator codes present in this ecosystem
#' @param scheme    "flat", "cat", or "focal:<code>"
#' @return Named numeric vector of weights, in the order of `var_codes`
build_indicator_weights <- function(var_codes, scheme) {
    n <- length(var_codes)
    if (n == 0) stop("build_indicator_weights: no indicators supplied")

    base <- setNames(rep(1 / n, n), var_codes)

    if (scheme == "flat") {
        w <- base

    } else if (scheme == "cat") {
        abio <- intersect(var_codes, ECT_CATEGORIES[["abiotic"]])
        bio  <- intersect(var_codes, ECT_CATEGORIES[["biotic"]])
        w <- setNames(rep(0, n), var_codes)
        if (length(abio) > 0 && length(bio) > 0) {
            w[abio] <- 0.5 / length(abio)
            w[bio]  <- 0.5 / length(bio)
        } else if (length(abio) > 0) {
            # Only one block present: it takes all the weight (equals "flat").
            w[abio] <- 1 / length(abio)
        } else if (length(bio) > 0) {
            w[bio] <- 1 / length(bio)
        } else {
            stop(sprintf("build_indicator_weights: no abiotic/biotic indicators in (%s)",
                         paste(var_codes, collapse = ", ")))
        }

    } else if (startsWith(scheme, "focal:")) {
        focal <- sub("^focal:", "", scheme)
        # An indicator absent from this ecosystem cannot be up-weighted here; the
        # ecosystem falls back to the flat baseline so the focal vertex only ever
        # differs from w_flat where that indicator actually exists.
        if (!(focal %in% var_codes)) {
            w <- base
        } else if (n < 3) {
            # 2/n would leave zero (n == 2) or negative (n == 1) mass for the rest.
            cat(sprintf("    focal '%s': only %d indicator(s) available -- using flat weights\n",
                        focal, n))
            w <- base
        } else {
            rest <- setdiff(var_codes, focal)
            w <- base
            w[focal] <- 2 / n
            # Shrink the others by a COMMON factor so their relative sizes are
            # preserved (not reset to equal shares of the remainder).
            w[rest] <- base[rest] * ((1 - 2 / n) / sum(base[rest]))
        }

    } else {
        stop(sprintf("Unknown weighting scheme: '%s'. Use 'flat', 'cat', or 'focal:<code>'.",
                     scheme))
    }

    if (abs(sum(w) - 1) > 1e-9) {
        stop(sprintf("build_indicator_weights: weights for scheme '%s' sum to %.12f, not 1",
                     scheme, sum(w)))
    }
    if (any(w < 0)) {
        stop(sprintf("build_indicator_weights: scheme '%s' produced a negative weight", scheme))
    }

    return(w[var_codes])
}

#' Weighted composite of individual indicator anomalies, NA-aware.
#'
#' Generalises the equal-weight `mean(stack, na.rm = TRUE)` used by
#' calculate_ect_mean_anomaly: at each pixel the weights are renormalised over the
#' layers that are actually present there, so a pixel missing one indicator is not
#' penalised. Written out explicitly rather than via terra::weighted.mean so the
#' NA renormalisation is unambiguous.
#'
#'   num = sum_i  w_i * z_i          (NA contributes 0)
#'   den = sum_i  w_i * !is.na(z_i)
#'   C   = num / den                 (NA where den <= 0)
#'
#' @param anomaly_stack SpatRaster of per-indicator anomaly layers ("<code>_anom")
#' @param weights       Named numeric vector keyed by indicator code
#' @return SpatRaster with a single layer, or NULL
calculate_weighted_composite <- function(anomaly_stack, weights) {
    if (is.null(anomaly_stack) || nlyr(anomaly_stack) == 0) return(NULL)

    layer_names <- names(anomaly_stack)
    codes       <- sub("_anom$", "", layer_names)

    missing_w <- setdiff(codes, names(weights))
    if (length(missing_w) > 0) {
        stop(sprintf("calculate_weighted_composite: no weight for %s",
                     paste(missing_w, collapse = ", ")))
    }
    w <- as.numeric(weights[codes])

    # Drop zero-weight layers up front: they contribute nothing to num but would
    # still contribute 0 to den, so keeping them is harmless but wasteful.
    keep <- w > 0
    if (!any(keep)) return(NULL)
    anomaly_stack <- anomaly_stack[[which(keep)]]
    w <- w[keep]

    present <- !is.na(anomaly_stack)
    num <- sum(anomaly_stack * w, na.rm = TRUE)
    den <- sum(present * w, na.rm = TRUE)

    comp <- num / den
    comp[den <= 0] <- NA

    names(comp) <- "weighted_anomaly"
    return(comp)
}

# =============================================================================
# SCENARIO PROCESSING
# =============================================================================

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

if (!GENERATE_CONDITION_SCENARIOS) {
    cat("=== SKIPPING CONDITION ANOMALY SCENARIOS ",
        "(GENERATE_CONDITION_SCENARIOS = FALSE) ===\n", sep = "")
    cat("    Existing rasters in ", OUTPUT_DIR, " are left untouched.\n", sep = "")
}

# Apply CONDITION_TAG_FILTER (see the GENERATE_* block). Everything not listed is left
# alone on disk, so adding a scenario family does not rebuild the existing rasters.
.cond_scenarios <- if (!GENERATE_CONDITION_SCENARIOS) list() else
    Filter(function(s) is.null(CONDITION_TAG_FILTER) || s$tag %in% CONDITION_TAG_FILTER,
           CONDITION_SCENARIOS)

if (GENERATE_CONDITION_SCENARIOS && !is.null(CONDITION_TAG_FILTER)) {
    cat(sprintf("=== CONDITION_TAG_FILTER active: %d of %d scenarios will be written ===\n",
                length(.cond_scenarios), length(CONDITION_SCENARIOS)))
    cat("    ", paste(vapply(.cond_scenarios, function(s) s$tag, character(1)),
                      collapse = ", "), "\n", sep = "")
    .unknown <- setdiff(CONDITION_TAG_FILTER,
                        vapply(CONDITION_SCENARIOS, function(s) s$tag, character(1)))
    if (length(.unknown) > 0) {
        stop(sprintf("CONDITION_TAG_FILTER names unknown tag(s): %s",
                     paste(.unknown, collapse = ", ")))
    }
}

for (scenario in .cond_scenarios) {
    tag          <- scenario$tag
    benchmark    <- scenario$benchmark
    exclude_vars <- scenario$exclude_vars

    cat(sprintf("\n=== Scenario: %s (benchmark=%s, exclude=%s) ===\n",
                tag, benchmark,
                if (length(exclude_vars) == 0) "none" else paste(exclude_vars, collapse = ", ")))

    ect_results <- list(abiotic = NULL, biotic = NULL)

    for (ecosystem in ECOSYSTEM_TYPES) {
        cat(sprintf("\n--- %s ---\n", toupper(ecosystem)))

        # Use per-ecosystem indicator set from setup.R, restricted to abiotic/biotic
        eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
        eco_var_codes <- intersect(eco_all_codes, unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))

        eco_anomalies <- calculate_anomalies_by_ecosystem(
            ecosystem, eco_var_codes,
            benchmark = benchmark
        )
        if (is.null(eco_anomalies)) next

        for (ect_category in c("abiotic", "biotic")) {
            ect_anomaly <- calculate_ect_mean_anomaly(
                eco_anomalies, ect_category,
                exclude_vars = exclude_vars
            )
            if (!is.null(ect_anomaly)) {
                ect_results[[ect_category]] <- if (is.null(ect_results[[ect_category]])) {
                    ect_anomaly
                } else {
                    terra::cover(ect_results[[ect_category]], ect_anomaly)
                }
            }
        }
    }

    # Save abiotic and biotic rasters for this scenario
    for (ect_category in c("abiotic", "biotic")) {
        if (!is.null(ect_results[[ect_category]])) {
            out_path <- file.path(OUTPUT_DIR, sprintf("%s_%s.tif", ect_category, tag))
            writeRaster(ect_results[[ect_category]], out_path, overwrite = TRUE)
            cat(sprintf("  Saved %s (%.2f MB)\n", out_path, file.size(out_path) / 1024^2))
        } else {
            cat(sprintf("  Warning: No %s result for scenario %s\n", ect_category, tag))
        }
    }
}

# =============================================================================
# INDICATOR WEIGHTING SCENARIO PROCESSING
# =============================================================================
# Loop nesting is INVERTED relative to the condition scenarios above: ecosystem
# outer, weighting scheme inner. calculate_anomalies_by_ecosystem (the z-scoring,
# which is the expensive step) gives the same result for every weighting scheme at
# a given benchmark, so it runs once per ecosystem instead of once per scheme.

cat("\n\n=== CALCULATING INDICATOR WEIGHTING SCENARIOS (vertex set) ===\n")
cat(sprintf("  %d schemes x %d benchmark(s): %s\n",
            length(WEIGHT_SCHEMES), length(WEIGHT_BENCHMARKS),
            paste(names(WEIGHT_SCHEMES), collapse = ", ")))

weight_manifest_rows <- list()

for (benchmark in if (GENERATE_WEIGHT_SCENARIOS) WEIGHT_BENCHMARKS else character(0)) {

    # One mosaicked composite per scheme, accumulated across ecosystems.
    weight_results <- setNames(vector("list", length(WEIGHT_SCHEMES)),
                               names(WEIGHT_SCHEMES))

    for (ecosystem in ECOSYSTEM_TYPES) {
        cat(sprintf("\n--- %s [benchmark=%s] ---\n", toupper(ecosystem), benchmark))

        eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
        eco_var_codes <- intersect(eco_all_codes, unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))

        eco_anomalies <- calculate_anomalies_by_ecosystem(
            ecosystem, eco_var_codes, benchmark = benchmark
        )
        if (is.null(eco_anomalies)) next

        avail_codes <- sub("_anom$", "", names(eco_anomalies))

        # --- Footprint pinning -------------------------------------------------
        # The weighted composite spans ALL indicators, so it is non-NA wherever ANY
        # of them is present - a SUPERSET of the per-category footprints. The
        # optimisation drops any pixel that is NaN in any input layer
        # (data_loader.load_initial_conditions), so an enlarged footprint would
        # quietly grow the decision space and make these runs incomparable to the
        # existing global_all baseline. Pin the composite to the intersection of the
        # two equal-weight category footprints, which is exactly what the existing
        # rasters give the optimiser.
        fp_ab <- calculate_ect_mean_anomaly(eco_anomalies, "abiotic")
        fp_bi <- calculate_ect_mean_anomaly(eco_anomalies, "biotic")
        footprint <- if (!is.null(fp_ab) && !is.null(fp_bi)) {
            # Arithmetic propagates NA, so this is 0 where BOTH are present and NA
            # anywhere either is missing.
            fp_ab * 0 + fp_bi * 0
        } else if (!is.null(fp_ab)) {
            fp_ab * 0
        } else if (!is.null(fp_bi)) {
            fp_bi * 0
        } else {
            NULL
        }
        if (is.null(footprint)) {
            cat("    No category composite available -- skipping ecosystem\n")
            next
        }

        for (suffix in names(WEIGHT_SCHEMES)) {
            scheme <- WEIGHT_SCHEMES[[suffix]]
            w <- build_indicator_weights(avail_codes, scheme)

            comp <- calculate_weighted_composite(eco_anomalies, w)
            if (is.null(comp)) {
                cat(sprintf("    %s: no weighted layers -- skipping\n", suffix))
                next
            }
            comp <- mask(comp, footprint)

            weight_results[[suffix]] <- if (is.null(weight_results[[suffix]])) {
                comp
            } else {
                terra::cover(weight_results[[suffix]], comp)
            }

            weight_manifest_rows[[length(weight_manifest_rows) + 1]] <- data.frame(
                tag        = paste0(benchmark, "_", suffix),
                benchmark  = benchmark,
                scheme     = scheme,
                ecosystem  = ecosystem,
                indicator  = names(w),
                weight     = as.numeric(w),
                stringsAsFactors = FALSE
            )
        }

        cat(sprintf("    %d scheme(s) composited over %d indicator(s): %s\n",
                    length(WEIGHT_SCHEMES), length(avail_codes),
                    paste(avail_codes, collapse = ", ")))
    }

    # --- Write ----------------------------------------------------------------
    # The SAME composite goes to both filenames. The optimisation sums the abiotic
    # and biotic rasters 1:1, so duplicating the all-indicator composite makes the
    # objective 2 * effect * w(C) with the weighting fully expressed in C, leaving
    # the Python objective code untouched.
    for (suffix in names(WEIGHT_SCHEMES)) {
        tag <- paste0(benchmark, "_", suffix)
        if (is.null(weight_results[[suffix]])) {
            cat(sprintf("  Warning: no result for weighting scenario %s\n", tag))
            next
        }
        for (ect_category in c("abiotic", "biotic")) {
            out_r <- weight_results[[suffix]]
            names(out_r) <- paste0(ect_category, "_anomaly")
            out_path <- file.path(OUTPUT_DIR, sprintf("%s_%s.tif", ect_category, tag))
            writeRaster(out_r, out_path, overwrite = TRUE)
        }
        cat(sprintf("  Saved abiotic_%s.tif + biotic_%s.tif (identical by design)\n",
                    tag, tag))
    }
}

# --- Weight manifest ---------------------------------------------------------
# Records exactly which weight each indicator received in each ecosystem, for
# verification and for reporting the design.
DIAG_DIR <- "diagnostics"
dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)

if (length(weight_manifest_rows) > 0) {
    weight_manifest <- do.call(rbind, weight_manifest_rows)
    manifest_path <- file.path(DIAG_DIR, "indicator_weight_manifest.csv")
    write.csv(weight_manifest, manifest_path, row.names = FALSE)
    cat(sprintf("\n  Saved weight manifest: %s (%d rows)\n",
                manifest_path, nrow(weight_manifest)))

    # Every (tag, ecosystem) group must sum to 1.
    grp_sums <- aggregate(weight ~ tag + ecosystem, data = weight_manifest, FUN = sum)
    bad <- grp_sums[abs(grp_sums$weight - 1) > 1e-9, ]
    if (nrow(bad) > 0) {
        print(bad)
        stop("Weight manifest: some (tag, ecosystem) groups do not sum to 1")
    }
    cat(sprintf("  Weight check OK: all %d (tag, ecosystem) groups sum to 1\n",
                nrow(grp_sums)))
}

# =============================================================================
# SIMPLEX WEIGHT SCENARIOS (extended weighting campaign)
# =============================================================================
# Same construction as the vertex set above - one all-indicator weighted composite per
# tag, written to BOTH the abiotic_ and biotic_ filenames, pinned to the same footprint -
# but the weights are read from a CSV of explicit union vectors instead of resolved from
# a scheme rule. Loop nesting matches the vertex section (ecosystem outer, tag inner) so
# the expensive z-scoring runs once per ecosystem rather than once per tag.

if (GENERATE_SIMPLEX_WEIGHT_SCENARIOS) {

cat("\n\n=== GENERATING SIMPLEX WEIGHT SCENARIOS ===\n")

if (!file.exists(WEIGHT_VECTOR_CSV)) {
    stop(sprintf(paste0("WEIGHT_VECTOR_CSV not found: %s\nGenerate it with:\n",
                        "  pixi run python Debugs_tests/weight_simplex_screen.py ",
                        "--emit-vectors"), WEIGHT_VECTOR_CSV))
}
wv <- read.csv(WEIGHT_VECTOR_CSV, stringsAsFactors = FALSE)
for (col in c("tag", "indicator", "weight")) {
    if (!col %in% names(wv)) {
        stop(sprintf("%s has no '%s' column (found: %s)",
                     WEIGHT_VECTOR_CSV, col, paste(names(wv), collapse = ", ")))
    }
}

# Every union vector must be a probability vector before it is restricted per ecosystem.
.tag_sums <- tapply(wv$weight, wv$tag, sum)
.bad_sums <- names(.tag_sums)[abs(.tag_sums - 1) > 1e-6]
if (length(.bad_sums) > 0) {
    stop(sprintf("union weight vectors do not sum to 1 for: %s",
                 paste(.bad_sums, collapse = ", ")))
}
# An indicator in the CSV that this pipeline does not know about means the two sides have
# drifted apart, which would silently drop that indicator's weight.
.unknown <- setdiff(unique(wv$indicator), unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))
if (length(.unknown) > 0) {
    stop(sprintf("%s names indicator(s) absent from ECT_CATEGORIES: %s",
                 WEIGHT_VECTOR_CSV, paste(.unknown, collapse = ", ")))
}

simplex_tags <- unique(wv$tag)
if (SIMPLEX_SKIP_EXISTING) {
    .exists <- vapply(simplex_tags, function(tg) {
        all(file.exists(file.path(OUTPUT_DIR, sprintf("%s_%s.tif",
                                                      c("abiotic", "biotic"), tg))))
    }, logical(1))
    if (any(.exists)) {
        cat(sprintf("  %d tag(s) already on disk - SKIPPED, not rewritten: %s\n",
                    sum(.exists), paste(simplex_tags[.exists], collapse = ", ")))
        simplex_tags <- simplex_tags[!.exists]
    }
}

if (length(simplex_tags) == 0) {
    cat("  Nothing to generate.\n")
} else {

cat(sprintf("  %d tag(s) to generate at benchmark '%s'\n",
            length(simplex_tags), SIMPLEX_WEIGHT_BENCHMARK))

simplex_results <- setNames(vector("list", length(simplex_tags)), simplex_tags)
simplex_manifest_rows <- list()

for (ecosystem in ECOSYSTEM_TYPES) {
    cat(sprintf("\n  %s ecosystem...\n", ecosystem))

    eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
    eco_var_codes <- intersect(eco_all_codes,
                               unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))
    eco_anomalies <- calculate_anomalies_by_ecosystem(
        ecosystem, eco_var_codes, benchmark = SIMPLEX_WEIGHT_BENCHMARK
    )
    if (is.null(eco_anomalies)) {
        cat("    No anomalies for this ecosystem -- skipping\n")
        next
    }
    avail_codes <- sub("_anom$", "", names(eco_anomalies))

    # Identical footprint pinning to the vertex section: the composite spans all
    # indicators and so is non-NA on a SUPERSET of the per-category footprints; pinning
    # keeps the decision space the same as every other scenario the optimiser has seen.
    fp_ab <- calculate_ect_mean_anomaly(eco_anomalies, "abiotic")
    fp_bi <- calculate_ect_mean_anomaly(eco_anomalies, "biotic")
    footprint <- if (!is.null(fp_ab) && !is.null(fp_bi)) {
        fp_ab * 0 + fp_bi * 0
    } else if (!is.null(fp_ab)) {
        fp_ab * 0
    } else if (!is.null(fp_bi)) {
        fp_bi * 0
    } else {
        NULL
    }
    if (is.null(footprint)) {
        cat("    No category composite available -- skipping ecosystem\n")
        next
    }

    for (tg in simplex_tags) {
        rows <- wv[wv$tag == tg, ]
        w_union <- setNames(as.numeric(rows$weight), rows$indicator)
        w <- w_union[intersect(avail_codes, names(w_union))]
        if (length(w) == 0) next
        # Restrict to this ecosystem's indicators and renormalise. The composite is
        # num/den and so is scale-invariant in w, meaning this renormalisation does not
        # change a single pixel - it exists so the manifest records a probability vector
        # and the sum-to-1 check below is meaningful.
        w <- w / sum(w)

        comp <- calculate_weighted_composite(eco_anomalies, w)
        if (is.null(comp)) next
        comp <- mask(comp, footprint)
        simplex_results[[tg]] <- if (is.null(simplex_results[[tg]])) {
            comp
        } else {
            terra::cover(simplex_results[[tg]], comp)
        }

        simplex_manifest_rows[[length(simplex_manifest_rows) + 1]] <- data.frame(
            tag = tg, benchmark = SIMPLEX_WEIGHT_BENCHMARK, scheme = "explicit",
            ecosystem = ecosystem, indicator = names(w), weight = as.numeric(w),
            stringsAsFactors = FALSE
        )
    }
    cat(sprintf("    %d tag(s) composited over %d indicator(s): %s\n",
                length(simplex_tags), length(avail_codes),
                paste(avail_codes, collapse = ", ")))
}

for (tg in simplex_tags) {
    if (is.null(simplex_results[[tg]])) {
        cat(sprintf("  Warning: no result for simplex scenario %s\n", tg))
        next
    }
    for (ect_category in c("abiotic", "biotic")) {
        out_r <- simplex_results[[tg]]
        names(out_r) <- paste0(ect_category, "_anomaly")
        writeRaster(out_r, file.path(OUTPUT_DIR, sprintf("%s_%s.tif", ect_category, tg)),
                    overwrite = TRUE)
    }
    cat(sprintf("  Saved abiotic_%s.tif + biotic_%s.tif\n", tg, tg))
}

if (length(simplex_manifest_rows) > 0) {
    simplex_manifest <- do.call(rbind, simplex_manifest_rows)
    dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)
    simplex_path <- file.path(DIAG_DIR, "simplex_weight_manifest.csv")
    write.csv(simplex_manifest, simplex_path, row.names = FALSE)
    grp <- aggregate(weight ~ tag + ecosystem, data = simplex_manifest, FUN = sum)
    bad <- grp[abs(grp$weight - 1) > 1e-9, ]
    if (nrow(bad) > 0) {
        print(bad)
        stop("Simplex manifest: some (tag, ecosystem) groups do not sum to 1")
    }
    cat(sprintf("\n  Saved manifest: %s (%d rows)\n", simplex_path,
                nrow(simplex_manifest)))
    cat(sprintf("  Weight check OK: all %d (tag, ecosystem) groups sum to 1\n",
                nrow(grp)))
}

}
}

# =============================================================================
# PER-INDICATOR ANOMALY LAYER EXPORT
# =============================================================================
# The individual indicator z-scores are otherwise transient: they live only inside
# calculate_anomalies_by_ecosystem() and are averaged away into the ECT composites.
# This writes them out unchanged, one raster per (ecosystem, indicator), plus the
# footprint each weighted composite is pinned to.
#
# calculate_anomalies_by_ecosystem() is reused as-is ON PURPOSE: directionality,
# benchmark statistics and ecosystem masking are then identical to the production
# rasters by construction rather than by reimplementation, which is what lets the
# python side validate its rebuild against the 13 existing global_w_* composites.

if (GENERATE_INDICATOR_LAYERS) {

cat("\n\n=== EXPORTING PER-INDICATOR ANOMALY LAYERS ===\n")

IND_LAYER_DIR <- file.path(OUTPUT_DIR, "indicators")
dir.create(IND_LAYER_DIR, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("  Output: %s/  (benchmark: global)\n", IND_LAYER_DIR))

ind_manifest_rows <- list()

for (ecosystem in ECOSYSTEM_TYPES) {
    cat(sprintf("\n  %s ecosystem...\n", ecosystem))

    # Same indicator set the condition and weighting scenarios use for this ecosystem.
    eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
    eco_var_codes <- intersect(eco_all_codes,
                               unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))

    eco_anomalies <- calculate_anomalies_by_ecosystem(
        ecosystem, eco_var_codes, benchmark = "global"
    )
    if (is.null(eco_anomalies)) {
        cat("    No anomalies for this ecosystem -- skipping\n")
        next
    }

    for (lyr_name in names(eco_anomalies)) {
        code <- sub("_anom$", "", lyr_name)
        r_out <- eco_anomalies[[lyr_name]]
        names(r_out) <- code
        out_path <- file.path(IND_LAYER_DIR,
                              sprintf("%s_%s_global.tif", ecosystem, code))
        writeRaster(r_out, out_path, overwrite = TRUE)

        category <- if (code %in% ECT_CATEGORIES[["abiotic"]]) "abiotic" else "biotic"
        ind_manifest_rows[[length(ind_manifest_rows) + 1]] <- data.frame(
            ecosystem = ecosystem, indicator = code, category = category,
            n_valid   = as.numeric(global(!is.na(r_out), "sum", na.rm = TRUE)[[1]]),
            file      = basename(out_path),
            stringsAsFactors = FALSE
        )
    }

    # Footprint the weighted composites are pinned to: non-NA only where BOTH category
    # means exist. Identical expression to the weighting section above, so a python
    # rebuild masks to exactly the same pixels the optimiser saw.
    fp_ab <- calculate_ect_mean_anomaly(eco_anomalies, "abiotic")
    fp_bi <- calculate_ect_mean_anomaly(eco_anomalies, "biotic")
    footprint <- if (!is.null(fp_ab) && !is.null(fp_bi)) {
        fp_ab * 0 + fp_bi * 0
    } else if (!is.null(fp_ab)) {
        fp_ab * 0
    } else if (!is.null(fp_bi)) {
        fp_bi * 0
    } else {
        NULL
    }
    if (!is.null(footprint)) {
        names(footprint) <- "footprint"
        fp_path <- file.path(IND_LAYER_DIR,
                             sprintf("%s_footprint_global.tif", ecosystem))
        writeRaster(footprint, fp_path, overwrite = TRUE)
        cat(sprintf("    %d indicator layer(s) + footprint written\n",
                    nlyr(eco_anomalies)))
    } else {
        cat("    WARNING: no footprint could be built for this ecosystem\n")
    }
}

if (length(ind_manifest_rows) > 0) {
    ind_manifest <- do.call(rbind, ind_manifest_rows)
    ind_manifest_path <- file.path(IND_LAYER_DIR, "indicator_layer_manifest.csv")
    write.csv(ind_manifest, ind_manifest_path, row.names = FALSE)
    cat(sprintf("\n  Saved manifest: %s (%d layers)\n",
                ind_manifest_path, nrow(ind_manifest)))
    print(ind_manifest[, c("ecosystem", "indicator", "category", "n_valid")])
} else {
    cat("\n  WARNING: no indicator layers were written\n")
}

}

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=== SCENARIO PROCESSING COMPLETE ===\n")
all_tifs <- list.files(OUTPUT_DIR, pattern = "\\.tif$", full.names = FALSE)
cat(sprintf("Total rasters in %s: %d\n", OUTPUT_DIR, length(all_tifs)))

if (GENERATE_CONDITION_SCENARIOS) {
    # Count what was ACTUALLY written (post-CONDITION_TAG_FILTER), not the size of the
    # master list - otherwise a filtered run over-reports.
    cat(sprintf("  Condition scenarios generated: %d (%d scenarios x 2 rasters)\n",
                length(.cond_scenarios) * 2, length(.cond_scenarios)))
    if (!is.null(CONDITION_TAG_FILTER)) {
        cat(sprintf("    (CONDITION_TAG_FILTER active - %d of %d scenarios; the rest were left untouched)\n",
                    length(.cond_scenarios), length(CONDITION_SCENARIOS)))
    }
}
if (GENERATE_WEIGHT_SCENARIOS) {
    n_expected <- length(WEIGHT_SCHEMES) * length(WEIGHT_BENCHMARKS) * 2
    weight_tifs <- unlist(lapply(WEIGHT_BENCHMARKS, function(b) {
        as.vector(outer(c("abiotic", "biotic"),
                        paste0(b, "_", names(WEIGHT_SCHEMES)),
                        function(a, t) sprintf("%s_%s.tif", a, t)))
    }))
    n_found <- sum(weight_tifs %in% all_tifs)
    cat(sprintf("  Weighting scenarios: %d/%d rasters present\n", n_found, n_expected))
    missing <- setdiff(weight_tifs, all_tifs)
    if (length(missing) > 0) {
        cat("  MISSING:\n")
        for (f in missing) cat(sprintf("    %s\n", f))
    } else {
        for (f in weight_tifs) cat(sprintf("    %s\n", f))
    }
}
cat("\nEcosystem condition anomaly scenarios completed!\n")

# =============================================================================
# BASELINE INDICATOR CONTRIBUTION STATISTICS
# =============================================================================
# For the global_all baseline scenario, quantifies each indicator's contribution
# to the abiotic and biotic composites across all eligible pixels (pooled across
# all ecosystem types).
#
# Requires: ec_data and mask_by_ecosystem() from setup.R (loaded at top of script).
#
# Statistics:
#   mean_abs_contribution   - mean |z-score| across pixels; how strongly the
#                             indicator pulls the composite on average
#   coefficient of variation - indicator's share of total within-category variance;
#                             how spatially heterogeneous this indicator is relative
#                             to its peers
#   correlation_with_composite - Pearson r vs. the ECT composite; high = redundant
#                             (dropping it won't shift the composite much), low =
#                             unique signal (dropping it will cause observable change)
#   r_squared               - cor^2; variance of composite explained by indicator
#   dominance_frequency_pct - % of pixels where this indicator has the largest |z|
#                             in its ECT category; shows who drives extreme values

#' Compute per-indicator contribution statistics for one ECT category.
#' @param indicator_vals named list of numeric vectors (pooled pixel z-scores)
#' @param composite_vals named list of numeric vectors (pooled pixel z-scores)
#' @return data.frame with one row per indicator
compute_indicator_contribution_stats <- function(indicator_vals, composite_vals) {
    stats_rows <- lapply(names(indicator_vals), function(ind) {
        x    <- indicator_vals[[ind]]
        c_all <- composite_vals[[ind]]
        keep <- !is.na(x) & !is.na(c_all)
        x_k  <- x[keep]
        c_k  <- c_all[keep]

        mac  <- mean(abs(x_k))
        cv   <- if (mac > 0) sd(abs(x_k), na.rm = TRUE) / mac else NA_real_
        r    <- if (length(x_k) > 1 && sd(x_k) > 0 && sd(c_k) > 0) cor(x_k, c_k) else NA_real_
        r2   <- if (!is.na(r)) r^2 else NA_real_

        data.frame(
            indicator              = ind,
            n_pixels               = length(x_k),
            mean_abs_contribution  = mac,
            cv    = cv,
            correlation_with_composite = r,
            r_squared              = r2,
            stringsAsFactors       = FALSE
        )
    })
    result <- do.call(rbind, stats_rows)
    result
}

if (GENERATE_BASELINE_DIAGNOSTICS) {

cat("\n=== BASELINE INDICATOR CONTRIBUTION STATISTICS (global_all) ===\n\n")

DIAG_DIR <- "diagnostics"
dir.create(DIAG_DIR, recursive = TRUE, showWarnings = FALSE)

# Accumulators: per-ECT-category, per-indicator - pixel value vectors
baseline_indicator_vals   <- list(abiotic = list(), biotic = list())
baseline_composite_vals   <- list(abiotic = list(), biotic = list())
baseline_dominance_counts <- list(abiotic = integer(0), biotic = integer(0))

for (ecosystem in ECOSYSTEM_TYPES) {
    cat(sprintf("  Processing %s ecosystem...\n", ecosystem))

    # Use per-ecosystem indicator set from setup.R, restricted to abiotic/biotic
    eco_all_codes <- names(ec_categories[["EC variables"]][[ecosystem]])
    eco_var_codes <- intersect(eco_all_codes, unlist(ECT_CATEGORIES[c("abiotic", "biotic")]))

    eco_anomalies <- calculate_anomalies_by_ecosystem(
        ecosystem, eco_var_codes, benchmark = "global"
    )
    if (is.null(eco_anomalies)) next

    for (ect_category in c("abiotic", "biotic")) {
        ind_codes <- ECT_CATEGORIES[[ect_category]]

        all_layer_names <- names(eco_anomalies)
        cat_layers <- unlist(lapply(ind_codes, function(v) {
            all_layer_names[grepl(paste0("^", v, "_anom"), all_layer_names)]
        }))
        cat_layers <- cat_layers[cat_layers %in% all_layer_names]
        if (length(cat_layers) == 0) next

        ind_names <- sub("_anom$", "", cat_layers)

        composite_path <- file.path(OUTPUT_DIR, sprintf("%s_global_all.tif", ect_category))
        if (!file.exists(composite_path)) {
            cat(sprintf("    Composite raster not found: %s\n", composite_path))
            next
        }
        r_composite <- rast(composite_path)
        ind_stack   <- eco_anomalies[[cat_layers]]

        r_comp_matched <- if (!compareGeom(ind_stack, r_composite, stopOnError = FALSE)) {
            resample(r_composite, ind_stack, method = "near")
        } else {
            r_composite
        }

        eco_mask_vals <- as.vector(values(ind_stack[[1]]))
        eco_pixels    <- which(!is.na(eco_mask_vals))
        comp_vals_eco <- as.vector(values(r_comp_matched))[eco_pixels]

        # Accumulate indicator pixel vectors (composite paired per-indicator)
        for (i in seq_along(ind_names)) {
            ind  <- ind_names[i]
            vals <- as.vector(values(ind_stack[[i]]))[eco_pixels]
            baseline_indicator_vals[[ect_category]][[ind]] <-
                c(baseline_indicator_vals[[ect_category]][[ind]], vals)
            baseline_composite_vals[[ect_category]][[ind]] <-
                c(baseline_composite_vals[[ect_category]][[ind]], comp_vals_eco)
        }

        # Dominance: pixel-wise which indicator has largest |z|
        if (length(cat_layers) > 1) {
            abs_stack   <- abs(ind_stack)
            dom_indices <- as.vector(values(app(abs_stack, which.max)))[eco_pixels]
            dom_indices <- dom_indices[!is.na(dom_indices)]
            dom_counts  <- tabulate(dom_indices, nbins = length(ind_names))
            names(dom_counts) <- ind_names
            for (ind in ind_names) {
                prev <- baseline_dominance_counts[[ect_category]][ind]
                baseline_dominance_counts[[ect_category]][ind] <-
                    if (is.null(prev) || is.na(prev)) dom_counts[ind]
                    else prev + dom_counts[ind]
            }
        } else {
            ind <- ind_names[1]
            prev <- baseline_dominance_counts[[ect_category]][ind]
            baseline_dominance_counts[[ect_category]][ind] <-
                if (is.null(prev) || is.na(prev)) length(eco_pixels)
                else prev + length(eco_pixels)
        }
    }
}

# Compute stats and assemble output table
all_stats <- lapply(c("abiotic", "biotic"), function(ect_category) {
    ind_vals   <- baseline_indicator_vals[[ect_category]]
    comp_vals  <- baseline_composite_vals[[ect_category]]
    dom_counts <- baseline_dominance_counts[[ect_category]]

    if (length(ind_vals) == 0) {
        cat(sprintf("  No data accumulated for %s\n", ect_category))
        return(NULL)
    }
    cat(sprintf("\nComputing stats for %s (%d indicators, %d pooled pixels)...\n",
                ect_category, length(ind_vals), length(comp_vals)))

    df        <- compute_indicator_contribution_stats(ind_vals, comp_vals)
    total_dom <- sum(dom_counts, na.rm = TRUE)
    df$dominance_frequency_pct <- if (total_dom > 0) {
        round(100 * dom_counts[df$indicator] / total_dom, 2)
    } else {
        rep(NA_real_, nrow(df))
    }
    df$ect_category <- ect_category
    df[, c("ect_category", "indicator", "n_pixels",
           "mean_abs_contribution", "cv",
           "correlation_with_composite", "r_squared",
           "dominance_frequency_pct")]
})

contribution_stats <- do.call(rbind, all_stats)

# Save
out_csv <- file.path(DIAG_DIR, "indicator_contribution_baseline.csv")
write.csv(contribution_stats, out_csv, row.names = FALSE)
cat(sprintf("\n  Saved: %s\n", out_csv))

# Print summary
cat("\n--- INDICATOR CONTRIBUTION SUMMARY (baseline: global_all) ---\n")
print(contribution_stats, digits = 3, row.names = FALSE)
cat("\nIndicator contribution statistics complete!\n")

} else {
    cat("\n=== SKIPPING BASELINE INDICATOR CONTRIBUTION STATISTICS ",
        "(GENERATE_BASELINE_DIAGNOSTICS = FALSE) ===\n", sep = "")
}
