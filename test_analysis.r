df <- read.csv("intermediate_results/scenarios_20260106.csv")
df_lng <- pivot_longer(df, cols = starts_with("objective_"), names_to = "Objective", values_to = "Obval")
df_lng$burden_sharing <- ifelse(df_lng$burden_sharing == "no", 0, 1)
library(ggplot2)
df0 <- df_lng %>% filter(Objective == "objective_0")%>% select(-n_pixels_restored)%>%
                    pivot_longer(cols=max_restoration_fraction:improvement_effect,names_to = "Param", values_to = "Param_value")
df1 <- df_lng %>% filter(Objective == "objective_1")
df2 <- df_lng %>% filter(Objective == "objective_2")

df_all <- df_lng %>% select(-n_pixels_restored)%>%
                    pivot_longer(cols=max_restoration_fraction:improvement_effect,names_to = "Param", values_to = "Param_value")
df_all$Param_value <- as.factor(df_all$Param_value)

df0$Param_value <- as.factor(df0$Param_value)

ggplot(df0)+
    geom_boxplot(aes(x=factor(Param_value), y=Obval))+
    facet_wrap(~Param, scales="free")

-	Variance and distribution (density plots) of the objective values per scenario (1 plot per objective)
-	Map of the robust locations (overall) with boxes below showing plots of one fixed parameter at a time.
-	2D scatter plot and density plot to show tradeoffs (coloured by scenario)


df <- read.csv("scenarios_1301.csv")
library(emmeans)

# df must contain:

df <- df %>%
  mutate(
    frac = factor(max_restoration_fraction),
    clust = factor(spatial_clustering),
    ab = factor(abiotic_effect),
    bio = factor(biotic_effect)
  )

## Option 1: one model per objective, planned contrasts per parameter

objective_cols <- c("abiotic_anomaly", "biotic_anomaly")  

fit_one_objective <- function(dat, ycol) {
  f <- as.formula(paste0(ycol, " ~ frac + clust + ab + bio"))
  m <- lm(f, data = dat)

  res <- list(
    model = m,
    frac = summary(contrast(emmeans(m, ~ frac), "pairwise", adjust = "BH")),
    clust = summary(contrast(emmeans(m, ~ clust), "pairwise", adjust = "BH")),
    ab = summary(contrast(emmeans(m, ~ ab), "pairwise", adjust = "BH")),
    bio = summary(contrast(emmeans(m, ~ bio), "pairwise", adjust = "BH"))
  )
  res
}

results_by_objective <- setNames(
  lapply(objective_cols, \(y) fit_one_objective(df, y)),
  objective_cols
)

# Example: view contrasts for p1 on obj1
#results_by_objective[["abiotic_anomaly"]]$frac

params <- c("frac", "clust", "ab", "bio")

effect_strength <- function(model, param) {
  emm <- emmeans(model, as.formula(paste0("~ ", param)))
  emm_df <- as.data.frame(emm)

  rng <- diff(range(emm_df$emmean, na.rm = TRUE))
  rsd <- sigma(model)

  tibble(
    parameter = param,
    effect_std = rng / rsd,
    effect_raw = rng
  )
}

heat_df <- imap_dfr(
  results_by_objective,
  \(res, obj) {
    map_dfr(params, \(p) effect_strength(res$model, p)) %>%
      mutate(objective = obj)
  }
)

# order objectives and parameters if you want
heat_df <- heat_df %>%
  mutate(
    objective = factor(objective, levels = unique(objective)),
    parameter = factor(parameter, levels = params)
  )


## Compute a distance to the Pareto front, then model that distance
## This code computes distance to the empirical nondominated set in your data.

# 2a. Helper: identify nondominated points
# Assumption: all objectives are to be minimised.
# If some objectives are to be maximised, see "directions" below.

is_nondominated <- function(X) {
  n <- nrow(X)
  nd <- rep(TRUE, n)
  for (i in seq_len(n)) {
    if (!nd[i]) next
    for (j in seq_len(n)) {
      if (i == j) next
      # j dominates i if j is no worse in all objectives and better in at least one
      if (all(X[j, ] <= X[i, ]) && any(X[j, ] < X[i, ])) {
        nd[i] <- FALSE
        break
      }
    }
  }
  nd
}

# 2b. Choose objectives and directions
objs <- c("abiotic_anomaly", "biotic_anomaly")  # edit
directions <- c("min", "min")  # one per objective, use "max" where needed

X <- as.matrix(df[, objs])

# Convert max objectives to min by multiplying by minus one
for (k in seq_along(objs)) {
  if (directions[k] == "max") X[, k] <- -X[, k]
}

# 2c. Normalise objectives so distances are comparable across scales
rng <- apply(X, 2, range, na.rm = TRUE)
Xn <- sweep(X, 2, rng[1, ], "-")
Xn <- sweep(Xn, 2, (rng[2, ] - rng[1, ]), "/")

# 2d. Pareto front as empirical nondominated set
nd_flag <- is_nondominated(Xn)
front <- Xn[nd_flag, , drop = FALSE]

# 2e. Distance to front: nearest point on the front (discrete approximation)
dist_to_front <- function(x, Fr) {
  d2 <- rowSums((Fr - matrix(x, nrow(Fr), ncol(Fr), byrow = TRUE))^2)
  sqrt(min(d2))
}
df$dist_pareto <- apply(Xn, 1, dist_to_front, Fr = front)

# 2f. Model distance, smaller is better
m_dist <- lm(dist_pareto ~ frac + clust + ab + bio, data = df)

# Planned contrasts
#c_p1_dist <- summary(contrast(emmeans(m_dist, ~ frac), "pairwise", adjust = "BH"))
#c_p2_dist <- summary(contrast(emmeans(m_dist, ~ clust), "pairwise", adjust = "BH"))
#c_p3_dist <- summary(contrast(emmeans(m_dist, ~ ab), "pairwise", adjust = "BH"))
#c_p4_dist <- summary(contrast(emmeans(m_dist, ~ bio), "pairwise", adjust = "BH"))

build_effect_rows <- function(model, objective_name) {
  map_dfr(params, \(p) effect_strength(model, p)) %>%
    mutate(objective = objective_name)
}
heat_dist <- build_effect_rows(m_dist, "dist_pareto")

heat_df <- bind_rows(heat_df, heat_dist) %>%
  mutate(
    parameter = factor(parameter, levels = params),
    objective = factor(objective, levels = c(names(results_by_objective), "dist_pareto"))
  )

ggplot(heat_df, aes(x = parameter, y = objective, fill = effect_std)) +
  geom_tile() +
  labs(
    x = "Parameter",
    y = "Objective (including dist_pareto)",
    fill = "Effect strength (range divided by residual sd)",
    title = "Parameter effect strength across objectives and Pareto distance"
  )


library(reticulate)
library(dplyr)
library(tidyr)
library(ggplot2)

# 1) Load pickle
pickle <- import("pickle")
np <- import("numpy", convert = FALSE)

pkl_path <- "results_1301.pkl"  # change

con <- base::file(pkl_path, open = "rb")
res <- pickle$load(con)
close(con)