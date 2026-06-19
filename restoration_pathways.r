library(tidyverse)
library(grid)

# Time sequence
t_start <- 3
t <- seq(0, 30, by = 0.1)

# Starting condition and endpoint assumptions
c0 <- 0
stakeholder_endpoint <- 0.70
minimum_threshold <- 0.50
climate_limited_endpoint <- 0.34

# Helper function: rescaled logistic, starts at c0 and ends at target
delayed_logistic <- function(time, c0, target, midpoint = 16, slope = 0.35) {
  raw <- 1 / (1 + exp(-slope * (time - midpoint)))
  raw_min <- min(raw, na.rm = TRUE)
  raw_max <- max(raw, na.rm = TRUE)
  scaled <- (raw - raw_min) / (raw_max - raw_min)
  c0 + scaled * (target - c0)
}

# Helper function: rescaled exponential, starts at c0 and ends at target
rescaled_exp <- function(time, c0, target, rate) {
  raw <- 1 - exp(-rate * time)
  scaled <- raw / max(raw, na.rm = TRUE)
  c0 + scaled * (target - c0)
}

recovery_df <- tibble(time = t) %>%
  mutate(
    time_since_restoration = pmax(time - t_start, 0),
    
    `Fast recovery` = if_else(
      time < t_start,
      NA_real_,
      rescaled_exp(
        time = time_since_restoration,
        c0 = c0,
        target = stakeholder_endpoint,
        rate = 0.23
      )
    ),
    
    `Gradual recovery` = if_else(
      time < t_start,
      NA_real_,
      rescaled_exp(
        time = time_since_restoration,
        c0 = c0,
        target = stakeholder_endpoint,
        rate = 0.08
      )
    ),
    
    `Delayed recovery` = if_else(
      time < t_start,
      NA_real_,
      delayed_logistic(
        time = time_since_restoration,
        c0 = c0,
        target = stakeholder_endpoint,
        midpoint = 16,
        slope = 0.35
      )
    ),
    
    `Climate limited recovery` = if_else(
      time < t_start,
      NA_real_,
      climate_limited_endpoint -
        (climate_limited_endpoint - c0) * exp(-0.12 * time_since_restoration)
    )
  ) %>%
  select(-time_since_restoration) %>%
  pivot_longer(
    cols = -time,
    names_to = "pathway",
    values_to = "condition"
  )

# Endpoint labels for direct line annotation
end_labels <- recovery_df %>%
  filter(!is.na(condition)) %>%
  group_by(pathway) %>%
  slice_max(time, n = 1, with_ties = FALSE) %>%
  ungroup()

# Label only climate limited recovery
climate_label <- end_labels %>%
  filter(pathway == "Climate limited recovery")

# Plot
pathways <- ggplot(recovery_df, aes(x = time, y = condition, colour = pathway)) +
  annotate(
    "rect",
    xmin = -Inf, xmax = Inf,
    ymin = minimum_threshold, ymax = stakeholder_endpoint,
    alpha = 0.08,
    fill = "steelblue"
  ) +
  geom_hline(
    yintercept = stakeholder_endpoint,
    linetype = "dashed",
    colour = "purple4",
    linewidth = 0.7
  ) +
  geom_hline(
    yintercept = minimum_threshold,
    linetype = "dotted",
    colour = "steelblue4",
    linewidth = 0.7
  ) +
  geom_line(linewidth = 1.1, na.rm = TRUE) +
  geom_point(
    data = tibble(time = t_start, condition = c0),
    aes(x = time, y = condition),
    inherit.aes = FALSE,
    colour = "black",
    size = 3
  ) +
  geom_text(
    data = climate_label,
    aes(
      x = time + 0.5,
      y = condition,
      label = pathway,
      colour = pathway
    ),
    hjust = 0,
    vjust = 0.5,
    size = 3.5,
    show.legend = FALSE
  ) +
  annotate(
    "segment",
    x = t_start + 2,
    xend = t_start + 2,
    y = c0,
    yend = stakeholder_endpoint,
    arrow = arrow(ends = "both", length = unit(0.15, "cm")),
    colour = "purple4"
  ) +
  annotate(
    "text",
    x = t_start + 1.3,
    y = (c0 + stakeholder_endpoint) / 2,
    label = "Recovery\ngap",
    hjust = 1,
    colour = "purple4",
    size = 3.5
  ) +
  annotate(
    "text",
    x = t_start - 0.9,
    y = c0,
    label = "Current\ncondition",
    hjust = 1,
    vjust = 0.5,
    size = 3.5
  ) +
  annotate(
    "text",
    x = max(t) + 0.5,
    y = stakeholder_endpoint + 0.02,
    label = "Aspirational condition",
    hjust = 0,
    colour = "purple4",
    fontface = "bold",
    size = 3.4
  ) +
  annotate(
    "text",
    x = max(t) + 0.5,
    y = minimum_threshold,
    label = "Minimum acceptable\ncondition",
    hjust = 0,
    colour = "steelblue4",
    size = 3.4
  ) +
  coord_cartesian(
    xlim = c(0, max(t) + 8),
    ylim = c(0, 0.8),
    clip = "off"
  ) +
  scale_colour_manual(
    values = c(
      "Fast recovery" = "forestgreen",
      "Gradual recovery" = "steelblue4",
      "Delayed recovery" = "darkorange",
      "Climate limited recovery" = "firebrick"
    )
  ) +
  labs(
    x = "Time",
    y = "Ecosystem condition",
    colour = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.margin = margin(10, 90, 10, 60),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )

ggsave(
  filename = "figs/recovery_pathways.png",
  plot = pathways,
  width = 10,
  height = 5,
  dpi = 300
)