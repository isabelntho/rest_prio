#' Produce visualisations of the EI outputs
#'
#' Input: Files of summary metrics of NCP provision
#' Output: Summary measures of NCP provision across configurations.
#'
#' @environment Summarisation
#' @config $FUTURE_EI_CONFIG_FILE (yaml file)
#' @date 2024-10-15
#' @author Benjamin Black
#'
#'
#' @docType script
#'

# # Create an environment variable for the config file
# Laptop
Sys.setenv(RCD_CONFIG_FILE = "config.yml")

# Desktop
#Sys.setenv(RCD_CONFIG_FILE = "C:/Users/bblack/switchdrive/git_laptop/Robust-conservation-decisions/50_Analysis/config.yml")

Publication_fig_dir <- "publication/figures"

# Load libraries
packs <- c(
  "stringr",
  "terra",
  "readxl",
  "data.table",
  "dtwclust",
  "bigmemory",
  "yaml",
  "tidyr",
  "dplyr",
  "doParallel",
  "foreach",
  "GGally",
  "future",
  "future.apply",
  "ggthemes",
  "tidyterra",
  "classInt",
  "sf",
  "biscale",
  "tidyterra",
  "patchwork",
  "ggfx",
  "extrafont",
  "raster",
  "ggplot2",
  "colorspace"
)


# check which packages are not installed
not_installed <- packs[!(packs %in% installed.packages()[, "Package"])]
# install missing packages
if (length(not_installed)) {
  install.packages(not_installed)
}
# load packages
invisible(lapply(packs, require, character.only = TRUE))

#import fonts
windowsFonts(sans = "Roboto")
loadfonts(device = "win")
loadfonts(device = "postscript")


## Functions #####
#' theme_publication
#'
#' custom ggplot2 theme to apply to all plots
#'
#' @param base_size numeric base font size
#' @param base_family character base font family
#' @return ggplot2 theme
#' @export

theme_publication <- function(
  base_size = 14,
  base_family = "Fira Sans"
) {
  (theme_foundation(
    base_size = base_size,
    base_family = base_family
  ) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5),
      text = element_text(),
      panel.background = element_rect(colour = NA),
      plot.background = element_rect(colour = NA),
      panel.border = element_rect(colour = NA),
      axis.title = element_text(face = "bold", size = rel(0.8)),
      axis.title.y = element_text(angle = 90, vjust = 2),
      axis.title.x = element_text(vjust = -0.2),
      axis.text = element_text(),
      axis.line = element_line(colour = "black"),
      axis.ticks = element_line(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.key = element_rect(colour = NA),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.key.size = unit(0.2, "cm"),
      legend.title = element_text(face = "bold", hjust = 0.5),
      plot.margin = unit(c(10, 5, 5, 5), "mm"),
      strip.background = element_rect(colour = "#f0f0f0", fill = "#f0f0f0"),
      strip.text = element_text(face = "bold")
    ))
}

#' thesis_color
#'
#' Custom colour palette to be used in document
#'
#'
thesis_color <- function(...) {
  thesis_colors <- c(
    'Seafoam' = "#86B4A4",
    'Beigetan' = "#AC9E7C",
    'Lightgreen' = "#8D9973",
    'Olivegreen' = "#4F5838",
    'Forestgreen' = "#2D4631",
    'Emeraldgreen' = "#20342B",
    'Bluegreen' = "#1C2D27",
    'Brown' = "#413328",
    'Redbrown' = "#4E3028",
    'Rust' = "#8A341F",
    'Terracotta' = "#A96524",
    'Orange' = "#C06F2E",
    'Beige' = "#D0A26E",
    'Paleorange' = "#D69D68",
    'Bone' = "#F6E3C2",
    'Tan' = "#BDA776",
    'Mustard' = "#B77D2B",
    'Ochre' = "#9A792E"
  )

  cols <- c(...)

  if (is.null(cols)) {
    return(thesis_colors)
  }

  thesis_colors[cols]
}

gen_discrete_ramp <- function(n, org_colour) {
  # vector name of color
  col_name <- names(org_colour)

  # create empty vector to store colours
  cols <- c()

  # loop over n-clusters (i..e number of required colours)
  for (i in 1:n) {
    # if i is 1 then the first colour is the scenario colour
    if (i == 1) {
      cols[i] <- c(org_colour)
    } else {
      #use the preceding colour to generate a lighter variant
      cols[i] <- colorspace::lighten(cols[i - 1], amount = 0.2)
    }
  }

  names(cols) <- sapply(1:n, function(x) paste0(col_name, x))
  return(cols)
}

# apply function to generate discrete colour ramps for all colours in thesis_color
discrete_ramps <- lapply(names(thesis_color()), function(x) {
  # generate the discrete colour variants
  ramp <- gen_discrete_ramp(n = 5, org_colour = thesis_color(x))
})
names(discrete_ramps) <- paste0(names(thesis_color()), "_ramp")


#' thesis_palette
#'
#' Function to combined defined colours into palettes
#'
#' @param palette character: palette name

thesis_palette <- function(palette = "main", ...) {
  thesis_palettes <- c(
    discrete_ramps,
    list(
      All = thesis_color(),
      'CAPAADC' = c(
        "control" = unname(thesis_color("Rust")),
        "PAs no ELCs" = unname(thesis_color("Emeraldgreen")),
        "PAs with ELCs" = unname(thesis_color("Mustard"))
      ),
      'ICoLCM' = c(
        discrete_ramps[["Mustard_ramp"]][1],
        discrete_ramps[["Mustard_ramp"]][3],
        discrete_ramps[["Bluegreen_ramp"]][1],
        discrete_ramps[["Bluegreen_ramp"]][3]
      ),
      'ICoLCM_FS' = thesis_color('Redbrown', "Rust", "Terracotta"),
      'Scenarios' = c(
        "BAU" = unname(discrete_ramps[['Tan_ramp']][3]),
        "BAU-CC" = unname(discrete_ramps[['Redbrown_ramp']][3]),
        "EI-SOC" = unname(discrete_ramps[['Seafoam_ramp']][1]),
        "EI-CUL" = unname(discrete_ramps[["Mustard_ramp"]][3]),
        "EI-NAT" = unname(discrete_ramps[['Forestgreen_ramp']][2])
      ),
      'NCPs' = c(
        "CAR" = unname(discrete_ramps[['Olivegreen_ramp']][3]),
        "FF" = unname(discrete_ramps[['Ochre_ramp']][3]),
        "HAB" = unname(discrete_ramps[['Lightgreen_ramp']][3]),
        "ID" = unname(discrete_ramps[['Emeraldgreen_ramp']][3]),
        "NDR" = unname(discrete_ramps[['Brown_ramp']][3]),
        "PC" = unname(discrete_ramps[['Rust_ramp']][3]),
        "POL" = unname(discrete_ramps[['Orange_ramp']][3]),
        "REC" = unname(discrete_ramps[['Bone_ramp']][3]),
        "SDR" = unname(discrete_ramps[['Terracotta_ramp']][3]),
        "WY" = unname(discrete_ramps[['Seafoam_ramp']][3])
      )
    )
  )

  thesis_palettes[[palette]]
}

# To view your colour palette
# scales::show_col(thesis_palette("Scenarios"), cex_label = 2)

#' palette_gen
#'
#' Helper function to create discrete colour scales
#'
#' @param palette Character: name of the palette from thesis_palette()
#' @param direction Numeric to indicate direction of the scale

palette_gen <- function(palette = "main", direction = 1) {
  function(n) {
    if (n > length(thesis_palette(palette))) {
      warning("Not enough colors in this palette!")
    } else {
      all_colors <- thesis_palette(palette)

      all_colors <- unname(unlist(all_colors))

      all_colors <- if (direction >= 0) all_colors else rev(all_colors)

      color_list <- all_colors[1:n]
    }
  }
}

#' palette_gen_c
#'
#' Helper function to create continuous colour scales
#'
#' @param palette Character: name of the palette from thesis_palette()
#' @param direction Numeric to indicate direction of the scale

palette_gen_c <- function(palette = "main", direction = 1, ...) {
  pal <- thesis_palette(palette)

  pal <- if (direction >= 0) pal else rev(pal)

  colorRampPalette(pal, ...)
}


#' scale_fill_thesis
#'
#' ggplot2 wrapper function to apply a fill argument using discrete scale produced using palette_gen()
#'
#' @param palette Character: name of the palette from thesis_palette()
#' @param direction Numeric to indicate direction of the scale

scale_fill_thesis <- function(palette = "main", direction = 1, ...) {
  ggplot2::discrete_scale(
    "fill",
    "thesis",
    palette_gen(palette, direction),
    ...
  )
}

#' scale_colour_thesis
#'
#' ggplot2 wrapper function to apply a colour argument using discrete scale produced using palette_gen()
#'
#' @param palette Character: name of the palette from thesis_palette()
#' @param direction Numeric to indicate direction of the scale

scale_colour_thesis <- function(palette = "main", direction = 1, ...) {
  ggplot2::discrete_scale(
    "colour",
    "thesis",
    palette_gen(palette, direction),
    ...
  )
}

# replicate function using both spellings of colour (conventional approach)
scale_color_thesis <- scale_colour_thesis


#' scale_colour_thesis_c
#'
#' ggplot2 wrapper function to apply a colour argument using continuous scale produced using palette_gen()
#'
#' @param palette Character: name of the palette from thesis_palette()
#' @param direction Numeric to indicate direction of the scale

scale_colour_thesis_c <- function(palette = "main", direction = 1, ...) {
  pal <- palette_gen_c(palette = palette, direction = direction)

  scale_color_gradientn(colors = pal(256), ...)
}

# replicate function using both spellings of colour (conventional approach)
scale_color_thesis_c <- scale_colour_thesis_c

#' Cluster_colours
#'
#' Small function to define a vector of colours for the clusters under each scenario
#' for n clusters generate n colours, the 1st of which is the original scenario colour
#' and the others are lighter variants.
#'
#' @param n_clusters Number of clusters
#' @param scenario Name of scenario to generate cluster colours for
#' @param Scenario_pal Named vector of colours for each scenario
#' @return Vector of colours of length n_clusters

Cluster_colours <- function(n_clusters, scenario, Scenario_pal) {
  # Subset Scenario_pal to the scenario of interest to get colour
  scenario_colour <- Scenario_pal[names(Scenario_pal) == scenario]

  # create empty vector to store colours
  cols <- c()

  # loop over n-clusters (i..e number of required colours)
  for (i in 1:n_clusters) {
    # if i is 1 then the first colour is the scenario colour
    if (i == 1) {
      cols[i] <- c(scenario_colour)
    } else {
      #use the preceding colour to generate a lighter variant
      cols[i] <- colorspace::lighten(cols[i - 1], amount = 0.2)
    }
  }
  return(cols)
}

#' Tabular_summary
#' Produce a tabular summary across all of the summary metrics for each NCP and scenario
#'
#' @param Results_data List of dataframes containing the summary metrics for each NCP
#' @param Summary_metrics Vector of summary metrics to calculate
#' @param NCPs_to_visualise Vector of NCPs to visualise
#' @param Scenarios_to_visualise Vector of scenarios to visualise
#' @param tabular_output_dir Directory to save the tabular summary output
#' @param Value_scaling Vector of value scaling options to calculate summary stats for

Tabular_summary <- function(
  Results_data = Results_data,
  Summary_metrics = config$Summary_metrics,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Scenarios_to_visualise = config$Scenarios_to_visualise,
  tabular_output_dir = tabular_output_dir,
  Value_scaling = c("unscaled", "rescaled")
) {
  # Loop over value scaling options
  sapply(Value_scaling, function(scale_option) {
    # Loop over metrics, scenarios and NCPs to calculate summary stats for each
    Summary_stats <- lapply(Summary_metrics, function(metric) {
      # Select dataframe based on scale option and metric
      if (
        metric %in%
          c(
            "Avg_pos_change",
            "Avg_neg_change",
            "Std_pos_change",
            "Std_neg_change"
          )
      ) {
        Dataset_tag <- "NCP_change_stats"
      } else if (metric %in% c("sum", "mean", "sd")) {
        Dataset_tag <- "NCP_sum_stats"
      } else if (metric %in% c("SSIM", "SIM", "SIV", "SIP")) {
        Dataset_tag <- "NCP_pattern_stats"
      } else if (metric %in% c("Agg_metric", "Agg_metric_rescale")) {
        Dataset_tag <- "NCP_Agg_stats"
      }

      # if scale_option is rescaled then use the rescaled dataset
      if (scale_option == "rescaled") {
        paste0(Dataset_tag, "_rescale")
      }

      # separate the dataset from the list
      NCP_stats <- Results_data[[Dataset_tag]]

      # Pivot NCP sum stats to wide format based on Time_steps using the metric as the
      #value and dropping other columns
      NCP_stats_time_wide <- NCP_stats %>%
        pivot_wider(
          names_from = Time_step,
          values_from = metric,
          id_cols = c(Config_ID, NCP, Scenario)
        )

      # Loop over scenarios
      Scenario_results <- lapply(Scenarios_to_visualise, function(scenario) {
        # Loop over NCPs
        NCP_results <- lapply(NCPs_to_visualise, function(NCP) {
          # subset to just NCP == NCP and scenario == scenario
          NCP_scenario <- NCP_stats_time_wide[
            NCP_stats_time_wide$NCP == NCP &
              NCP_stats_time_wide$Scenario == scenario,
          ]

          # for each value in Time_step calculate the min, max, std and
          # inter-quartile range of the sum of NCP for the current metric
          NCP_scenario_long <- NCP_scenario %>%
            pivot_longer(
              cols = -c(Config_ID, NCP, Scenario),
              names_to = "Time_step",
              values_to = "Value"
            ) %>%
            group_by(Time_step) %>%
            summarise(
              Min = min(Value, na.rm = TRUE),
              Max = max(Value, na.rm = TRUE),
              Mean = mean(Value, na.rm = TRUE),
              Sd = sd(Value, na.rm = TRUE),
              IQR = IQR(Value, na.rm = TRUE)
            )

          # Add NCP and Scenario to the data frame
          NCP_scenario_long$NCP <- NCP
          NCP_scenario_long$Scenario <- scenario

          return(NCP_scenario_long)
        })
        names(NCP_results) <- NCPs_to_visualise
        return(NCP_results)
      })
      names(Scenario_results) <- Scenarios_to_visualise
      return(Scenario_results)
    })
    names(Summary_stats) <- Summary_metrics

    # Save the summary stats
    if (scale_option == "unscaled") {
      saveRDS(
        Summary_stats,
        file.path(tabular_output_dir, "NCP_summary_stats.rds")
      )
    } else {
      saveRDS(
        Summary_stats,
        file.path(tabular_output_dir, "NCP_summary_stats_rescaled.rds")
      )
    }
  })
}


#' Static_analysis_plots
#'
#' Function to create parallel coordinates plots of the summary metrics
#' of rescaled NCP values, depending on the metric the resulting value is either
#' an average over time, from the final time step, or the value of the
#'  initial time step minus the final step. The value scaling option allows
#'  for plots to be produced on unscaled and rescaled values.
#'
#'  @param Value_scaling Vector of value scaling options
#'  @param Summary_metrics Vector of summary metrics to plot
#'  @param Static_plots_dir Directory to save plots to
#'  @param Results_data List of dataframes containing the results of the analysis
#'  @param NCPs_to_visualise Vector of NCPs to visualise
#'  @param Scenario_pal Named vector of colours for each scenario
#'  @param Clean_metric_names Named vector of clean metric names

Static_analysis_plots <- function(
  Value_scaling = c("unscaled", "rescaled"),
  Summary_metrics = config$Summary_metrics,
  Static_plots_dir = Static_plots_dir,
  Results_data = Results_data,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Scenario_pal = Scenario_pal,
  Clean_metric_names = Clean_metric_names
) {
  # Loop over value scaling options
  lapply(Value_scaling, function(scale_option) {
    cat("Creating static analysis plots under scale option", scale_option, "\n")

    # Loop over metrics producing a plot for each
    lapply(Summary_metrics, function(metric) {
      cat(
        "Creating static analysis plots under scale option: ",
        scale_option,
        " for metric: ",
        metric,
        "\n"
      )

      # Select dataframe based on scale option and metric
      if (
        metric %in%
          c(
            "Avg_pos_change",
            "Avg_neg_change",
            "Std_pos_change",
            "Std_neg_change"
          )
      ) {
        Dataset_tag <- "NCP_change_stats"
      } else if (metric %in% c("sum", "mean", "sd")) {
        Dataset_tag <- "NCP_sum_stats"
      } else if (metric %in% c("SSIM", "SIM", "SIV", "SIP")) {
        Dataset_tag <- "NCP_pattern_stats"
      } else if (metric %in% c("Agg_metric", "Agg_metric_rescale")) {
        Dataset_tag <- "NCP_Agg_stats"
      }

      # if scale_option is rescaled then use the rescaled dataset
      if (scale_option == "rescaled") {
        paste0(Dataset_tag, "_rescale")
      }

      # separate the dataset from the list
      NCP_stats <- Results_data[[Dataset_tag]]

      # remove the other metrics from the dataframe
      NCP_stats <- NCP_stats[, c(info_cols, metric)]

      #rename the metric column metric
      names(NCP_stats)[names(NCP_stats) == metric] <- "metric"

      # Get minimum and maximum time steps
      min_time <- as.character(min(NCP_stats$Time_step))
      max_time <- as.character(max(NCP_stats$Time_step))

      # for the summary metrics (i.e. those in NCP_sum_stats) it makes sense to
      # substract the start value from the end value to get the change in the metric
      if (Dataset_tag == "NCP_sum_stats") {
        # Filter the NCP_sum_stats to only include Time steps that are either
        # the min or max time step
        NCP_stats_start_end <- NCP_stats %>%
          dplyr::filter(Time_step == min_time | Time_step == max_time)

        # For each config_ID and each NCP subtract the values of the metric for 2020
        # from the values for 2060 and pivot to wide based on NCP.
        NCP_stats <- NCP_stats_start_end %>%
          group_by(Config_ID, NCP) %>%
          summarise(
            diff = metric[Time_step == max_time] -
              metric[Time_step == min_time],
            Scenario = first(Scenario)
          )

        # Remove metric column
        NCP_stats$metric <- NULL

        # Rename diff column to metric
        names(NCP_stats)[names(NCP_stats) == "diff"] <- "metric"
      } else if (Dataset_tag == "NCP_change_stats") {
        # For the NCP change stats it makes sense to get an average
        # of the value across the time points

        NCP_stats <- NCP_stats %>%
          group_by(Config_ID, NCP) %>%
          summarise(
            metric = mean(metric, na.rm = TRUE),
            Scenario = first(Scenario)
          )
      } else if (Dataset_tag == "NCP_pattern_stats") {
        # If the metric is one of the pattern stats then it makes sense to only take
        # the 2060 value as this is already calculated with reference to the 2020 value

        # subset to only values of the maximum time step
        NCP_stats <- NCP_stats %>% dplyr::filter(Time_step == max_time)
      } else if (Dataset_tag == "NCP_Agg_stats") {
        # Subset to the highest value in Time_step and then select only relevant columns
        NCP_stats <- NCP_stats %>%
          dplyr::filter(Time_step == max_time) %>%
          dplyr::select(Config_ID, NCP, Scenario, metric)
      }

      # Now rescale the values to be between 0 and 1
      NCP_stats$metric <- (NCP_stats$metric - min(NCP_stats$metric)) /
        (max(NCP_stats$metric) - min(NCP_stats$metric))

      # Pivot to wide format
      NCP_stats_wide <- NCP_stats %>%
        pivot_wider(
          names_from = NCP,
          values_from = metric,
          id_cols = c(Config_ID, Scenario)
        )

      # Get the column indices of NCPs in the dataframe
      NCP_cols <- which(names(NCP_stats_wide) %in% NCPs_to_visualise)

      # get the clean name of the current metric
      clean_metric <- Clean_metric_names[which(
        names(Clean_metric_names) == metric
      )]

      # Create a custom Y axis label based on the metric
      if (Dataset_tag == "NCP_sum_stats") {
        y_lab <- paste0(
          clean_metric,
          " of provision in ",
          max_time,
          " minus ",
          min_time
        )
      } else if (Dataset_tag == "NCP_change_stats") {
        y_lab <- paste0(clean_metric, " averaged over all time steps")
      } else if (Dataset_tag == "NCP_pattern_stats") {
        y_lab <- paste0(clean_metric, " value in ", max_time)
      } else if (Dataset_tag == "NCP_Agg_stats") {
        y_lab <- paste0(clean_metric, " of service in ", max_time)
      }

      # create a box plot
      box_p <- ggplot(NCP_stats, aes(x = NCP, y = metric)) +
        geom_boxplot(
          aes(color = Scenario, fill = Scenario),
          position = position_dodge(width = 0.6),
          outlier.shape = NA,
          size = 0.5,
          width = 0.6
        ) +
        geom_jitter(
          aes(color = Scenario),
          position = position_jitterdodge(
            jitter.width = 0.1,
            dodge.width = 0.6
          ),
          size = 0.5
        ) +
        scale_x_discrete(
          limit = c(
            'ID',
            'PC',
            'CAR',
            'FF',
            'HAB',
            'NDR',
            'POL',
            'REC',
            'SDR',
            'WY'
          )
        ) +
        scale_color_manual(
          values = alpha(c(thesis_palette('Scenarios')), 0.5)
        ) +
        scale_fill_manual(values = alpha(c(thesis_palette('Scenarios')), 0.2)) + # Adjust transparency here
        labs(x = "Ecosystem Service Indicator", y = y_lab) +
        theme_publication()

      # Create a parallel coordinates plot of the metric across the NCPs
      pc <- ggparcoord(
        NCP_stats_wide,
        columns = NCP_cols,
        groupColumn = "Scenario",
        showPoints = FALSE,
        alphaLines = 0.3,
        scale = "globalminmax",
        boxplot = FALSE,
      ) +
        labs(x = "Ecosystem Service Indicator", y = y_lab) +
        scale_x_discrete(
          limit = c(
            'ID',
            'PC',
            'CAR',
            'FF',
            'HAB',
            'NDR',
            'POL',
            'REC',
            'SDR',
            'WY'
          )
        ) +
        scale_color_manual(values = alpha(c(thesis_palette('Scenarios')), .8)) +
        theme_publication()

      # Create path to save plot
      if (Dataset_tag == "NCP_sum_stats") {
        plot_path <- file.path(
          Static_plots_dir,
          paste0("Metric_", metric, "_", max_time, "_minus_", min_time)
        )
      } else if (Dataset_tag == "NCP_change_stats") {
        plot_path <- file.path(
          Static_plots_dir,
          paste0("Metric_", metric, "_average_over_time")
        )
      } else if (Dataset_tag == "NCP_pattern_stats") {
        plot_path <- file.path(
          Static_plots_dir,
          paste0("Metric_", metric, "_2060")
        )
      } else if (Dataset_tag == "NCP_Agg_stats") {
        plot_path <- file.path(
          Static_plots_dir,
          paste0("Metric_", metric, "_", max_time, "_minus_", min_time)
        )
      }

      #Add a tag for unsclaed or rescaled
      if (scale_option == "unscaled") {
        plot_path <- paste0(plot_path, "_unscaled.png")
      } else if (scale_option == "rescaled") {
        plot_path <- paste0(plot_path, "_rescaled.png")
      }

      # seperate plot paths for the box plot and parallel coords plot
      box_plot_path <- gsub(".png", "_boxplot.png", plot_path)
      pc_plot_path <- gsub(".png", "_parallel_coords.png", plot_path)

      # save box plot
      ggsave(
        box_plot_path,
        box_p,
        dpi = 300,
        width = 30,
        height = 21,
        units = "cm"
      )

      # save parallel coords plot
      ggsave(pc_plot_path, pc, dpi = 300, width = 30, height = 21, units = "cm")

      #also save the ggplot objects as an rds file

      # replace the '.png' with '.rds'
      ggplot_box_path <- gsub(".png", ".rds", box_plot_path)
      ggplot_pc_path <- gsub(".png", ".rds", pc_plot_path)

      # save the ggplot object
      saveRDS(box_p, ggplot_box_path)
      saveRDS(pc, ggplot_pc_path)
    })
  })
}

#' Dynamic_analysis_plots
#'
#' This function creates parallel coordinates plots of summary metrics
#' of NCP provision over time. The value_scaling argument allows the user to
#' specify whether to use unscaled or rescaled values. The NCP_aggregation argument
#' allows the user to specify whether to produce plots for individual NCPs or
#' plots aggregated across NCPs.
#'
#' @param Value_scaling A character vector specifying whether to use unscaled or rescaled values
#' @param Summary_metrics A character vector specifying the metrics to plot
#' @param Dynamic_plots_dir A character string specifying the directory to save the plots
#' @param Results_data A list of dataframes containing the results of the analysis
#' @param NCPs_to_visualise A character vector specifying the NCPs to visualise
#' @param Scenario_pal A character vector specifying the colour palette for the scenarios
#' @param Clean_metric_names A named character vector specifying the clean names of the metrics
#' @param NCP_aggregation A character string: with options "Individual", "Aggregated" or "Both"
#'  specifying whether to produce plots for individual NCPs or aggregated NCPs or both.
#'

Dynamic_analysis_plots <- function(
  Value_scaling = c("unscaled", "rescaled"),
  Summary_metrics = config$Summary_metrics,
  Dynamic_plots_dir = Dynamic_plots_dir,
  Results_data = Results_data,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Scenario_pal = Scenario_pal,
  Clean_metric_names = Clean_metric_names,
  NCP_aggregation = "Both" # Options: "Individual", "Aggregated", "Both"
) {
  # Loop over value scaling options
  sapply(Value_scaling, function(scale_option) {
    cat(
      "Creating dynamic analysis plots under scale option",
      scale_option,
      "\n"
    )

    # Loop over metrics producing a plot for each
    sapply(Summary_metrics, function(metric) {
      cat(
        "Creating dynamic analysis plots under scale option: ",
        scale_option,
        " for metric: ",
        metric,
        "\n"
      )

      # Select dataframe based on scale option and metric
      if (
        metric %in%
          c(
            "Avg_pos_change",
            "Avg_neg_change",
            "Std_pos_change",
            "Std_neg_change"
          )
      ) {
        Dataset_tag <- "NCP_change_stats"
      } else if (metric %in% c("sum", "mean", "sd")) {
        Dataset_tag <- "NCP_sum_stats"
      } else if (metric %in% c("SSIM", "SIM", "SIV", "SIP")) {
        Dataset_tag <- "NCP_pattern_stats"
      } else if (metric %in% c("Agg_metric", "Agg_metric_rescale")) {
        Dataset_tag <- "NCP_Agg_stats"
      }

      # if scale_option is rescaled then use the rescaled dataset
      if (scale_option == "rescaled") {
        paste0(Dataset_tag, "_rescale")
      }

      # separate the dataset from the list
      NCP_stats <- Results_data[[Dataset_tag]]

      # remove the other metrics from the dataframe
      NCP_stats <- NCP_stats[, c(info_cols, metric)]

      #rename the metric column metric
      names(NCP_stats)[names(NCP_stats) == metric] <- "metric"

      # Get the unique time steps
      Time_steps <- unique(NCP_stats$Time_step)

      # get the clean name of the current metric
      clean_metric <- Clean_metric_names[names(Clean_metric_names == metric)]

      # the current metric is not for aggregated NCPs then proceed by looping over individual NCPs
      if (NCP_aggregation == "Individual" | NCP_aggregation == "Both") {
        # Loop over NCPs producing a plot for each
        lapply(NCPs_to_visualise, function(NCP) {
          # create a dir for this NCP results
          NCP_dir <- file.path(Dynamic_plots_dir, NCP)
          dir.create(NCP_dir, showWarnings = FALSE)

          # Filter NCP_stats to current NCP
          NCP_stats <- NCP_stats[NCP_stats$NCP == NCP, ]

          # Rescale the metric column using min max
          NCP_stats$metric <- (NCP_stats$metric - min(NCP_stats$metric)) /
            (max(NCP_stats$metric) - min(NCP_stats$metric))

          # Pivot the dataframe to wide format
          NCP_stats <- NCP_stats %>%
            pivot_wider(names_from = Time_step, values_from = metric)

          # Get the column indices of Time_steps in the dataframe
          Time_cols <- which(names(NCP_stats) %in% Time_steps)

          # Create a custom Y axis label based on the metric
          if (Dataset_tag == "NCP_sum_stats") {
            y_lab <- paste0(clean_metric, "of provision of ES indicator: ", NCP)
          } else if (Dataset_tag == "NCP_change_stats") {
            y_lab <- paste0(
              clean_metric,
              " of provision of ES indicator: ",
              NCP,
              " between time steps"
            )
          } else if (Dataset_tag == "NCP_pattern_stats") {
            y_lab <- paste0(
              clean_metric,
              " value of time step with reference to 2020"
            )
          } else if (Dataset_tag == "NCP_Agg_stats") {
            y_lab <- paste0(clean_metric, " on ES indicator: ", NCP)
          }

          # Create a parallel coordinates plot of the difference in NCP provision between 2020 and 2060
          p <- ggparcoord(
            NCP_stats,
            columns = Time_cols,
            groupColumn = "Scenario",
            showPoints = FALSE,
            alphaLines = 0.3,
            scale = "globalminmax",
            boxplot = FALSE
          ) +
            labs(x = "Time", y = y_lab) +
            scale_color_manual(values = Scenario_pal) +
            theme_Publication()

          if (scale_option == "unscaled") {
            plot_path <- file.path(
              NCP_dir,
              paste0(NCP, "_", metric, "_dynamic_plot_unscaled.png")
            )
          } else {
            plot_path <- file.path(
              NCP_dir,
              paste0(NCP, "_", metric, "_dynamic_plot_rescaled.png")
            )
          }

          # save plot
          ggsave(plot_path, p, dpi = 300, width = 30, height = 21, units = "cm")
        })
      } else if (NCP_aggregation == "Aggregated" | NCP_aggregation == "Both") {
        # create a dir for the aggregated NCP results
        Dynamic_agg_dir <- file.path(Dynamic_plots_dir, "Aggregated")
        if (!dir.exists(Dynamic_agg_dir)) {
          dir.create(Dynamic_agg_dir, showWarnings = FALSE)
        }

        # loop over the config_IDs and take an average of the metric across the NCPs
        NCP_stats <- NCP_stats %>%
          group_by(Config_ID, Time_step, Scenario) %>%
          summarise(metric = mean(metric))

        # normalise the metric values using their min and max without grouping
        NCP_stats$metric <- sapply(NCP_stats$metric, function(x) {
          (x - min(NCP_stats$metric)) /
            (max(NCP_stats$metric) - min(NCP_stats$metric))
        })

        # Pivot to wide format
        NCP_stats <- NCP_stats %>%
          pivot_wider(
            names_from = Time_step,
            values_from = metric,
            id_cols = c(Config_ID, Scenario)
          )

        # Get the column indices of Time_steps in the dataframe
        Time_cols <- which(names(NCP_stats) %in% Time_steps)

        # Create a custom Y axis label based on the metric
        if (Dataset_tag == "NCP_sum_stats") {
          y_lab <- paste0(
            clean_metric,
            "of provision aggregated across all ES indicators"
          )
        } else if (Dataset_tag == "NCP_change_stats") {
          y_lab <- paste0(
            clean_metric,
            " of provision aggregated across all ES indicators"
          )
        } else if (Dataset_tag == "NCP_pattern_stats") {
          y_lab <- paste0(
            clean_metric,
            " value of time step with reference to 2020, aggregated across all ES indicators"
          )
        } else if (Dataset_tag == "NCP_Agg_stats") {
          y_lab <- paste0(
            clean_metric,
            " on ES indicator aggregated across all ES indicators"
          )
        }

        # Create a parallel coordinates plot of the difference in NCP provision between 2020 and 2060
        p <- ggparcoord(
          NCP_stats,
          columns = Time_cols,
          groupColumn = "Scenario",
          showPoints = FALSE,
          alphaLines = 0.3,
          scale = "globalminmax",
          boxplot = FALSE
        ) +
          labs(x = "Time", y = y_lab) +
          scale_color_manual(values = Scenario_pal) +
          theme_Publication()

        if (scale_option == "unscaled") {
          plot_path <- file.path(
            Dynamic_agg_dir,
            paste0("Aggregated_NCPs_", metric, "_dynamic_plot_unscaled.png")
          )
        } else {
          plot_path <- file.path(
            Dynamic_agg_dir,
            paste0("Aggregated_NCPs_", metric, "_dynamic_plot_rescaled.png")
          )
        }

        # save plot
        ggsave(plot_path, p, dpi = 300, width = 30, height = 21, units = "cm")
      }
    })
  })
}


#' Undesirable_deviation
#'
#' Helper function used by Summarise_spatial_outputs to calculates the
#' undesirable deviation metric (Kwakkel et al. 2016) for a given set of values.
#' The undesirable deviation metric is a measure of robustness that incorporates
#' both a measure of performance and variation variance by calculating the
#' sum of the deviation (regret) away from the median performance value for all
#' instances below the median.
# (which is considered the expected value).
#'
#' @param f A numeric vector of values for which to calculate the undesirable deviation metric
#'
#' @references Kwakkel, Jan H., Sibel Eker, and Erik Pruyt. 2016.
#' ‘How Robust Is a Robust Policy? Comparing Alternative Robustness Metrics for
#'  Robust Decision-Making’. In Robustness Analysis in Decision Aiding,
#'  Optimization, and Analytics, edited by Michael Doumpos,
#'  Constantin Zopounidis, and Evangelos Grigoroudis, 221–37.
#'  Cham: Springer International Publishing.
#'  https://doi.org/10.1007/978-3-319-33121-8_10.

Undesirable_deviation <- function(f) {
  # calculating regret from median
  median <- median(f)

  # subtract the median from the value
  regret <- f - median

  # Now work with worst half of values

  # sort regret values
  regret <- sort(regret)

  # get the number of values
  n <- length(regret)

  # get the number of half of the values rounded up to nearest integer
  n_half <- round(n / 2 + 0.51)

  # get the worst half of the values
  worst_regret <- regret[1:n_half]

  # get the sum of the worst half of the values
  sum_worst_regret <- sum(worst_regret)

  return(sum_worst_regret)
}


#' Summarise_spatial_outputs
#'
#' Produce summary measures of the rasters of sum of change over the time steps
#' of NCPs under each configuration

Summarise_spatial_outputs <- function(
  change_raster_dir,
  Spatial_summary_dir,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Aggregate_NCPS = TRUE,
  Spatial_summary_metrics = c(
    "mean",
    "stdev",
    "mean-variance",
    "undesirable_deviation"
  ),
  Parallel = TRUE,
  Ref_raster_path
) {
  # Load the ref_grid to set extent and crs
  ref_grid <- rast(Ref_raster_path)

  # List all tif raster paths
  rast_paths <- list.files(
    change_raster_dir,
    pattern = ".tif",
    full.names = TRUE
  )

  # subset to only those that contain the NCPs to visualise
  rast_paths <- rast_paths[grep(
    paste(NCPs_to_visualise, collapse = "|"),
    rast_paths
  )]

  # Check the geometries of the layers to make sure that they can all be stacked,
  # i.e. they have matching res, extent and crs
  suppressWarnings(
    Geom_check <- future_sapply(rast_paths, function(x) {
      r <- rast(x)
      r_comp <- compareGeom(
        x = ref_grid,
        y = r,
        ext = TRUE,
        crs = TRUE,
        warncrs = FALSE,
        stopOnError = FALSE
      )

      return(r_comp)
    })
  )
  message(" Completed geometry check of rasters...")

  # Convert Geom_check to df
  Geom_check_df <- as.data.frame(Geom_check)

  # Convert row names to column called path
  Geom_check_df$path <- rownames(Geom_check_df)

  # set parallel processing if specified
  if (Parallel) {
    future::plan(future::multisession)
  } else {
    future::plan(future::sequential)
  }

  # Remove row names
  rownames(Geom_check_df) <- NULL

  # If any entries in path == FALSE then the layers may need to be adjusted
  if (!all(Geom_check_df$Geom_check)) {
    message(
      " Some rasters have incorrect geometries, attempting to reproject..."
    )

    # Subset to to only Geom_check == FALSE
    Incorrect_geom_paths <- Geom_check_df[
      Geom_check_df$Geom_check == FALSE,
    ]$path

    # Loop over the rasters with incorrect geoms and reproject them
    suppressWarnings(future_sapply(Incorrect_geom_paths, function(x) {
      #Load ref_grid
      ref_grid <- rast(Ref_raster_path)

      # Load raster
      r <- rast(x)

      # check if extent matches Ref_grid
      ext_check <- compareGeom(
        x = ref_grid,
        y = r,
        ext = TRUE,
        stopOnError = FALSE,
        warncrs = FALSE,
        messages = TRUE
      )

      # check if crs matches Ref_grid
      crs_check <- compareGeom(
        x = ref_grid,
        y = r,
        crs = TRUE,
        stopOnError = FALSE,
        warncrs = FALSE,
        messages = TRUE
      )

      # if either is incorrect then reprojection is needed
      if (ext_check == FALSE | crs_check == FALSE) {
        r_proj <- project(r, ref_grid)
        writeRaster(r_proj, x, overwrite = TRUE)
      }
    }))
  }

  # stack all of the rasters
  rast_stk <- rast(rast_paths)

  if (any(c("mean", "mean-variance") %in% Spatial_summary_metrics)) {
    # calculate the mean across the raster layers
    rast_mean <- mean(rast_stk)
    mean_min <- minmax(rast_mean)[1]
    mean_max <- minmax(rast_mean)[2]

    # rescale values according to min max
    rast_mean_rescaled <- (rast_mean - mean_min) / (mean_max - mean_min)

    # Save the mean result
    writeRaster(
      rast_mean,
      file.path(Spatial_summary_dir, "Mean_sum_of_change_all_NCPs.tif"),
      overwrite = TRUE
    )
    writeRaster(
      rast_mean_rescaled,
      file.path(
        Spatial_summary_dir,
        "Mean_sum_of_change_all_NCPs_rescaled.tif"
      ),
      overwrite = TRUE
    )
  }

  if (any(c("stdev", "mean-variance") %in% Spatial_summary_metrics)) {
    # calculate the standard deviation
    rast_stdev <- stdev(rast_stk)
    stdev_min <- minmax(rast_stdev)[1]
    stdev_max <- minmax(rast_stdev)[2]

    # rescale values according to min max
    rast_stdev_rescaled <- (rast_stdev - stdev_min) / (stdev_max - stdev_min)

    # Save the stdev result
    writeRaster(
      rast_stdev,
      file.path(Spatial_summary_dir, "Stdev_sum_of_change_all_NCPs.tif"),
      overwrite = TRUE
    )
    writeRaster(
      rast_stdev_rescaled,
      file.path(
        Spatial_summary_dir,
        "Stdev_sum_of_change_all_NCPs_rescaled.tif"
      ),
      overwrite = TRUE
    )
  }

  if ("mean-variance" %in% Spatial_summary_metrics) {
    # Calculate the mean variance,the +1 is included to handle situations where
    # either the mean or standard deviation is close to zero.
    rast_mean_var <- (rast_mean + 1) / (rast_stdev + 1)

    # rescale values according to min max
    mean_var_min <- minmax(rast_mean_var)[1]
    mean_var_max <- minmax(rast_mean_var)[2]

    rast_mean_var_rescaled <- (rast_mean_var - mean_var_min) /
      (mean_var_max - mean_var_min)

    # Save the mean variance result
    writeRaster(
      rast_mean_var,
      file.path(
        Spatial_summary_dir,
        "Mean_variance_sum_of_change_all_NCPs.tif"
      ),
      overwrite = TRUE
    )
    writeRaster(
      rast_mean_var_rescaled,
      file.path(
        Spatial_summary_dir,
        "Mean_variance_sum_of_change_all_NCPs_rescaled.tif"
      ),
      overwrite = TRUE
    )
  }

  # Do a clean up of already computed layers to free up memory
  suppressWarnings(
    rm(
      rast_mean,
      rast_mean_rescaled,
      rast_stdev,
      rast_stdev_rescaled,
      rast_mean_var,
      rast_mean_var_rescaled
    )
  )

  if ("undesirable-deviation" %in% Spatial_summary_metrics) {
    message("Calculating undesirable deviation spatial summary metric...")

    # Calculate the undesirable deviation applying the function to all cells in the Spatraster
    if (Parallel) {
      terraOptions(nthreads = detectCores())
    }
    rast_undesirable_dev <- app(rast_stk, fun = Undesirable_deviation)

    # rescale values according to min max
    undesirable_dev_min <- minmax(rast_undesirable_dev)[1]
    undesirable_dev_max <- minmax(rast_undesirable_dev)[2]

    rast_undesirable_dev_rescaled <- (rast_undesirable_dev -
      undesirable_dev_min) /
      (undesirable_dev_max - undesirable_dev_min)

    # Save the undesirable deviation result
    writeRaster(
      rast_undesirable_dev,
      file.path(
        Spatial_summary_dir,
        "Undesirable_deviation_sum_of_change_all_NCPs.tif"
      ),
      overwrite = TRUE
    )
    writeRaster(
      rast_undesirable_dev_rescaled,
      file.path(
        Spatial_summary_dir,
        "Undesirable_deviation_sum_of_change_all_NCPs_rescaled.tif"
      ),
      overwrite = TRUE
    )
  }

  future::plan(future::sequential)
}

# Main #####

# Load config from $FUTURE_EI_CONFIG_FILE
config_file <- Sys.getenv("RCD_CONFIG_FILE")
if (!file.exists(config_file)) {
  stop(paste0(
    "Config file RCD_CONFIG_FILE (",
    config_file,
    ") does not exist."
  ))
}
config <- yaml.load_file(config_file)
bash_vars <- config$bash_variables # general bash variables
config <- config$Visualisation # only visualisation variables

# if InputDir is not set, use bash variable
# $FUTURE_EI_OUTPUT_DIR/$SUMMARISATION_OUTPUT_BASE_DIR
if (is.null(config$InputDir) || config$InputDir == "") {
  config$InputDir <- file.path(
    bash_vars$FUTURE_EI_OUTPUT_DIR,
    bash_vars$SUMMARISATION_OUTPUT_DIR
  )
}
# if OutputDir is not set, use bash variable CLUSTERING_OUTPUT_DIR
if (is.null(config$OutputDir) || config$OutputDir == "") {
  config$OutputDir <- file.path(
    bash_vars$FUTURE_EI_OUTPUT_DIR,
    bash_vars$VISUALISATION_OUTPUT_DIR
  )
}
# Check if InputDir is now set
if (is.null(config$InputDir) || config$InputDir == "") {
  stop("InputDir nor NCP_OUTPUT_BASE_DIR set in FUTURE_EI_CONFIG_FILE")
}
# Check if InputDir exists
if (!dir.exists(config$InputDir)) {
  stop(cat("InputDir does not exist: ", config$InputDir, "\n"))
}
# Check if OutputDir is set
if (is.null(config$OutputDir) || config$OutputDir == "") {
  stop("OutputDir nor SUMMARISATION_OUTPUT_DIR set in FUTURE_EI_CONFIG_FILE")
}
# Check if OutputDir exists and create if not
if (!dir.exists(config$OutputDir)) {
  dir.create(config$OutputDir, recursive = TRUE)
}

cat("Preparing various visualisations of Future EI outputs \n")

cat("Working directory is:", getwd(), "\n")
cat("Input directory set to:", config$InputDir, "\n")
cat("Output directory set to:", config$OutputDir, "\n")

# create a dir to save tabular outputs in
tabular_output_dir <- file.path(config$OutputDir, "Tabular_outputs")
if (!dir.exists(tabular_output_dir)) {
  dir.create(tabular_output_dir)
}

# Create a dir to save static analysis plots in
Static_plots_dir <- file.path(config$OutputDir, "Static_analysis_plots")
if (!dir.exists(Static_plots_dir)) {
  dir.create(Static_plots_dir)
}

# Create a dir to save dynamic analysis plots in
Dynamic_plots_dir <- file.path(config$OutputDir, "Dynamic_analysis_plots")
if (!dir.exists(Dynamic_plots_dir)) {
  dir.create(Dynamic_plots_dir)
}

# Create a dir for spatial summary results
Spatial_summary_dir <- file.path(config$OutputDir, "Spatial_analysis")
if (!dir.exists(Spatial_summary_dir)) {
  dir.create(Spatial_summary_dir)
}

# The various summary metrics have been saved across seperate files
# read these as in a list

# Load in the Normalized NCP provision summary stats
# (Sum, Mean, Std. for each NCP in each year under all configurations)
NCP_sum_stats <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCP_summary_stats.rds"
))

# Remove Path and Norm_path columns
NCP_sum_stats <- NCP_sum_stats[, -c(1, 2)]

# Also load in the rescaled version of NCP_sum_stats
NCP_sum_stats_rescale <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCP_summary_stats_rescaled.rds"
))

# Remove Path column
NCP_sum_stats_rescale <- NCP_sum_stats_rescale[, -c(1, 2)]

# Load in the Normalized NCP change stats
# (avg. and std. of the positive and negative change in NCP
# provision for each NCP between each time point across all configurations
NCP_change_stats <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCPs_change_summary.rds"
))

# Also load in the rescaled version of NCP_change_stats
NCP_change_stats_rescale <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCP_change_stats_rescaled.rds"
))

# Load in the NCP pattern metric results (SSIM)
NCP_pattern_stats <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCPs_SSIM_summary.rds"
))

# Load in the NCP pattern metric results (SSIM) rescaled
NCP_pattern_stats_rescale <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCP_SSIM_rescaled.rds"
))

# Load the unscaled aggregated metric results
NCP_Agg_stats_unscaled <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCP_aggregated_metric_unscaled.rds"
))

# remove the columns: sum, Avg_pos_change, Avg_neg_change, SSIM to not cause confusion
NCP_Agg_stats_unscaled <- NCP_Agg_stats_unscaled %>%
  select(-sum, -Avg_pos_change, -Avg_neg_change, -SSIM)

# Load the rescaled aggregated metric results
NCP_Agg_stats_rescaled <- readRDS(file.path(
  config$NCP_summarisation_dir,
  "NCP_aggregated_metric_rescaled.rds"
))

# remove the columns: sum, Avg_pos_change, Avg_neg_change, SSIM to not cause confusion
NCP_Agg_stats_rescaled <- NCP_Agg_stats_rescaled %>%
  select(-sum, -Avg_pos_change, -Avg_neg_change, -SSIM)

# For cleaner passing to functions combine objects in a list
Results_data <- list(
  NCP_sum_stats = NCP_sum_stats,
  NCP_sum_stats_rescale = NCP_sum_stats_rescale,
  NCP_change_stats = NCP_change_stats,
  NCP_change_stats_rescale = NCP_change_stats_rescale,
  NCP_pattern_stats = NCP_pattern_stats,
  NCP_pattern_stats_rescale = NCP_pattern_stats_rescale,
  NCP_Agg_stats = NCP_Agg_stats_unscaled,
  NCP_Agg_stats_rescaled = NCP_Agg_stats_rescaled
)

# Vector info column names
info_cols <- c("Config_ID", "NCP", "Scenario", "Time_step")

# if configs$Summary_metrics is empty then get names of unique metrics from
# column names in all entries of Results_data by comparing to info_cols
if (is.null(config$Summary_metrics) || config$Summary_metrics == "") {
  config$Summary_metrics <- unique(unlist(sapply(Results_data, function(x) {
    setdiff(names(x), info_cols)
  })))
}

if (is.null(config$NCPs_to_visualise) || config$NCPs_to_visualise == "") {
  config$NCPs_to_visualise <- unique(Results_data[["NCP_sum_stats"]]$NCP)
}

# create a list of clean names for the metrics
Clean_metric_names <- c(
  "sum" = "Sum",
  "mean" = "Mean",
  "std" = "Std",
  "Avg_pos_change" = "Average Positive Change",
  "Avg_neg_change" = "Average Negative Change",
  "Std_pos_change" = "Standard Deviation of Positive Change",
  "Std_neg_change" = "Standard Deviation of Negative Change",
  "SSIM" = "SSIM",
  "SIM" = "SIM",
  "SIV" = "SIV",
  "SIP" = "SIP",
  "Agg_metric" = "ES impact",
  "Agg_metric_rescaled" = "ES impact"
)

# if config$Scenarios_to_visualise is empty then get names of unique scenarios from
# column names in all entries of Results_data by comparing to info_cols
if (is.null(config$Scenarios_to_visualise)) {
  config$Scenarios_to_visualise <- unique(
    Results_data[["NCP_sum_stats"]]$Scenario
  )
}

# create a list of clean names for the scenarios
Clean_scenario_names <- c(
  "BAU-CC" = "BAU + RCP 8.5",
  "BAU" = "Business as Usual",
  "EI-SOC" = "EI for Society",
  "EI-CUL" = "EI as Culture",
  "EI-NAT" = "EI for Nature"
)

#Define a colour palette for the scenarios
Scenario_pal <- c(
  "BAU-CC" = '#a8aba5',
  "BAU" = "#d1d3cf",
  "EI-SOC" = "#29898f",
  "EI-CUL" = "#f59f78",
  "EI-NAT" = "#6ca147"
)

# Run the function to create the tabular summary
# Tabular_summary(
#   Results_data = Results_data,
#   Summary_metrics = config$Summary_metrics,
#   NCPs_to_visualise = config$NCPs_to_visualise,
#   Scenarios_to_visualise = config$Scenarios_to_visualise,
#   tabular_output_dir = tabular_output_dir,
#   Value_scaling = config$Value_scaling
#   )

### End state/Outcome analysis #####

# run the function to create all the plots for the end state analysis
Static_analysis_plots(
  Value_scaling = config$Value_scaling,
  Summary_metrics = config$Summary_metrics,
  Static_plots_dir = Static_plots_dir,
  Results_data = Results_data,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Clean_metric_names = Clean_metric_names,
  Scenario_pal = Scenario_pal
)

# Create the plot for the publication as a composition with:
# 1. Parallel coordinates plot of the unscaled aggregated metric across all NCPs
# 2. PLots of each of the metrics that make up the aggregated metric:
# 2.1 Plot of the change in sum of provision between 2020 and 2060
# 2.2 Plot of the average positive change in provision across the time points
# 2.3 Plot of the average negative change in provision across the time points
# 2.4 Plot of the SSIM values of 2060 across the time points

# Main plot path
Static_plots <- list(
  Main_plot = readRDS(file.path(
    Static_plots_dir,
    "Metric_Agg_metric_2060_minus_2025_plot_unscaled.rds"
  )),
  Sum_plot = readRDS(file.path(
    Static_plots_dir,
    "Metric_sum_2060_minus_2020_plot_unscaled.rds"
  )),
  SSIM_plot = readRDS(file.path(
    Static_plots_dir,
    "Metric_SSIM_2060_plot_unscaled.rds"
  )),
  Avg_pos_plot = readRDS(file.path(
    Static_plots_dir,
    "Metric_Avg_pos_change_average_over_time_plot_unscaled.rds"
  )),
  Avg_neg_plot = readRDS(file.path(
    Static_plots_dir,
    "Metric_Avg_neg_change_average_over_time_plot_unscaled.rds"
  ))
)


# loop over list of plots and make changes:
# - Set all Y axis limits to 0,1
# - Remove x-axis title
# - Increase legedn key size and remove alpha
# - Set order of the NCPs on the Y-axis
# - remove all plot margins
Static_plots <- lapply(Static_plots, function(x) {
  x <- x +
    scale_y_continuous(limits = c(0, 1)) +
    scale_x_discrete(
      limit = c('CAR', 'FF', 'HAB', 'NDR', 'POL', 'REC', 'SDR', 'WY')
    ) +
    guides(
      colour = guide_legend(override.aes = list(linewidth = 2, alpha = 1))
    ) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_text(face = "plain"),
      legend.key.size = unit(0.5, "cm"),
      legend.title = element_text(face = "bold"),
      legend.position.inside = c(0.9, 0.9),
      legend.position = "inside",
      legend.justification = c("top"),
      legend.box.just = "right",
      legend.direction = "vertical",
      #legend.margin = margin(6, 6, 6, 6),
      plot.margin = margin(0, 0, 0, 0)
    )
  return(x)
})

# Adjust Y-axis titles and remove legends from all plots except the main plot
Static_plots$Main_plot <- Static_plots$Main_plot +
  labs(y = "Normalized change in ES impact \n between 2020 and 2060")
Static_plots$Sum_plot <- Static_plots$Sum_plot +
  labs(
    y = "Normalized change in sum of ES provision \n between 2020 and 2060"
  ) +
  theme(legend.position = "none")
Static_plots$SSIM_plot <- Static_plots$SSIM_plot +
  labs(y = "Normalized SSIM value of 2060 \n with reference to 2020") +
  theme(legend.position = "none")
Static_plots$Avg_pos_plot <- Static_plots$Avg_pos_plot +
  labs(
    y = "Normalized mean of average positive change \n in ES provision between 2020 and 2060"
  ) +
  theme(legend.position = "none")
Static_plots$Avg_neg_plot <- Static_plots$Avg_neg_plot +
  labs(
    y = "Normalized mean of average negative change \n in ES provision between 2020 and 2060"
  ) +
  theme(legend.position = "none")

# combined plots using patchwork

#set up grids for the plots in both portrait and landscape
portrait <- "
  11
  23
  45
"

landscape <- "
  123
  145
"

Static_comp_p <- Static_plots$Main_plot +
  Static_plots$Sum_plot +
  Static_plots$SSIM_plot +
  Static_plots$Avg_pos_plot +
  Static_plots$Avg_neg_plot +
  plot_annotation(tag_levels = 'A', tag_suffix = '.') +
  plot_layout(
    design = portrait,
    heights = c(0.4, 0.3, 0.3)
  ) &
  theme(
    text = element_text(family = "Roboto", size = 10),
    plot.tag = element_text(family = "Roboto", face = "bold")
  )

Static_comp_l <- Static_plots$Main_plot +
  Static_plots$Sum_plot +
  Static_plots$SSIM_plot +
  Static_plots$Avg_pos_plot +
  Static_plots$Avg_neg_plot +
  plot_annotation(tag_levels = 'A', tag_suffix = '.') +
  plot_layout(
    design = landscape,
    widths = c(0.4, 0.3, 0.3)
  ) &
  theme(
    text = element_text(family = "Roboto", size = 10),
    plot.tag = element_text(family = "Roboto", face = "bold")
  )

# Save plot in portrait orientation
ggsave(
  file.path(Static_plots_dir, "Outcome_plot_portrait.png"),
  Static_comp_p,
  dpi = 300,
  width = 21,
  height = 30,
  units = "cm"
)
ggsave(
  file.path(Static_plots_dir, "Outcome_plot_portrait.svg"),
  Static_comp_p,
  dpi = 300,
  width = 21,
  height = 30,
  units = "cm"
)

# Save plot in landscape orientation
ggsave(
  file.path(Static_plots_dir, "Outcome_plot_landscape.png"),
  Static_comp_l,
  dpi = 300,
  width = 30,
  height = 21,
  units = "cm"
)
ggsave(
  file.path(Static_plots_dir, "Outcome_plot_landscape.svg"),
  Static_comp_l,
  dpi = 300,
  width = 30,
  height = 21,
  units = "cm"
)

#save a copy of the portrait plot to the publication figures folder
Outcome_fig_dir <- file.path(Publication_fig_dir, "outcome_analysis")
if (!dir.exists(Outcome_fig_dir)) {
  dir.create(Outcome_fig_dir)
}
file.copy(
  from = file.path(Static_plots_dir, "Outcome_plot_portrait.png"),
  to = file.path(Outcome_fig_dir, "Outcome_plot_portrait.png"),
  overwrite = TRUE
)

### Time-series/trend analysis #####

# run the function to create the time series/ trend analysis plots
Dynamic_analysis_plots(
  Value_scaling = config$Value_scaling,
  Summary_metrics = config$Summary_metrics,
  Dynamic_plots_dir = Dynamic_plots_dir,
  Results_data = Results_data,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Scenario_pal = Scenario_pal,
  Clean_metric_names = Clean_metric_names,
  NCP_aggregation = "Aggregated"
)


### Spatial analysis #####

# Two possibilities for a robustness metric:
# 1. Mean-variance
# 2. Undesirable deviation metric

# MacPhail et al. 2018 provide a good overview of the limitation of mean-variance:
# the mean-variance metric attempts to balance the mean and variability of the
# performance of a decision alternative over different scenarios.
# However, a disadvantage of considering a combination of the mean and variance
# is that the resultant metric is not always monotonically increasing
# (Ray et al., 2013). Moreover, when considering variance, good and bad
# deviations from the mean are treated equally (Takriti & Ahmed, 2004).
#
# The undesirable deviations metric overcomes this limitation of mean-variance,
# while still providing a measure of variability.

# Calculate summarised spatial outputs
Summarise_spatial_outputs(
  change_raster_dir = config$Change_raster_dir,
  NCPs_to_visualise = config$NCPs_to_visualise,
  Aggregate_NCPS = TRUE,
  Spatial_summary_metrics = c("undesirable_deviation"),
  Parallel = TRUE,
  Ref_raster_path = "F:/Future-EI/Future-EI_output/Results/Ref_grid.tif",
  Spatial_summary_dir = Spatial_summary_dir
)

### Create plot from rasters #####
Mean <- rast(
  "visualisations_outputs/Spatial_analysis/Mean_sum_of_change_all_NCPs.tif"
)
Stdev <- rast(
  "visualisations_outputs/Spatial_analysis/Stdev_sum_of_change_all_NCPs.tif"
)
Undes_dev <- rast(
  "visualisations_outputs/Spatial_analysis/Undesirable_deviation_sum_of_change_all_NCPs.tif"
)

# read country borders
country_geo <- read_sf("Tools/Misc_map_layers/g2l15.shp")

# read lakes
lake_geo <- read_sf("Tools/Misc_map_layers/g2s15.shp")
lake_geo <- st_transform(lake_geo, crs = crs(Mean))

# read in raster of relief
relief <- raster("Tools/Misc_map_layers/02-relief-ascii.asc") %>%
  # hide relief outside of Switzerland by masking with country borders
  mask(country_geo) %>%
  projectRaster(Mean) %>%
  as("SpatialPixelsDataFrame") %>%
  as.data.frame() %>%
  rename(value = `X02.relief.ascii`)

# re-project country_geo
country_geo <- st_transform(country_geo, crs = crs(Mean))

# #### Single variable robustness map #####
#
# Robustness_rast <- rast("visualisations_outputs/Spatial_analysis/Mean_variance_sum_of_change_all_NCPs.tif")
#
# # get raster values
# vals <- values(Robustness_rast, na.rm = TRUE)
#
# # set number of breaks
# n_brks <- 7
#
# # calculate natural breaks (Jenks)
# breaks <- classIntervals(vals, n = n_brks, style = "jenks")
#
# # reclassify the raster according to the breaks
# Robust_discrete <- classify(Robustness_rast, breaks$brks, include.lowest=TRUE, brackets=TRUE)
#
# # set up a colour palette from the low, mid and high colours
# pal <- colorRampPalette(c("#873426", "#C6DEE0", "#2D5155"))
#
# # Create a ggplot from the raster
#
# Robustness_map <- ggplot() +
#   geom_spatraster(data = Robust_discrete,
#                   na.rm = TRUE,
#                   maxcell = ncell(Robust_discrete)) +
#   scale_fill_manual(values = pal(n_brks),
#                     na.value = "transparent",
#                     )+
#   guides(fill = guide_legend(reverse = TRUE))+
#   theme_minimal()
# plot(Robustness_map)

#### Map 1: Bivariate robustness map #####

# Stack the mean and undersirable deviation rasters
Mean_var <- c(Mean, Undes_dev)

# name the layers
names(Mean_var) <- c("Mean", "Var")

# convert to df
Mean_var_df <- as.data.frame(Mean_var, xy = TRUE)

# normalise the undesirable deviation to be between 0 and 1
Mean_var_df$Var_norm <- (Mean_var_df$Var - min(Mean_var_df$Var)) /
  (max(Mean_var_df$Var) - min(Mean_var_df$Var))

# vector number of classes for breaks
Map1_num_class <- 4

# classify both the mean and variance measures
Mean_var_class <- bi_class(
  Mean_var_df,
  x = Mean,
  y = Var_norm,
  style = "quantile",
  dim = Map1_num_class
)

# Create breaks
Mean_var_breaks <- bi_class_breaks(
  Mean_var_df,
  x = Mean,
  y = Var_norm,
  style = "quantile",
  dim = Map1_num_class,
  dig_lab = 2,
  split = FALSE
)

# Set up a colour palette
pallet <- "BlueOr"

# Create map
map1 <- ggplot() +
  #   with_shadow(geom_spatvector(data = CH_vect, fill = "white", color = "black", linewidth = 0.20),
  # colour = "black",
  # x_offset = 2,
  # y_offset = 2,
  # sigma = 3,
  # stack = TRUE) +
  geom_raster(
    data = relief,
    inherit.aes = FALSE,
    aes(x = x, y = y, alpha = value)
  ) +
  # use the "alpha hack" (as the "fill" aesthetic is already taken)
  scale_alpha(name = "", range = c(0.6, 0), guide = F) +
  geom_raster(
    data = Mean_var_class,
    aes(x = x, y = y, fill = bi_class),
    show.legend = FALSE
  ) +
  geom_sf(data = lake_geo, fill = "#D6F1FF", color = "transparent") +
  bi_scale_fill(
    pal = pallet,
    dim = Map1_num_class,
    flip_axes = FALSE,
    rotate_pal = FALSE,
    na.value = "white",
    aes(alpha = 0.5)
  ) +
  theme_Publication() +
  theme(
    axis.text = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_blank()
  )

# Create the legend for the bivariate map
legend1 <- bi_legend(
  pal = pallet,
  flip_axes = FALSE,
  rotate_pal = FALSE,
  dim = Map1_num_class,
  breaks = Mean_var_breaks,
  xlab = "Mean sum of change",
  ylab = "Norm. undersiable deviation",
  arrow = FALSE
) +
  theme(
    text = element_text(size = 8, family = "Roboto"),
    axis.text.x = element_text(angle = -25, hjust = 0),
    axis.text.y = element_text(angle = -25, hjust = 0.25),
    #axis.line = element_blank(),
    panel.background = element_rect(fill = 'transparent'),
    plot.background = element_rect(fill = 'transparent', color = NA)
  )

# combine using patchwork
map1_final <- map1 +
  # legend inset on right
  # inset_element(legend1,
  #               left = 0.7,
  #               bottom = 0.7,
  #               right = unit(1, "npc"),
  #               top = unit(1, "npc"),
  #               align_to = "full")
  # legend inset on left
  inset_element(
    legend1,
    left = 0,
    bottom = 0.65,
    right = 0.25,
    top = 1,
    align_to = 'full'
  )


# Save the map
ggsave(
  map1_final,
  filename = file.path(
    Spatial_summary_dir,
    paste0("Mean_var_bivariate_", Map1_num_class, "_classes.png")
  ),
  dpi = 300,
  width = 30,
  height = 21,
  units = "cm"
)

#### Map 2: Robustness x Freq. of inclusion of New EI areas ####
# use ggmagnify with a shapefile of the existing PAs: https://hughjonesd.github.io/ggmagnify/

# Summarise the frequency of cells inclusion as EI areas across the configurations

# Dir containing maps used for EI areas for all configurations
EI_map_dir <- "E:/LULCC_CH_Ensemble/Data/EI_intervention_layers/Future_EI"

# Raster of current (existing) protected areas in Switzerland
Current_PAs <- rast("Tools/Existing_PAs.tif")

# Set all NA values to 0
Current_PAs[is.na(Current_PAs)] <- 0

# list all the rasters in the directory of EI maps
EI_rast_paths <- list.files(
  EI_map_dir,
  pattern = ".tif",
  full.names = TRUE,
  recursive = TRUE
)

# Remove any tif.ovr files
EI_rast_paths <- EI_rast_paths[!grepl(".tif.ovr", EI_rast_paths)]

# Subset to only those that contain the tag '2060' as this is the last
# simulation year and hence will contain all the areas protected over the
# course of the simulation time steps
EI_rast_paths <- EI_rast_paths[grep("2060", EI_rast_paths)]

# The number of EI rast paths represent the number of unique configurations
# which is needed to calculate frequency of inclusion as a %
Num_configs <- length(EI_rast_paths)

# Load all raster layers
EI_rast_stk <- rast(EI_rast_paths)

# sum across all layers in the stack
# removing NAs to not disrupt the calculation
EI_rast_sum <- sum(EI_rast_stk, na.rm = TRUE)

# mask out the current PAs to leave only the new EI areas
EI_rast_sum_new_areas <- mask(EI_rast_sum, Current_PAs, maskvalues = 1)

#save the layer
writeRaster(
  EI_rast_sum_new_areas,
  file.path(Spatial_summary_dir, "EI_inclusion_frequency.tif"),
  overwrite = TRUE
)

# TO DELETE: Load the raster
EI_rast_sum_new_areas <- rast(file.path(
  Spatial_summary_dir,
  "EI_inclusion_frequency.tif"
))

# project the raster to the same CRS as the mean and variance rasters
EI_rast_sum_new_areas <- project(EI_rast_sum_new_areas, Mean)

# StacK the Mean, undersirable deviation and frequency of EI inclusion rasters
Mean_var_EI <- c(Mean, Undes_dev, EI_rast_sum_new_areas)

# name the layers
names(Mean_var_EI) <- c("Mean", "Var", "EI_freq")

# Convert to a data frame
Mean_var_EI_df <- as.data.frame(Mean_var_EI, xy = TRUE)

# subset to complete cases
Mean_var_EI_df <- Mean_var_EI_df[complete.cases(Mean_var_EI_df), ]

# calculate the frequency of EI inclusion as a percentage of the number of configurations
Mean_var_EI_df$EI_perc <- (Mean_var_EI_df$EI_freq / Num_configs) * 100

# Make Var absolute
Mean_var_EI_df$Var <- abs(Mean_var_EI_df$Var)

# normalise the undesirable deviation to be between 0 and 1
Mean_var_EI_df$Var_norm <- (Mean_var_EI_df$Var - min(Mean_var_EI_df$Var)) /
  (max(Mean_var_EI_df$Var) - min(Mean_var_EI_df$Var))

# normalise the mean sum of change to be between 0 and 1
Mean_var_EI_df$Mean_norm <- (Mean_var_EI_df$Mean - min(Mean_var_EI_df$Mean)) /
  (max(Mean_var_EI_df$Mean) - min(Mean_var_EI_df$Mean))

# calculate a ratio of the mean/ undersirable deviation to use instead of mean-variance
Mean_var_EI_df$Perf_var <- (Mean_var_EI_df$Mean_norm + 1) /
  (Mean_var_EI_df$Var + 1)

#Do a box plot of the performance variance
outliers <- boxplot.stats(Mean_var_EI_df$Perf_var)$out

# remove the outliers
Mean_var_EI_df <- Mean_var_EI_df[!Mean_var_EI_df$Perf_var %in% outliers, ]

# Min max re scale the performance variance
Mean_var_EI_df$Perf_var_norm <- (Mean_var_EI_df$Perf_var -
  min(Mean_var_EI_df$Perf_var)) /
  (max(Mean_var_EI_df$Perf_var) - min(Mean_var_EI_df$Perf_var))

Map2_num_class <- 4

# apply manual break points to EI_perc
Mean_var_EI_df$EI_perc_class <- cut(
  Mean_var_EI_df$EI_perc,
  breaks = c(0, 25, 50, 75, max(Mean_var_EI_df$EI_perc))
)

# For each variable determine which classification method is most appropriate
# Logically equal breaks would create a nice axis for the % of inclusion variable
# but actually given that the maximum % is not 100% this is not necessarily true

# plot equal breaks
#plot(classIntervals(Mean_var_EI_df$EI_perc, n = Map2_num_class, style = "equal"))
# plot of equal breaks shows there is a very high frequency of pixels in the
# category of 0.0325-27.9% and much lower frequency in the other categories
# Conclusion equal breaks are not appropriate

# plot quantile breaks
#plot(classIntervals(Mean_var_EI_df$EI_perc, n = Map2_num_class, style = "quantile"))

# plot fisher jenks breaks
#plot(classIntervals(Mean_var_EI_df$EI_perc, n = Map2_num_class, style = "fisher", warnLargeN = TRUE))

# plot quantile breaks for the performance variance
#pal1 <- c("wheat1", "red3")
#plot(classIntervals(Mean_var_EI_df$Perf_var, n = Map2_num_class, style = "quantile"), pal = pal1)

# Classify both variables
Mean_var_EI_class <- bi_class(
  Mean_var_EI_df,
  x = Perf_var_norm,
  y = EI_perc_class,
  style = "quantile",
  dim = Map2_num_class
)

# get the percentage of pixels in each class as a % of the total number of pixels
class_percs <- as.data.frame(
  (table(Mean_var_EI_class$bi_class) / nrow(Mean_var_EI_class)) * 100
)

# add an x column using the first number in Var1
class_percs$x <- as.numeric(str_split(class_percs$Var1, "-", simplify = TRUE)[,
  1
])

# add a y column using the second number in Var1
class_percs$y <- as.numeric(str_split(class_percs$Var1, "-", simplify = TRUE)[,
  2
])

# round freq to 2 decimal places and add a % sign
class_percs$Freq <- paste0(round(class_percs$Freq, 2), "%")

# Create breaks
Mean_var_EI_breaks <- bi_class_breaks(
  Mean_var_EI_df,
  x = Perf_var_norm,
  y = EI_perc_class,
  style = "quantile",
  dim = Map2_num_class,
  dig_lab = 2,
  split = FALSE
)

# Set up a colour palette
pallet <- "DkViolet2"

# Create map
map2 <- ggplot() +
  #   with_shadow(geom_spatvector(data = CH_vect, fill = "white", color = "black", linewidth = 0.20),
  # colour = "black",
  # x_offset = 2,
  # y_offset = 2,
  # sigma = 3,
  # stack = TRUE) +
  geom_raster(
    data = relief,
    inherit.aes = FALSE,
    aes(x = x, y = y, alpha = value)
  ) +
  # use the "alpha hack" (as the "fill" aesthetic is already taken)
  scale_alpha(name = "", range = c(0.6, 0), guide = F) +
  geom_raster(
    data = Mean_var_EI_class,
    aes(x = x, y = y, fill = bi_class),
    show.legend = FALSE
  ) +
  geom_sf(data = lake_geo, fill = "#D6F1FF", color = "transparent") +
  bi_scale_fill(
    pal = pallet,
    dim = Map2_num_class,
    flip_axes = FALSE,
    rotate_pal = FALSE,
    na.value = "white",
    aes(alpha = 0.5)
  ) +
  theme_Publication() +
  theme(
    axis.text = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_blank()
  )

# create a map subsetting values to bi_class == 4-4
High_freq_robust <- ggplot() +
  geom_raster(
    data = relief,
    inherit.aes = FALSE,
    aes(x = x, y = y, alpha = value)
  ) +
  scale_alpha(name = "", range = c(0.6, 0), guide = F) +
  geom_raster(
    data = Mean_var_EI_class[
      Mean_var_EI_class$bi_class %in% c("4-4", "4-3", "3-4"),
    ],
    aes(x = x, y = y, fill = bi_class),
    show.legend = FALSE
  ) +
  geom_sf(data = lake_geo, fill = "#D6F1FF", color = "transparent") +
  bi_scale_fill(
    pal = pallet,
    dim = Map2_num_class,
    flip_axes = FALSE,
    rotate_pal = FALSE,
    na.value = "white",
    aes(alpha = 0.5)
  ) +
  theme_Publication() +
  theme(
    axis.text = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_blank()
  )

# save the map
ggsave(
  High_freq_robust,
  filename = file.path(Spatial_summary_dir, "High_freq_robust_map.png"),
  dpi = 300,
  width = 30,
  height = 21,
  units = "cm"
)

# subset to areas which were selected infrequently but displayed high robustness i.e. 3-1
Missed_opportunities <- ggplot() +
  geom_raster(
    data = relief,
    inherit.aes = FALSE,
    aes(x = x, y = y, alpha = value)
  ) +
  scale_alpha(name = "", range = c(0.6, 0), guide = F) +
  geom_raster(
    data = Mean_var_EI_class[Mean_var_EI_class$bi_class == "4-1", ],
    aes(x = x, y = y, fill = bi_class),
    show.legend = FALSE
  ) +
  geom_sf(data = lake_geo, fill = "#D6F1FF", color = "transparent") +
  bi_scale_fill(
    pal = pallet,
    dim = Map2_num_class,
    flip_axes = FALSE,
    rotate_pal = FALSE,
    na.value = "white",
    aes(alpha = 0.5)
  ) +
  theme_Publication() +
  theme(
    axis.text = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_blank()
  )

# save map
ggsave(
  Missed_opportunities,
  filename = file.path(Spatial_summary_dir, "Missed_opportunities_map.png"),
  dpi = 300,
  width = 30,
  height = 21,
  units = "cm"
)

# Create the legend for the bivariate map
legend2 <- bi_legend(
  pal = pallet,
  flip_axes = FALSE,
  rotate_pal = FALSE,
  dim = Map2_num_class,
  breaks = Mean_var_EI_breaks,
  xlab = "Robustness",
  ylab = "% Freq. of inclusion",
  arrow = FALSE
) +
  geom_text(
    data = class_percs[class_percs$Var1 == "4-1", ],
    aes(x = x, y = y, label = Freq),
    colour = "white",
    fontface = "bold",
    size = 3
  ) +
  theme(
    text = element_text(size = 8, family = "Roboto"),
    axis.text.x = element_text(angle = -25, hjust = 0),
    axis.text.y = element_text(angle = -25, hjust = 0.25),
    #axis.line = element_blank(),
    panel.background = element_rect(fill = 'transparent'),
    plot.background = element_rect(fill = 'transparent', color = NA)
  )

# combine using patchwork
map2_final <- map2 +
  # legend inset on right
  # inset_element(legend2,
  #               left = 0.7,
  #               bottom = 0.7,
  #               right = unit(1, "npc"),
  #               top = unit(1, "npc"),
  #               align_to = "full")
  # legend inset on left
  inset_element(
    legend2,
    left = 0,
    bottom = 0.65,
    right = 0.25,
    top = 1,
    align_to = 'full'
  )

#plot(map2_final)

# Save the map
ggsave(
  map2_final,
  filename = file.path(
    Spatial_summary_dir,
    paste0("Mean_var_EI_bivariate_", Map2_num_class, "_classes.png")
  ),
  dpi = 300,
  width = 30,
  height = 21,
  units = "cm"
)


# Save a copy of both maps to the publication figures dir
#save a copy of the map to the publication figures folder
Spatial_fig_dir <- file.path(Publication_fig_dir, "spatial_analysis")
if (!dir.exists(Spatial_fig_dir)) {
  dir.create(Spatial_fig_dir)
}
file.copy(
  from = file.path(Spatial_summary_dir, "Mean_var_bivariate_4_classes.png"),
  to = file.path(Spatial_fig_dir, "Robustness_map_full.png"),
  overwrite = TRUE
)
file.copy(
  from = file.path(Spatial_summary_dir, "Mean_var_EI_bivariate_4_classes.png"),
  to = file.path(Spatial_fig_dir, "Robustness_map_EI.png"),
  overwrite = TRUE
)

### Supplementary material plots #####

# Clustering summary plot
#1.  Gather all the "_cluster_data.rds" files for the indiv_NCPs for each scenario and bind together
#dummy path: clustering_output/Clustering_results/Agg_metric/best_configs/Max_k4/perf_balance/Indiv_NCPs/Within_scenarios/NCP/SCENARIO/Agg_metric_CONFIG_NCP_SCENARIO_cluster_data.rds

#2. create a parallel coordinates plot of the cluster series for each NCP and scenario with set colours for the clusters and a darker variant of these for the centroids

# label each plot with the cluster determined to be 'good' or colour this differently in the plot
# create a common legend

# arrange as a grid of plots with the NCPs as rows and the scenarios as columns