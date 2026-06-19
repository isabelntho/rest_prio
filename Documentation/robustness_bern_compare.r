
library(terra)

blce <- rast("Y:/CH_Kanton_Bern/03_Workspaces/05_Web_platform/robustness_output_calc_ignore/robustness_class_2060_ES_SDM.tif")

sens_dir <- "C:/Users/inicholson/Documents/rest_prio/Documentation/notebook_outputs/grid_plots/abc_grid/spatial_tifs/"
#  need to re-run 
sens <- rast(paste0(sens_dir, "rfop_strategy_summary.tif"))
sens <- sens["baseline_strategy_class"]

#align the rasters first
blce <- resample(blce, sens, method = "near")
#set crs for sens
crs(sens) <- crs(blce)
comp <- c(blce, sens)

comp_df <- as.data.frame(comp, xy = TRUE)
comp_df <- comp_df %>%
  filter(baseline_strategy_class == 3)

#separate the column label by the "+" sign
comp_df <- comp_df %>%
  separate(label, into = c("label1", "label2"), sep = "\\+")

comp_df$label1 <- gsub("\\d: ", "", comp_df$label1)
comp_df <- comp_df %>% drop_na()
comp_df$label1 <- recode(comp_df$label1, "hohe Stab. " = "High", "mittlere Stab. " = "Medium",
                                               "tiefe Stab. " = "Low")
comp_df$label1 <- factor(comp_df$label1, levels = c("Low", "Medium", "High"))

comp_df$label2 <- recode(comp_df$label2, " hohe Leist." = "High", " mittlere Leist." = "Medium",
                                               " tiefe Leist." = "Low")
comp_df$label2 <- factor(comp_df$label2, levels = c("Low", "Medium", "High"))

# plot areas where sens = robust, coloured by blce category
ggplot() +
  geom_raster(data = comp_df, aes(x = x, y = y, fill = label)) +
  theme_minimal() +
  labs(title = "ES/BD uncertainty to 2060")

# also plot as bar chart
comp_df %>%
  group_by(label1, label2) %>%
  summarise(count = n(), .groups = "drop") %>%
  ggplot(aes(x = label1, y = count)) +
  geom_bar(stat = "identity") +
  facet_wrap(~label2) +
  labs(title = "Count of robust pixels by BLCE category", x= "Stability", y="Count")+
  theme_minimal()
