#Bivariate set-up
labs <- c("1 - 1", "1 - 2", "1 - 3", "2 - 1", "2 - 2", "2 - 3", "3 - 1", "3 - 2", "3 - 3")

#colour scheme from here: https://observablehq.com/@angiehjort/bivariate-color-generator
cols <- c("#d3d3d3", "#88cfcf", "#00c5c5", "#c881bb", "#817eb6", "#0079ae", "#b70092", "#76008f", "#000089")

legend <- tibble(group = labs, fill = cols)

#function to classify continuous variables - used SD approach here, can be replaced
classify3_sd <- function(x){
  cut(x,
      breaks = c(-Inf, -sd(x), sd(x), Inf),
      labels = c("3", "2", "1"))
}

#data needs to be in a data frame
# df <- as.data.frame(raster_stack, xy=TRUE)

df <- df |>
  mutate(
    x_class    = classify3_sd(var_x),
    y_class     = classify3_sd(var_y),
    xy = paste(x_class, y_class, sep = " - "),
  )

df <- left_join(df, legend, by=c("xy"="group"))

#Plot map
p1 <- ggplot()+
  geom_raster( data = df, aes( x=x,  y=y,  fill=fill  ),
    interpolate = TRUE  ) +
  scale_fill_identity() +
  labs( title = "Title") +
  theme_void() +
  theme(legend.position = 'none')

#Plot legend
p2 <- legend_ab %>%
  separate(group, into = c("X", "Y"), sep = " - ") %>%
  mutate(X = as.integer(X),
         Y = as.integer(Y)) %>%
  ggplot() +
  geom_tile(mapping = aes(x = X, y = Y, fill = fill)) +
  scale_fill_identity() +
  scale_x_discrete(breaks = 1:3, labels = c("-ve","Mid", "+ve")) +
  scale_y_discrete(breaks = 1:3, labels = c("-ve","Mid", "+ve")) +
  labs(x = "X",  y = "Y") +
  theme_void() +
  theme(axis.title = element_text(size = 7),
    axis.text = element_text(size=7),
    axis.title.y = element_text(angle = 90)) +
  coord_fixed()

inset_xy <- p1 +
  inset_element(p2,    left   = 0.68, bottom = 0.68,  right  = 0.99,  top    = 0.99,
    align_to = "panel")

inset_xy