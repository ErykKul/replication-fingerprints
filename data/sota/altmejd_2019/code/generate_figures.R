# Generates all figures for main paper and SI
# saves to ../paper/figures and ../paper/tables
setwd("~/Dropbox/Research/Reproducibility/Predicting Replication/git/code/")
library(data.table)
library(tidyverse)
library(dtplyr)
library(ggplot2)
library(cowplot)
library(ggbiplot)
library(RColorBrewer)
library(pander)
library(xtable)
library(reshape)
library(caret)
library(randomForest)
library(margins)
library(pROC)
library(stats)
library(plotROC)
library(extrafont)
loadfonts(device="postscript", quiet = TRUE)

source("functions.R")

# save figures and tables here:
fig_dir <- "../figures"
dir.create(file.path(fig_dir, "bw", "si"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(fig_dir, "color", "si"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(fig_dir, "presentation"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(fig_dir, "tables", "si"), showWarnings = FALSE, recursive = TRUE)

# produce plots in color AND grayyscale
col_suffix <- c("bw", "color")
fig_dpi <- 300
fig_filetype <- ".pdf"
inch <- 0.393701 # cowplot/ggplot prefers inches
fig_width <- 17.8*inch # 17.8 cm = 7.01 in
fig_height_pres <- 4
fig_aspect_ratio_pres <- 16/9
font_fam <- "Arial"

# While working with plots, make sure size of interactive device is same as pdf
quartz.options(width = fig_width, height = fig_width, dpi = fig_dpi)

# Cowplot default theme settings
theme_set(theme_cowplot(font_size = 12))
# theme_replace(font_family = "Arial")
theme_replace(
    #text = element_text(family = font_fam), # error?
    axis.text = element_text(family = font_fam),
    axis.text.x = element_text(family = font_fam),
    axis.text.y = element_text(family = font_fam),
    axis.title = element_text(family = font_fam),
    axis.title.x = element_text(family = font_fam),
    axis.title.y = element_text(family = font_fam),
    legend.text = element_text(family = font_fam),
    legend.title = element_text(family = font_fam),
    strip.text = element_text(family = font_fam),
    strip.text.y = element_text(family = font_fam),
    plot.caption = element_text(family = font_fam),
    plot.tag = element_text(family = font_fam),
    plot.title = element_text(family = font_fam),
    plot.subtitle = element_text(family = font_fam)
)

# ==============================================================================
# MAIN PAPER
# ==============================================================================

# Load data (same prcedure as in nested_cv_prepare_data.R)
rawdata <- readRDS("../data/data.rds")
varnames <- as.data.table(read_tsv("../data/variables.tsv"))

# We only use the aggregated ML data
rawdata <- rawdata[aggregated == TRUE | is.na(aggregated)]
data <- rawdata[drop == FALSE] # droplist
drop_cols <- c("title", "authors.o", "journal", "volume", "issue",
               "aggregated", "lab_id", "drop", "experiment_country.o",
               "experiment_country.r", "experiment_language.o",
               "experiment_language.r", "online.o", "online.r", "subjects.o",
               "subjects.r")
data[, (drop_cols) := NULL]

data$seniority.r <- as.factor(data$seniority.r)
data$seniority.o <- as.factor(data$seniority.o)
data$effect_type <- as.factor(data$effect_type)
data$compensation.o <- as.factor(data$compensation.o)
data$compensation.r <- as.factor(data$compensation.r)
data$replicated <- factor(data$replicated)
levels(data$replicated) <- c("Not Replicated", "Replicated")
levels(data$project) <- c("EE", "ML1", "ML3", "RPP")

# ==============================================================================
# FIGURE 1
# 2 descriptive plots (effect size and correlation heatmap)

for (i in seq_along(col_suffix)) {
    bw <- c(TRUE, FALSE)[i]

    # Plot1: Effect size
    effect_size.plot <- ggplot(data,
                            aes(x = effect_size.o, y = effect_size.r, color = project, alpha = 1)) + # 0.5
                            geom_abline(intercept = 0, slope = 0, linetype="dotted", alpha = 1) + # 0.5
                            geom_abline(intercept = 0, slope = 1, linetype="solid", alpha = 1) + # 0.8
                            geom_point(aes(shape = replicated), size = 4, alpha = 1) + # 0.8
                            scale_y_continuous(limits = c(-0.5, 1)) +
                            scale_x_continuous(limits = c(0, 1)) +
                            pr_theme_paper() +
                            scale_colour_brewer(palette=ifelse(bw, "Greys", "Set1")) +
                            xlab("Original Effect Size") +
                            ylab("Replication Effect Size") +
                            guides(color = guide_legend(title = "Project"),
                                   shape = guide_legend(title = NULL))
    plot1 <- effect_size.plot + guides(color = guide_legend(title = NULL)) +
        theme(legend.position = "top",
                axis.title = element_text(size = 12, face = "bold"),
                legend.text = element_text(size = rel(0.7)),
                legend.key.size = unit(1.5, "pt"),
                legend.spacing = unit(0, "pt"),
                legend.margin = ggplot2::margin(1, 0, -0.5, 0, unit = "lines"),
                plot.margin = ggplot2::margin(-0.5, 0, 0.5, 0.5, unit = "lines"))

    # Plot2: Correlation heatmap
    # Exclude categorical variables
    df <- remove_cols(data,
                    c("id", "pub_year", "endprice", "transactions",
                        "trading_volume", "discipline", "effect_type",
                        "compensation.o", "compensation.r", "project",
                        "seniority.o", "seniority.r",
                        "author_citations_avg.o", "author_citations_avg.r"),
                    copy = TRUE)
    df[, replicated := as.numeric(replicated)]
    df <- df[complete.cases(df),]
    df <- df[, order(names(df)), with = FALSE]
    # Melt a table with a row for each correlation coefficient
    # Use rank-order for all variables (also for binary vars)
    df <- as.data.table(melt(cor(df, method="spearman")))
    colnames(df) <- c("Var1", "Var2", "value") # fix bug with variable naming
    df[Var2 %in% c("replicated", "relative_es"), dep_var2 := 1L]
    setorder(df, dep_var2, Var2, Var1, na.last = TRUE) # set order to get dep vars first

    df[varnames, Var1_n := i.v_n, on = .(Var1 = v)]
    df[varnames, Var2_n := i.v_n, on = .(Var2 = v)]
    df[Var1_n == "Sample Size (O)", Var1_n := "N Obs. (O)"]
    df[Var2_n == "Sample Size (O)", Var2_n := "N Obs. (O)"]
    df[, Var1_n := factor(Var1_n)]
    df[, Var2_n := factor(Var2_n)]

    df[value == 1, value := NA] # make diagonal missing
    if (bw) df[, label := ifelse(value > 0, "+", "-")] # add +/- signs for greyscale
    cols <- if(bw) {
        c(rev(brewer.pal(5, "Greys")[2:5]), "#F7F7F7", brewer.pal(5, "Greys")[2:5])
    } else brewer.pal(9, "RdBu") # Color palette choice
    corr.plot <- ggplot(data = df, aes(x = Var1_n, y = Var2_n)) +
        geom_tile(aes(fill = value)) +
        scale_fill_gradient2(low = cols[1], mid = cols[5],
                            high = cols[9], guide = "colourbar",
                            breaks = c(-1, 0, 1), labels = c(-1, 0, 1),
                            limits = c(-1, 1),
                            na.value = ifelse(bw, "black" , "grey40")) +
        guides(fill = guide_colorbar(title = "Spearman r",
                                    title.vjust = 0.8,
                                    label.position = "bottom",
                                    direction = "horizontal",
                                    ticks = FALSE)) +
        pr_theme_paper() %+replace%
        theme(legend.position = "top",
              legend.margin = ggplot2::margin(0.5, 0, -0.5, 0, unit = "lines"),
              legend.title = element_text(size = 12, face = "bold"),
              axis.title = element_blank(),
              axis.text = element_text(size = rel(0.90)),
              axis.text.x = element_text(angle = 40, vjust = 1, hjust = 1))

    if(bw) corr.plot <- corr.plot + geom_text(aes(label = label), color = "grey70",
                                              size = 3.5) # add +/- when bw

    # Save full version to appendix
    ggsave(paste0(fig_dir, "/", col_suffix[i], "/si/", "S1_Fig" , fig_filetype), plot = corr.plot,
       dpi = fig_dpi, width = fig_width,) # corr_plot

    # Zoomed in version
    drop_ycols <- c("p_value.r", "author_citations_max.o", "author_citations_max.r",
                    "authors_male.o", "authors_male.r", "n_authors.o",
                    "power_planned.r", "n_authors.r", "n_planned.r", "n.r",
                    "power.r", "us_lab.o", "us_lab.r", "effect_size.r", "length",
                    "same_country", "same_subjects", "same_online", "same_language")
    df <- df[!(df$Var2 %in% drop_ycols),]
    drop_xcols <- c("n_planned.r", "power_planned.r")
    df <- df[!(df$Var1 %in% drop_xcols),]
    corr_zoomed.plot <- ggplot(data = df,
                        aes(x = Var1_n, y = Var2_n)) + geom_tile(aes(fill = value)) +
                        scale_fill_gradient2(low = cols[1], mid = cols[5],
                                            high = cols[9], guide = "colourbar",
                                            breaks = c(-1, 0, 1), labels = c(-1, 0, 1),
                                            limits = c(-1, 1),
                                            na.value = ifelse(bw, "black" , "grey40")) +
                        guides(fill = guide_colorbar(title = "Spearman r",
                                                    title.vjust = 0.8,
                                                    label.position = "bottom",
                                                    direction = "horizontal",
                                                    ticks = FALSE)) +
                        pr_theme_paper() %+replace%
                        theme(legend.position = "top",
                            legend.margin = ggplot2::margin(0.5, 0, -0.5, 0, unit = "lines"),
                            legend.title = element_text(size = 12, face = "bold"),
                            axis.title = element_blank(),
                            axis.text = element_text(size = rel(0.90)),
                            axis.text.x = element_text(angle = 40, vjust = 1, hjust = 1))

    # Add plus/minus signs for greyscale plot
    if(bw) corr_zoomed.plot <- corr_zoomed.plot +
        geom_text(aes(label = label), color = "grey70", size = 4.5)

    plot2 <- corr_zoomed.plot + theme(
        legend.margin = ggplot2::margin(1, 0, -0.5, 0, unit = "lines"),
        plot.margin = ggplot2::margin(-0.5, 0.5, 0, 0.5, unit = "lines"))
    pgrid <- plot_grid(plot1, plot2, ncol = 2, rel_widths = c(0.75, 1),
                       labels = c("", ""))

    save_plot(paste0(fig_dir, "/", col_suffix[i], "/Fig1", fig_filetype), pgrid, # es_corr_plot
              ncol = 2, # we're saving a grid plot of 2 columns
              nrow = 1, # and 2 rows
              dpi = fig_dpi, base_width = fig_width,
              base_height = fig_width * 0.7
              # each individual subplot should have an aspect ratio of 1.3
              )

    # Presentation figures only in color
    if (col_suffix[i] == "color") {
        save_plot(paste0(fig_dir, "/", "presentation/", "es_plot", fig_filetype), plot1,
                  dpi = fig_dpi, base_height = fig_height_pres,
                  base_aspect_ratio = fig_aspect_ratio_pres)
        save_plot(paste0(fig_dir, "/", "presentation/", "corr_plot", fig_filetype), plot2,
                  dpi = fig_dpi, base_height = fig_height_pres,
                  base_aspect_ratio = fig_aspect_ratio_pres)
    }

}

# ==============================================================================
# FIGURE 2
# Random Forest performance - comparing specifications in Class and Reg
load("../data/nested_cv_class_results.RData")
load("../data/nested_cv_reg_results.RData")

class_rf_specs <- class_results[c("LM (OLS or Logit)", "Random Forest",
                            "Only Basic Features",  "No Discipline",
                            "No Replication Features")]
class_rf_df <- list()
class_rf_df$values <- merge_results(class_rf_specs)

reg_rf_specs <- reg_results[c("LM (OLS or Logit)", "Random Forest",
                          "Only Basic Features", "No Discipline",
                          "No Replication Features")]
reg_rf_df <- list()
reg_rf_df$values <- merge_results(reg_rf_specs)


for (i in seq_along(col_suffix)) {
    bw <- c(TRUE, FALSE)[i]

    class_rf_comp_dotplot <-
        PR_dotplot(class_rf_df,
                model.order = c("Random Forest", "No Discipline",
                                "No Replication Features", "Only Basic Features",
                                "LM (OLS or Logit)"),
                metric.order = c("ROC", "Acc"),
                metric.names = c("Area under ROC", "Accuracy"),
                # metric.order = c("ROC", "Acc", "TPR", "TNR"),
                # metric.names = c("Area under ROC", "Accuracy",
                #                  "True Positive Rate", "True Negative Rate"),
                type = "iqr")


    reg_rf_comp_dotplot <-
        PR_dotplot(reg_rf_df,
                model.order = c("Random Forest", "No Discipline",
                                "No Replication Features", "Only Basic Features",
                                "LM (OLS or Logit)"),
                metric.order = c("Rsquared", "Corr"),
                metric.names = c("R-squared", "Correlation (r)"),
                # metric.order = c("Rsquared", "RMSE", "Corr"),
                # metric.names = c("R-squared", "Root MSE", "Correlation (r)"),
                type = "iqr")

    plot1 <- class_rf_comp_dotplot +
        theme(legend.position = "none",
              plot.margin = ggplot2::margin(1.5, 0.5, 0.5, 0, unit = "lines")) +
        ylim(0,1)

    plot2 <- reg_rf_comp_dotplot +
        theme(legend.position = c(0.98, 0.97),
              legend.justification = c("right", "top"),
              legend.box.just = "right",
              legend.background = element_rect(fill = "grey90"),
              legend.margin = ggplot2::margin(-1.8, 1.8, 1.8, 1.8),
              legend.key.size = unit(1, "lines"),
              legend.text = element_text(size = 10),
              legend.title = element_blank(),
              #legend.title = element_text(size = 9, face = "bold"),
              plot.margin = ggplot2::margin(1.5, 0.5, 0.5, 0, unit = "lines")) +
        ylim(0,1)

    pgrid <- plot_grid(plot1, plot2, ncol = 2, nrow = 1,
                    rel_widths = c(1, 1),
                    vjust = 1.5, hjust = -1.3,
                    labels = c("y = Replicated (Yes/No)", "y = Relative Effect Size"),
                    label_size = 14,
                    #label_fontfamily = "serif",
                    label_fontface = "plain"
                    )
    # add legend outside of plots with an outer plot grid
    # pgrid <- plot_grid(pgrid, get_legend(class_rf_comp_dotplot +
    #     theme(legend.margin = ggplot2::margin(0, unit = "lines"),
    #           legend.position = c(1,1), #"bottom",
    #           legend.justification = c("right", "bottom"),
    #           legend.direction = "vertical")),
    #     ncol = 1, rel_heights = c(1, .07))

    save_plot(paste0(fig_dir, "/", col_suffix[i], "/Fig3", fig_filetype), pgrid, # rf_comp_joined
            ncol = 2,
            nrow = 1,
            dpi = fig_dpi,
            base_width = fig_width,
            base_height = fig_width * 0.7
            )

    # Add legend to plot1 for presentation version
    plot1 <- class_rf_comp_dotplot +
        theme(legend.position = c(0.02, 0.98),
            legend.justification = c("left", "top"),
            legend.box.just = "left",
            legend.background = element_rect(fill = "grey90"),
            legend.margin = ggplot2::margin(-1.7, 1.8, 1.8, 1.8),
            legend.key.size = unit(1, "lines"),
            legend.text = element_text(size = 10),
            legend.title = element_blank(),
            #legend.title = element_text(size = 9, face = "bold"),
            plot.margin = ggplot2::margin(1, 0.3, 0.3, 0, unit = "lines")) +
        ylim(0,1)

    # Presentation figures only in color
    if (col_suffix[i] == "color") {
        save_plot(paste0(fig_dir, "/", "presentation/", "rf_comp_class", fig_filetype), plot1,
                  ncol = 1, nrow = 1,
                  dpi = fig_dpi, base_height = fig_height_pres,
                  base_aspect_ratio = fig_aspect_ratio_pres)

        save_plot(paste0(fig_dir, "/", "presentation/", "rf_comp_reg", fig_filetype), plot2,
                  ncol = 1, nrow = 1,
                  dpi = fig_dpi, base_height = fig_height_pres,
                  base_aspect_ratio = fig_aspect_ratio_pres)
    }
}

# ==============================================================================
# FIGURE 4
# Variable Importance
load("../data/full_models.RData")

vi_df <- gen_varimp_df(rf_models = list(fit_class.full, fit_reg.full),
                       lm_models = list(fit_class.lasso_lm, fit_reg.lasso_lm),
                       ame = list(ame_class, ame_reg))

# name shortening
vi_df <- as.data.table(vi_df)
vi_df[varnames, Feature_n := i.v_n, on = .(Feature = v)]
# vi_df$Feature <- gsub("highest_", "", vi_df$Feature)

# Nicer legend names
vi_df[type == "replicated", type := "Replicated"]
vi_df[type == "relative_es", type := "Relative Effect Size"]

for (i in seq_along(col_suffix)) {
    bw <- c(TRUE, FALSE)[i]

    y_min <- (0.3 * 100) * (-1) # lasso_share of plot space = 0.3
    vi_df$lab_pos <- y_min # vary label position by type
    vi_df[vi_df$type == "Relative Effect Size", ]$lab_pos <- y_min / 2

    g <- ggplot(aes(reorder(Feature_n, Importance,
                            FUN = function(x) { x[1] }), y = Importance,
                    fill = factor(type)), data = vi_df) +
        geom_bar(stat = "identity", width = 0.90,
                 position = position_dodge(width = 0.90)) +
        geom_label(aes(x = Feature_n, y = lab_pos, label = label, fill = type),
                   color = "black",
                   label.padding = unit(0.15, "lines"),
                   label.size = NA,
                   na.rm = TRUE, show.legend = FALSE,
                   size = 2.3, hjust = "left") +
        geom_hline(yintercept = 0, size = 0.2) +
        coord_flip() +
        guides(fill=guide_legend(title="Outcome variable",
                                    reverse = TRUE)) +
        scale_y_continuous(limits = c(y_min - 1,102), expand = c(0, 0),
                           breaks = c(0, 25, 50, 75, 100),
                           labels = c(0, 25, 50, 75, 100)) +
        pr_theme_paper() %+replace%
        theme(legend.position = "top", axis.title = element_blank(),
              axis.text.y = element_text(size = 8, hjust = 0.97, vjust = 0.25),
              legend.margin = ggplot2::margin(0.5, 0, -0.5, -5, unit = "lines"),
              legend.key.size = unit(1, "lines"),
              legend.text = element_text(size = 9),
              legend.title = element_text(size = 9, face = "bold")
              )

    g <- g + if(bw) {
        scale_fill_grey(start = 0.4, end = 0.7, na.value = "red")
    } else scale_fill_brewer(palette = "Set1")

    save_plot(paste0(fig_dir, "/", col_suffix[i], "/Fig4", fig_filetype), plot = g, #varimp_plot
              ncol = 1, nrow = 1,
              dpi = fig_dpi,
              scale = 0.8,
              base_height = fig_width * 1.4, base_width = fig_width)

    # PRESENTATION
    # Plot a horizontal version instead
    if (col_suffix[i] == "color") {
        y_min <- (0.3 * 100) * (-1) # lasso_share = 0.6
        vi_df$lab_pos <- y_min # vary label position by type
        vi_df[vi_df$type == "Relative Effect Size", ]$lab_pos <- y_min / 2
        g <- ggplot(data = vi_df,
                    aes(x = reorder(Feature_n, Importance, FUN = function(x) { -x[1] }),
                        y = Importance,
                        fill = factor(type, levels = c("Replicated", "Relative Effect Size")))) +
            geom_bar(stat = "identity", width = 0.90, position = position_dodge(width = 0.90)) +
            geom_text(aes(x = Feature_n, y = lab_pos, label = label, color = type),
                      na.rm = TRUE, show.legend = FALSE,
                      size = 1.6, angle = 45, hjust = 0.1, vjust = 0 , nudge_x = -0.2,
                      nudge_y = -0.05) +
            scale_colour_brewer(palette = "Set1") +
            scale_fill_brewer(palette = "Set1") +
            geom_hline(yintercept = 0, size = 0.2) +
            guides(fill=guide_legend(title="Outcome variable", reverse = FALSE)) +
            scale_y_continuous(limits = c(y_min - 2,102), expand = c(0, 0),
                               breaks = c(0, 25, 50, 75, 100),
                               labels = c(0, 25, 50, 75, 100)) +
            pr_theme_paper() %+replace%
            theme(legend.position = "top", axis.title = element_blank(),
                  axis.text.y = element_text(size = 6, hjust = 0.97, vjust = 0.25),
                  legend.margin = ggplot2::margin(0.5, 0, -0.5, -5, unit = "lines"),
                  legend.key.size = unit(1, "lines"),
                  legend.text = element_text(size = 9),
                  legend.title = element_text(size = 9, face = "bold"),
                  axis.text.x = element_text(size = 6, angle = 45, hjust = 1, vjust = 1),
                  plot.margin = ggplot2::margin(0.1, 0.5, 0.1, 1.5, unit = "lines")
                  ) +
            annotate("label", x = 31.84, y = 93, label = "Random Forest", size = 2) +
            annotate("label", x = 33.15, y = -8.5, label = "Lasso", size = 2)

        save_plot(paste0(fig_dir, "/", "presentation/", "varimp_plot", fig_filetype), plot = g,
                ncol = 1, nrow = 1, scale = 0.8,
                dpi = fig_dpi, base_height = fig_height_pres,
                base_aspect_ratio = fig_aspect_ratio_pres)
    }
}

# ==============================================================================
# SUPPLEMENTARY MATERIAL (APPENDIX)
# ==============================================================================

# ==============================================================================
# Table S1: Variable Information
df <- as.data.table(read_tsv("../data/variables.tsv"))
df <- df[!(v %in% c("title", "authors.o", "pub_year", "journal",
                         "volume", "issue" , "drop", "aggregated")), ]

print(xtable(df[,.(Variable = v_n, Description = d)],
             #caption = "Description of all variables in the data set. (O) means original study and (R) replication. Reg stands for the regression model with a continuous outcome measure, and Class for the classification model with binary outcome.",
             caption = "Variable Descriptions",
             label = "tbl:vars",
             align = c("l", "|l|", "p{7.5cm}|")),
      booktabs = TRUE, include.rownames=FALSE, size="\\tiny",
      caption.placement = "top",
      tabular.environment="longtable",
      floating = FALSE,
      #sanitize.text.function=identity, # don't sanitize
      file = paste0(fig_dir, "/", "tables/si/", "S1_Table.tex"))

# ==============================================================================
# Table S2: Summary Statistics

df <- copy(rawdata[drop == FALSE,])
df[, c("id", "title", "authors.o", "journal", "volume", "issue", "aggregated",
       "lab_id", "drop") := NULL]
binary <- names(df[, sapply(df, function(x) length(unique(x))) == 2, with = FALSE])
categorical <- names(df[, sapply(df, class) %in% c("character", "factor"), with = FALSE])

summary_table <- sapply(df[, -c(binary, categorical), with = FALSE], summary)
summary_table <- lapply(summary_table, function(x) {
        if(length(x) == 6) {
            x <- c(x, 0)
            names(x)[length(x)] <- "NA's"
            return(x)
        }
        return(x)
    })
summary_table <- do.call(rbind, summary_table)
st <- as.data.frame(round(summary_table, 3))
st <- st[order(row.names(st)),]
st$Variable <- paste0("\\texttt{", sanitize(rownames(st)), "}")
rownames(st) <- NULL
print(xtable(st[,c(8,1,3,4,6,7)], #digits = 3, auto = TRUE,
             caption = "Summary Statistics (continuous variables)",
             label = "tbl:summary_stats_cont"),
      booktabs = TRUE, include.rownames=FALSE, # size="\\tiny",
      caption.placement = "top",
      sanitize.text.function=identity, # don't sanitize
      file = paste0(fig_dir, "/", "tables/si/", "S2_Table_1.tex"))

# binary and categorical need separate tables with counts of each occurance
df[, binary, with = FALSE] %>%
    lapply(., as.factor) %>%
    lapply(., summary) %>%
    do.call(rbind, .) %>%
    data.table(Variable = paste0("\\texttt{", sanitize(rownames(.)), "}"), .) %>%
    xtable(., caption = "Summary Statistics (binary variables)",
           label = "tbl:summary_stats_bin") %>%
    print(., booktabs = TRUE, include.rownames=FALSE, # size="\\tiny",
          caption.placement = "top",
          sanitize.text.function=identity,
          file = paste0(fig_dir, "/", "tables/si/", "S2_Table_2.tex"))


# Create a frequency table for each Original/Replication pair
var_groups <- unique(gsub("\\.[or]$", "", categorical))
for (i in seq_along(var_groups)) {
    var <- var_groups[i]
    v <- categorical[categorical %in% paste0(var, c(".o", ".r"))]
    if (identical(v, character(0))) v <- var
    counts <- df[, v, with = FALSE] %>%
        lapply(., as.factor) %>%
        lapply(., summary)

    dt <- data.table(Freq=unique(unlist(lapply(counts, names))))
    setkey(dt, Freq)
    for (j in 1:length(v)) {
        c <- counts[[j]]
        if (length(v) > 1) {
            col <- paste0("(", gsub("^.*\\.", "", names(counts))[j], ")")
        } else col <- ""
        dt[Freq %in% names(c), (col) := c]
    }

    # Replace NA with 0
    for (j in seq_len(ncol(dt))) set(dt, which(is.na(dt[[j]])), j, 0)

    dt[, Freq := paste0("\\texttt{", sanitize(Freq), "}")]
    dt %>%
    xtable(., caption = paste0("Summary Statistics: (\\texttt{", sanitize(var), "})"),
           label = paste0("tbl:summary_stats_cat_", i)) %>%
    print(., booktabs = TRUE, include.rownames=FALSE, # size="\\tiny",
          caption.placement = "top",
          sanitize.text.function=identity,
          file = paste0(fig_dir, "/", "tables/si/", "S2_Table_3_", i, ".tex"))
}


# ==============================================================================
# Table S3: Prediction Market Accuracy

# rawdata includes droplist (to get same metrics as in previous papers)
df <- na.omit(rawdata[,.(endprice, replicated)])
pm_acc_pooled <- sum((df$endprice > 0.5 & df$replicated == 1) |
                     (df$endprice < 0.5 & df$replicated == 0))/nrow(df)
df <- na.omit(rawdata[project == "rpp",.(endprice, replicated)])
pm_acc_rpp <- sum((df$endprice > 0.5 & df$replicated == 1) |
                  (df$endprice < 0.5 & df$replicated == 0))/nrow(df)
df <- na.omit(rawdata[project == "ee",.(endprice, replicated)])
pm_acc_ee <- sum((df$endprice > 0.5 & df$replicated == 1) |
                 (df$endprice < 0.5 & df$replicated == 0))/nrow(df)
acc1 <- rbind(pm_acc_pooled, pm_acc_rpp, pm_acc_ee)

# Excluding droplist (normal data set)
df <- na.omit(rawdata[drop == FALSE, .(endprice, replicated)])
pm_acc_pooled <- sum((df$endprice > 0.5 & df$replicated == 1) |
                     (df$endprice < 0.5 & df$replicated == 0))/nrow(df)
df <- na.omit(rawdata[drop == FALSE & project == "rpp", .(endprice, replicated)])
pm_acc_rpp <- sum((df$endprice > 0.5 & df$replicated == 1) |
                  (df$endprice < 0.5 & df$replicated == 0))/nrow(df)
df <- na.omit(rawdata[drop == FALSE & project == "ee", .(endprice, replicated)])
pm_acc_ee <- sum((df$endprice > 0.5 & df$replicated == 1) |
                 (df$endprice < 0.5 & df$replicated == 0))/nrow(df)
acc2 <- rbind(pm_acc_pooled, pm_acc_rpp, pm_acc_ee)

acc <- cbind(acc1, acc2)
rownames(acc) <- c("Pooled PM Accuracy:",
                   "RPP PM Accuracy:",
                   "EE PM Accuracy:")
colnames(acc) <- c("Full dataset", "ML dataset")
tmp <- paste0("$", as.character(round(acc, 3)*100), "\\%$")
attributes(tmp) <- attributes(acc)
print(xtable(tmp,
             caption = "Prediction Market accuracy (at 50\\% probability cutoff)",
             label = "tbl:pm_acc"),
      booktabs = TRUE, file = paste0(fig_dir, "/", "tables/si/", "S3_Table.tex"),
      caption.placement = "top",
      sanitize.text.function=identity)

# ==============================================================================
# FIGURE S4: Full correlation plot (saved above)

# ==============================================================================
# FIGURE S1: Principal Components Analysis
# Should not use PCA on ordinal/categorical/binary data or outcome variables

pca_data <- data[, .(length, citations, effect_size.o, p_value.o, n.o,
                     power.o, n_authors.o, n_planned.r,
                     n_authors.r, author_citations_avg.o,
                     author_citations_max.o, authors_male.o,
                     author_citations_avg.r, author_citations_max.r,
                     authors_male.r, project)]
pca_data <- pca_data[complete.cases(pca_data)]
pr_pca <- prcomp(pca_data[, !"project", with = FALSE],
                 center = TRUE, scale. = TRUE)

pca.plot <- ggbiplot(pr_pca, groups = pca_data$project,
                    obs.scale = 1, var.scale = 1,
                    ellipse = TRUE, circle = TRUE) +
    pr_theme_paper() +
    theme(axis.title = element_text(size = 10, face = "bold")) +
    xlim(-3, 8) + ylim(-5.5, 4)

for (i in seq_along(col_suffix)) {
    bw <- c(TRUE, FALSE)[i]

    pca.plot.tmp <- pca.plot + scale_colour_brewer(palette=ifelse(bw, "Greys", "Set1"), name = "")

    save_plot(paste0(fig_dir, "/", col_suffix[i], "/si/S2_Fig", fig_filetype), plot = pca.plot.tmp, # pca
              dpi = fig_dpi, base_width = fig_width, base_height = fig_width * 0.8)
}

tmp <- t(summary(pr_pca)$importance)
colnames(tmp) <- c("s.d.", "Var. \\%", "Cumulative")
print(xtable(tmp, digits = 3, auto = TRUE,
             caption = "PCA", label = "tbl:pca"),
      booktabs = TRUE, file = paste0(fig_dir, "/", "tables/si/", "S4_Table.tex")) # tbl_pca

# ==============================================================================
# Figure S3
# Classification and Regression model comparisons

for (i in seq_along(col_suffix)) {
    bw <- c(TRUE, FALSE)[i]

    algos <- class_results[1:6]
    class_algo_df <- list()
    class_algo_df$values <- merge_results(algos)

    class_models_dotplot <-
        PR_dotplot(class_algo_df,
                model.order = c("Random Forest", "GBM", "SVM", "LASSO",
                                "Ref. Discipline Dummies", "LM (OLS or Logit)"),
                metric.order = c("ROC", "Acc"),
                metric.names = c("Area under ROC", "Accuracy"),
                # metric.order = c("ROC", "Acc", "TPR", "TNR"),
                # metric.names = c("Area under ROC", "Accuracy",
                #                  "True Positive Rate", "True Negative Rate"),
                type = "iqr")

    algos <- reg_results[c(1,3,4,5,6,7)] # skip "Ref. Original Effect Size"
    reg_algo_df <- list()
    reg_algo_df$values <- merge_results(algos)

    reg_models_dotplot <-
        PR_dotplot(reg_algo_df,
                model.order = c("Random Forest", "GBM", "SVM", "LASSO",
                                "Ref. Discipline Dummies", "LM (OLS or Logit)"),
                metric.order = c("Rsquared", "Corr"),
                metric.names = c("R-squared", "Correlation (r)"),
                type = "iqr")

    plot1 <- class_models_dotplot +
        theme(legend.position = "none",
            plot.margin = ggplot2::margin(1.5, 0.5, 0.5, 0, unit = "lines")) +
        ylim(0,1)
    plot2 <- reg_models_dotplot +
        theme(#legend.position = "none",
              legend.position = c(0.98, 0.97),
              legend.justification = c("right", "top"),
              legend.box.just = "right",
              legend.background = element_rect(fill = "grey90"),
              legend.margin = ggplot2::margin(-1.8, 1.8, 1.8, 1.8),
              legend.key.size = unit(1, "lines"),
              legend.text = element_text(size = 10),
              legend.title = element_blank(),
              plot.margin = ggplot2::margin(1.5, 0.5, 0.5, 0, unit = "lines")) +
        ylim(0,1)
    pgrid <- plot_grid(plot1, plot2, ncol = 2, rel_widths = c(1, 1),
                       vjust = 1.5, hjust = -1.3,
                       labels = c("y = Replicated (Yes/No)", "y = Relative Effect Size"),
                       label_size = 14,
                       label_fontface = "plain"
                      )
    # add legend with outer plot grid
    # pgrid <- plot_grid(pgrid, get_legend(class_models_dotplot +
    #     theme(legend.margin = ggplot2::margin(0, unit = "lines"))),
    #     ncol = 1, rel_heights = c(1, .07))

    save_plot(paste0(fig_dir, "/", col_suffix[i], "/si/", "S3_Fig", fig_filetype), pgrid, #models_comp_joined
            ncol = 2, nrow = 1,
            #base_height = 6,
            dpi = fig_dpi, base_width = fig_width,
            base_aspect_ratio = 0.9)
}

# ==============================================================================
# Evalutaion of SSRP out-of-sample Predictions
# ==============================================================================

key <- fread("../SSRP-evaluation/ssrp_data/codebook.csv")
predictions <- fread("../SSRP-evaluation/model/model_predictions.csv")
results <- fread("../SSRP-evaluation/ssrp_data/D3 - ReplicationResults.csv")
beliefs <- fread("../SSRP-evaluation/ssrp_data/D6 - MeanPeerBeliefs.csv")

predictions <- key[predictions, on = .(id)]
predictions[, pred_prep_pool := ifelse(p_s1 <= 0.5, p_s2, p_s1)]
predictions[, pred_es_pool := ifelse(p_s1 <= 0.5, es_s2, es_s1)]
setnames(predictions,
         c("p_s1", "p_s2", "es_s1", "es_s2"),
         c("pred_prep_s1", "pred_prep_s2", "pred_es_s1", "pred_es_s2"))

setnames(results,
         c("study",
           "r_os", "p_os",
           "r_rs1", "p_rs1", "rep_sr_rs1",
           "r_rs2", "p_rs2", "rep_sr_rs2",
           "r_rp", "p_rp", "rep_sr_rp"),
         c("ssrp_id",
           "orig_r", "orig_p",
           "res_es_s1", "res_p_s1", "res_s1",
           "res_es_s2", "res_p_s2", "res_s2",
           "res_es_pool", "res_p_pool", "res_pool"))
results <- results[, .(ssrp_id, orig_r, orig_p,
                       res_es_s1, res_p_s1, res_s1,
                       res_es_s2, res_p_s2, res_s2,
                       res_es_pool, res_p_pool, res_pool)]

setnames(beliefs, c("study"), c("ssrp_id"))
beliefs_s1 <- beliefs[treatment == "m2",
                      .(ssrp_id, pm_prep_s1 = m2_p, surv_prep_s1 = m2_b)]
beliefs_s2 <- beliefs[treatment == "m3",
                      .(ssrp_id, pm_prep_s2 = m3_p, surv_prep_s2 = m3_b)]

# Merge
dt <- results[predictions, on = .(ssrp_id)]
dt <- dt[beliefs_s1, on = .(ssrp_id)]
dt <- dt[beliefs_s2, on = .(ssrp_id)]

# ===========================
# 0/1 Replication Prediction

rep_dt <- list(s1 = dt[, .(label,
                           results = factor(res_s1, levels = c(0,1), labels = c("Not Replicated", "Replicated")),
                           Model = pred_prep_s1,
                           Market = pm_prep_s1,
                           Survey = surv_prep_s1
                           )],
               pool = dt[, .(label,
                             results = factor(res_pool, levels = c(0,1), labels = c("Not Replicated", "Replicated")),
                             Model = pred_prep_pool,
                             Market = pm_prep_s2,
                             Survey = surv_prep_s2
                             )])


# =====================
# Relative Effect Size

es_dt <- list(s1 = dt[, .(label,
                          es = res_es_s1 / orig_r,
                          pred_es = pred_es_s1,
                          results = factor(res_s1, levels = c(0,1), labels = c("Not Replicated", "Replicated")),
                          pred_results = factor(ifelse(pred_prep_s1 >= 0.5, 1, 0), levels = c(0,1), labels = c("Not Replicated", "Replicated")))],
              pool = dt[, .(label,
                            es = res_es_pool / orig_r,
                            pred_es = pred_es_pool,
                            results = factor(res_pool, levels = c(0,1), labels = c("Not Replicated", "Replicated")),
                            pred_results = factor(ifelse(pred_prep_pool >= 0.5, 1, 0), levels = c(0,1), labels = c("Not Replicated", "Replicated")))])



for (s in 1:2) {
    # Min and max predictions
    rep_dt[[s]][, xmin := pmin(Model, Market, Survey)]
    rep_dt[[s]][, xmax := pmax(Model, Market, Survey)]

    # Order labels
    # By Predicted Replication Probability
    lvls <- rep_dt[[s]][order(-Model), .(label, Model)]
    # By squared error
    lvls <- es_dt[[s]][, .(label, (pred_es - es) ^ 2)][order(-V2), label]

    # 1/0 replication
    rep_dt[[s]][, label := factor(label, levels = lvls, labels = lvls)]

    # Effect Size
    es_dt[[s]][, label := factor(label, levels = lvls, labels = lvls)]
}


# ======
# PLOTS

theme_set(pr_theme_paper() +
          theme(legend.position = "right",
                legend.margin = ggplot2::margin(0, -0.2, -1, -0.6, unit = "lines"),
                legend.key.size = unit(1, "lines"),
                legend.spacing.x = unit(0.1, "lines"),
                legend.spacing.y = unit(1, "lines"),
                #legend.box.margin = ggplot2::margin(0, 0, 0, -1, unit = "lines"),
                legend.text = element_text(size = 7),
                axis.title = element_text(size = 7, face = "bold"),
                axis.text = element_text(size = 6)
                ))


es_plot <- function(dt) {
    ggplot(dt, aes(y = label)) +
        geom_vline(aes(xintercept = 1), size = 0.15, linetype = 2) +
        geom_segment(aes(x = es, xend = pred_es, yend = label),
                     size = 0.15) +
        geom_point(aes(x = es, color = "Actual", shape = results), size = 1) +
        geom_point(aes(x = pred_es, color = "Predicted", shape = results), size = 1) + # shape = pred_results
        xlim(-0.5,1.5) +
        labs(x = "Relative Effect Size", y = "") +
        guides(color = guide_legend(title = NULL),
               shape = guide_legend(title = NULL))
}

rep_plot <- function(dt) {
    ggplot(dt, aes(y = label)) +
        geom_vline(aes(xintercept = 0.5), size = 0.15, linetype = 2) +
        geom_segment(aes(x = xmin, xend = xmax, yend = label),
                     size = 0.15) +
        geom_point(aes(x = Model, color = "Model", shape = results), size = 1) +
        geom_point(aes(x = Market, color = "Market", shape = results), size = 1) +
        geom_point(aes(x = Survey, color = "Survey", shape = results), size = 1) +
        xlim(0, 1) +
        labs(x = "Predicted Replication Probability", y = "") +
        guides(color = guide_legend(title = NULL),
               shape = guide_legend(title = NULL))
}

for (s in 1:2) {
    # Create ggplot objects
    g_rep <- rep_plot(rep_dt[[s]])
    g_es <- es_plot(es_dt[[s]])

    # Both color and BW
    for (i in col_suffix) {

        if (i == "color") {
            colors_es <- brewer.pal(5, "Set1")[4:5]
            colors_rep <- brewer.pal(5, "Set1")[1:3]
        } else if (i == "bw") {
            colors_es <- brewer.pal(3, "Greys")[2:3]
            colors_rep <- brewer.pal(4, "Greys")[2:4]
        }

        plot1 <- g_es +
            scale_color_manual(limits = c("Actual", "Predicted"),
                               labels = expression(Effect~Size[true],
                                                   Effect~Size[predicted]),
                               values = colors_es) +
            theme(axis.text.y = element_text(hjust = 1.02, vjust = 0.4),
                  plot.margin = ggplot2::margin(0.5, 0, 0.2, 0, unit = "lines"))

        plot2 <- g_rep +
            scale_color_manual(limits = c("Model", "Market", "Survey"),
                               labels = expression(Model~P[rep], Market~P[rep], Survey~P[rep]),
                               values = colors_rep) +
            theme(axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  plot.margin = ggplot2::margin(0.5, 0.5, 0.2, 0, unit = "lines"))

        # Fake data to create common label.
        f_dt <- data.table(label = 1:5,
                          results = c("Not Replicated", "Replicated", "Replicated", "Replicated", "Replicated"),
                          variable = c("Actual ES", "Predicted ES", "Model", "Market", "Survey"),
                          value = 1:5)

        f_dt[, `:=`(variable = factor(variable, levels = variable),
                    results = factor(results, levels = c("Not Replicated", "Replicated")))]

        g <- ggplot(data = f_dt, aes(y = label, x = value, color = variable, shape = results)) +
            geom_point() +
            scale_color_manual(labels = expression("Actual", "Predicted", Model~P[rep], Market~P[rep], Survey~P[rep]),
                               values = c(colors_es, colors_rep)) +
            guides(color = guide_legend(title = NULL),
                   shape = guide_legend(title = NULL)) +
            theme(legend.text = element_text(size = 6),
                  legend.key.size = unit(1, "lines"),
                  legend.text.align = 0,
                  legend.margin = ggplot2::margin(0, 1, 0, -2),
                  legend.spacing.x = unit(0.1, "lines"),
                  legend.spacing.y = unit(0.5, "lines"))

        if (i == "color") {
            # Color version has all legends on right side, as colors vary.
            # BW needs to have legends plot-by-plot.

            plot1 <- plot1 +
                theme(legend.position = "none")

            plot2 <- plot2 +
                theme(legend.position = "none")

            plots <- plot_grid(plot1, plot2, ncol = 2, nrow = 1,
                            rel_widths = c(1.65, 1))

            pgrid <- plot_grid(plots, get_legend(g), nrow = 1, ncol = 2, rel_widths = c(1, 0.16))

        } else if (i == "bw") {
            legend_theme_top <- theme(
                legend.position = "top",
                legend.justification = "right",
                legend.text = element_text(size = 6),
                legend.key.size = unit(0.9, "lines"),
                #legend.text.align = 0,
                legend.margin = ggplot2::margin(1, 0, -1, 0),
                legend.box.margin = ggplot2::margin(-10, 0, -10, 0),
                legend.spacing.x = unit(0.1, "lines"),
                legend.spacing.y = unit(0.1, "lines")
            )

            plot1 <- plot1 + legend_theme_top + guides(shape = FALSE)

            plot2 <- plot2 + legend_theme_top + guides(shape = FALSE)

            plots <- plot_grid(plot1, plot2, ncol = 2, nrow = 1, rel_widths = c(1.65, 1))

            # Only use shape legend from fake data
            g <- g + guides(color = FALSE)
            pgrid <- plot_grid(plots, get_legend(g), nrow = 1, ncol = 2, rel_widths = c(1, 0.16))
        }

        fig_name <- ifelse(s == 1, "S4_Fig", "Fig5")
        subfolder <- if(s == 2) i else paste0(i, "/si")
        save_plot(paste0(fig_dir, "/", subfolder, "/", fig_name, fig_filetype), #"/oos_grid_s", s,
                  plot = pgrid, scale = 1,
                  ncol = 2, nrow = 2,
                  dpi = fig_dpi,
                  base_height = fig_width / 4,
                  base_width = fig_width / 2)

        if (i == "color") {
            save_plot(paste0(fig_dir, "/", "presentation/oos_grid_s", s, fig_filetype),
                  plot = pgrid, scale = 0.5,
                  ncol = 2, nrow = 2,
                  dpi = fig_dpi,
                  base_height = fig_height_pres,
                  base_aspect_ratio = fig_aspect_ratio_pres)
        }
    }
}

# ==============================================================================
# ROC Curve
# ==============================================================================

training <- data.table(fit_class.full$pred[fit_class.full$pred$mtry == fit_class.full$finalModel$mtry, ])
training[, obs := as.integer(obs) - 1]
rocdata <- rbind(
    training[, .(D=obs, M=replicated, Type = "Validation Set (AUC=0.79)")],
    dt[,  .(D=res_pool, M=pred_prep_s2, Type = "OOS Test Set (AUC=0.71)")]
)

g <- ggplot(data = rocdata, aes(m = M, d = D, color = Type)) +
    geom_roc(size = 0.7,
             #n.cuts = 2,
             cutoffs.at = c(0.250, 0.50, 0.750),
             pointalpha = 1,
             linealpha = 1, # 0.8
             pointsize = 0.4,
             labelsize = 1.5, labelround = 1,
             show.legend = TRUE) +
    coord_equal() +
    style_roc(theme = theme_bw, guide = TRUE) + geom_abline(color = "grey50", alpha = 1) + # 0.1
    theme(legend.text = element_text(size = 6),
          legend.key.size = unit(1, "lines"),
          legend.text.align = 0,
          legend.margin = ggplot2::margin(0, 0, 0, 0),
          legend.spacing.x = unit(0.1, "lines"),
          legend.spacing.y = unit(0.5, "lines"),
          axis.text.x = element_text(angle = 40, vjust = 1, hjust = 1))

# g <- direct_label(g,
#                   labels = c("OOS Test Set (AUC=0.71)",
#                              "Validation Set (AUC=0.79)"),
#                   label.angle = -45,
#                   size = 1.8,
#                   nudge_x = c(0.18, 0.18),
#                   nudge_y = c(-0.10, 0.1))

save_plot(file.path(fig_dir, "color", paste0("Fig6", fig_filetype)),
                plot = g + scale_colour_brewer(palette = "Set1"),
                scale = 1,
                ncol = 2, nrow = 2,
                dpi = fig_dpi,
                base_height = fig_width / 4,
                base_width = fig_width / 2)

save_plot(file.path(fig_dir, "presentation", paste0("ROC_plot", fig_filetype)),
                plot = g + scale_colour_brewer(palette = "Set1"),
                scale = 1,
                ncol = 2, nrow = 2,
                dpi = fig_dpi,
                base_height = fig_height_pres,
                base_aspect_ratio = fig_aspect_ratio_pres)

save_plot(file.path(fig_dir, "bw", paste0("Fig6", fig_filetype)),
                plot = g + scale_colour_grey(),
                scale = 1,
                ncol = 2, nrow = 2,
                dpi = fig_dpi,
                base_height = fig_width / 4,
                base_width = fig_width / 2)

# ==============================================================================
# Calculate important stats
# ==============================================================================

info <- function(...) {
    cat(paste0(..., "\n"))
}

sink(file.path(fig_dir, "stats.txt"))
info("--------------")
info("Classification")
info("Training:")
info("- accuracy (CV mean): ", round(mean(class_rf_df$values$`Random Forest~Acc`), 3))
info("- accuracy (CV median): ", round(median(class_rf_df$values$`Random Forest~Acc`), 3))
info("- accuracy (full data): ", round((confusionMatrix(fit_class.full)$table[1,1] + confusionMatrix(fit_class.full)$table[2,2]) / 100, 3))
info("- AUC (CV mean): ", round(mean(class_rf_df$values$`Random Forest~ROC`), 3))
info("- AUC (CV median): ", round(median(class_rf_df$values$`Random Forest~ROC`), 3))
info("- AUC (full data): ", round(getTrainPerf(fit_class.full)$TrainROC, 3))
info("OOS:")
info("- accuracy (pooled): ", round(rep_dt$pool[, .(outcome=as.integer(results)-1, pred=(ifelse(Model >= 0.5, 1, 0)))][, mean(outcome == pred)], 3))
info("- accuracy (S1): ", round(rep_dt$s1[, .(outcome=as.integer(results)-1, pred=(ifelse(Model >= 0.5, 1, 0)))][, mean(outcome == pred)], 3))
info("- AUC (pooled):", round(suppressMessages(pROC::auc(rep_dt$pool[, as.integer(results) - 1], rep_dt$pool$Model)), 3))
info("- AUC (S1):", round(suppressMessages(pROC::auc(rep_dt$s1[, as.integer(results) - 1], rep_dt$s1$Model)), 3))

info("----------")
info("Regression")
info("Training:")
info("- RMSE (CV mean): ", round(mean(reg_rf_df$values$`Random Forest~RMSE`), 3))
info("- RMSE (CV median): ", round(median(reg_rf_df$values$`Random Forest~RMSE`), 3))
info("- RMSE (full data): ", round(getTrainPerf(fit_reg.full)$TrainRMSE, 3))
info("- rho (CV mean): ", round(mean(reg_rf_df$values$`Random Forest~Corr`), 3))
info("- rho (CV median): ", round(median(reg_rf_df$values$`Random Forest~Corr`), 3))
info("- rho (full data): ", round(suppressWarnings(cor.test(fit_reg.full$pred[fit_reg.full$pred$mtry == fit_reg.full$finalModel$mtry,]$obs, fit_reg.full$pred[fit_reg.full$pred$mtry == fit_reg.full$finalModel$mtry,]$pred, method = "spearman")$estimate), 3))
info("- R2 (CV mean): ", round(mean(reg_rf_df$values$`Random Forest~Rsquared`), 3))
info("- R2 (CV median): ", round(median(reg_rf_df$values$`Random Forest~Rsquared`), 3))
info("- R2 (full data): ", round(getTrainPerf(fit_reg.full)$TrainRsquared, 3))
info("OOS:")
info("- RMSE (pool):", round(sqrt(mean((es_dt$pool$es-es_dt$pool$pred_es)^2)), 3))
info("- RMSE (s1):", round(sqrt(mean((es_dt$s1$es-es_dt$s1$pred_es)^2)), 3))
info("- rho (pool): ", round(suppressWarnings(cor.test(es_dt$pool$pred_es, es_dt$pool$es, method = "spearman"))$estimate, 3))
info("- rho (s1): ", round(suppressWarnings(cor.test(es_dt$s1$pred_es, es_dt$s1$es, method = "spearman"))$estimate, 3))
info("- R2 (pool): ", round(suppressWarnings(cor.test(es_dt$pool$pred_es, es_dt$pool$es, method = "pearson"))$estimate^2, 3))
info("- R2 (s1): ", round(suppressWarnings(cor.test(es_dt$s1$pred_es, es_dt$s1$es, method = "pearson"))$estimate^2, 3))
sink()

# EXPORT EPS
# system2("find", args = c("../figures/**/*", "-name \"*.pdf\"", "-type f", "-exec pdftops -eps {} \\;"))
system2("find", args = c("../figures/**/*", "-name \"*.pdf\"", "-type f", "-exec gs -dNOPAUSE -dNoOutputFonts -sDEVICE=eps2write -o {}.eps {} \\;"))
