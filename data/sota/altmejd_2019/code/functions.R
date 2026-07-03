###############################################################################
# Functions for data generation and manipulation
###############################################################################


###############################################################################
# Data generation and manipulation
###############################################################################
# Function takes effect sizes and translates to corr. coeff (r)
# Effect size function based on Eskil Forsell's effect size calculations
# References:
# [1] Field, A. P. (2001). Meta-analysis of correlation coefficients: a Monte Carlo comparison of fixed-and random-effects methods. Psychological Methods, 6(2), 161.
# [2] Lipsey, M. W., & Wilson, D. B. (2001). Practical meta-analysis (Vol. 49). Sage publications Thousand Oaks, CA. Retrieved from http://www2.jura.uni-hamburg.de/instkrim/kriminologie/Mitarbeiter/Enzmann/Lehre/StatIIKrim/Wilson.pdf
# [3] Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. Science, 349(6251). http://doi.org/10.1126/science.aac4716
# [4] Rosenberg, M. S., & Plaistow, S. (2010). A generalized formula for converting chi-square tests to effect sizes for meta-analysis. PloS One, 5(4). Retrieved from http://dx.plos.org/10.1371/journal.pone.0010059
# [5] Viechtbauer, W., & others. (2010). Conducting meta-analyses in R with the metafor package. Journal of Statistical Software, 36(3), 1-48.
# [6] Fisher transformation. (2015, November 23). In Wikipedia, the free encyclopedia. Retrieved from https://en.wikipedia.org/w/index.php?title=Fisher_transformation&oldid=691927443
# [7] Lakens: https://osf.io/vbdah/

get_es <- function(x, stat, p1, p2) {
        d_to_r <- function(d, n1, n2) {
                # if no n is provided assume equal groups
                if(is.na(n1) | is.na(n2)) {
                        n1 <- n2 <- 1
                        warning("No 'n' provided, assuming equal groups.")
                }
                a <- ( (n1 + n2) ^ 2) / (n1 * n2)
                r <- d / sqrt(d ^ 2 + a)
                return(r)
        }
        f_to_r <- function(f, df1, df2) {
                # See ref. [3]
                return(sqrt( (f * (df1 / df2)) / ( ( (f * df1) / df2) + 1)) *
                             sqrt(1 / df1))
        }
        t_to_r <- function(t, df1, df2) {
                # See ref. [3]
                return(sign(t) * sqrt( (t ^ 2) / (t ^ 2 + df1)))
        }
        chi2_to_r <- function(chi2, df, N=df) {
                # See refs. [3,4]
                # uses N (sample size) not test df
                return(sqrt(chi2 / N))
        }
        z_to_r <- function(z, n, p2) {
                # See refs. [3,6]
                fisher_z <- z * sqrt(1 / (n - 3))
                return(tanh(fisher_z))
        }
        switch(tolower(stat),
                     d = return(d_to_r(x, p1, p2)),
                     f = return(f_to_r(x, p1, p2)),
                     t = return(t_to_r(x, p1, p2)),
                     z = return(z_to_r(x, p1, p2)),
                     chi2 = return(chi2_to_r(x, p1, p2)),
                     eta2 = return(sqrt(x)), # [7]
                     r = return(x))
        return(NA)
}

###############################################################################
# Function takes statistic and returns p value (two-tailed)
# Inspired by the RPP p-value recalculation function
# (Written by CHJ Hartgerink, RCM van Aert, MALM van Assen)

get_p <- function(x, stat, p1, p2) {
        f_to_p <- function(f, df1, df2) {
                return(pf(f, df1, df2, lower.tail = FALSE))
        }
        t_to_p <- function(t, df, p2) {
                return(pt(abs(t), df, lower.tail = FALSE) * 2)
        }
        chi2_to_p <- function(chi2, df, n) {
                return(pchisq(chi2, df, lower.tail = FALSE))
        }
        z_to_p <- function(z, p1, p2) {
                return(pnorm(abs(z), lower.tail = FALSE) * 2)
        }
        r_to_p <- function(r, df, N) {
                fis.r <- 0.5 * log( (1 + abs(r)) / (1 - abs(r)))
                se.fis.r <- sqrt(1 / (N - 3))
                return(pnorm(fis.r, mean = 0, sd = se.fis.r, lower.tail = FALSE) * 2)
        }
        d_to_p <- function(d, n1, n2) {
                return(r_to_p(get_es(d, "d", n1, n2), NA, n1 + n2))
        }
        switch(tolower(stat),
                     d = return(d_to_p(x, p1, p2)),
                     f = return(f_to_p(x, p1, p2)),
                     t = return(t_to_p(x, p1, p2)),
                     z = return(z_to_p(x, p1, p2)),
                     chi2 = return(chi2_to_p(x, p1, p2)),
                     r = return(r_to_p(x, p1, p2)))
        return(NA)
}

################
# Function for cleaning bibliographic data imported from bib file
bib_clean <- function(df) {
      df <- lapply(df, gsub, pattern = "\\textquotesingle", replacement = "'")
      df <- lapply(df, gsub, pattern = "\\\\\\^\\\\i", replacement = "î")
      df <- lapply(df, gsub, pattern = "\\\\'\\\\i", replacement = "í")
      df <- lapply(df, gsub, pattern = "[{}]", replacement = "")
      df <- lapply(df, gsub, pattern = "\\\\", replacement = "")
      df <- lapply(df, gsub, pattern = "\\\"([a-zA-Z]*)\\\"", replace = "“\\1”")
      return(as.data.frame(df))
}

################
# Remove columns from data table
remove_cols <- function(dt, cols, copy=FALSE) {
    if (copy) { return(copy(dt)[, (cols) := NULL]) }
    else { dt[, (cols) := NULL] }
}

################
# finds the modal (most common) factor
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

###############################################################################
# TRAINING AND SELECTION
###############################################################################

# Function to train a classification model
run_class_model <- function(x, y, method, seed=123, tr=NULL, ...) {
    require(caret)
        if(is.null(tr)) {
            tr <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 10,
                               classProbs = TRUE,
                               savePredictions = TRUE,
                               summaryFunction = twoClassSummary,
                               preProcOptions = list(thresh = 0.95) # for pca
                               )
        }
        set.seed(seed) # we reset the seed each time to pick same training folds
        train(x, y, method = method, trControl = tr, metric = "ROC", ...)
}

# Function to train regression model
run_reg_model <- function(x, y, method, seed=123, tr=NULL, ...) {
    require(caret)
        if (is.null(tr)) {
            tr <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 10,
                               savePredictions = TRUE,
                               summaryFunction = defaultSummary
                               )
        }
        set.seed(seed) # we reset the seed each time to pick same training folds
        train(x, y, method = method, trControl = tr, metric = "RMSE", ...)
}

# Summary function that takes a custom threshold level into account
my_class_summary <- function(df, t) {
    # caret: function(data, lev = levels(data$obs), model = NULL, t = 0.5) {
    # data is a data frame generated by train()
    # first two columns are "obs" and "pred" with observed and predicted outcomes
    # adapted from twoClassSummary()
    lev <- levels(df$obs)
    if (length(lev) > 2)
        stop(paste("Your outcome has", length(lev), "levels. The my_class_summary() function isn't appropriate."))
    if (!all(levels(df[, "pred"]) == lev))
        stop("levels of observed and predicted data do not match")
    if (ncol(df) != 4) stop("Provided data needs 4 cols, make sure classProbs is on in train()")
    #requireNamespaceQuietStop("ModelMetrics") # from caret version
    #suppressPackageStartupMessages(require(caret))

    # Generate positive class observed
    # (we use lev[2] to get "replicated" as positive)
    df$y <- ifelse(df$obs == lev[2], 1, 0)
    # When {classProbs = TRUE} "data" contains 4 cols
    # (two last named after classes containing probs)
    # we only return metrics at the chosen threshold (and not at 0.5)
    out <-
        c(ModelMetrics::auc(df$y, df[, lev[2]]),
          #caret::sensitivity(df[, "pred"], df[, "obs"], lev[1]),
          #caret::specificity(df[, "pred"], df[, "obs"], lev[2]),
          #sum(df[, "pred"] == df[, "obs"])/nrow(df),
          ModelMetrics::sensitivity(df$y, df[, lev[2]], cutoff = t),
          ModelMetrics::specificity(df$y, df[, lev[2]], cutoff = t),
          sum(ifelse(df[, lev[2]] > t, lev[2], lev[1]) == df$obs) / nrow(df))
    #names(out) <- c("ROC", "Sens", "Spec", "Acc", "SensThresh", "SpecThresh", "AccThresh")
    names(out) <- c("ROC", "TPR", "TNR", "Acc")
    return(out)
}

# Summary function for regression (with nested vars)
my_reg_summary <- function(data) {
    out <- c(caret::postResample(pred = data$pred, obs = data$obs),
             # Corr = cor(data$pred, data$obs, method = "pearson"))
             Corr = cor(data$pred, data$obs, method = "spearman"))
    return(out)
}

my_confmatrix <- function(df, t) {
    # Called by nested CV results to return confmatrix for each CV run
    # not used right now
    lev <- levels(df$obs)
    pos <- lev[2]
    preds <- factor(ifelse(df[, pos] >= t, lev[2], lev[1]))
    cm <- confusionMatrix(preds, df$obs, positive = lev[2])
    out <- cm$table[1:4]
    names(out) <- c("A", "B", "C", "D") # number of obs in each conf-box
    return(out)
}

###############################################################################
# NESTED CV
###############################################################################

gen_results <- function(m, x, y, resultfunc, opt_thresh = FALSE) {
    # m, x, y can be lists of arbitrary (but equal) length
    # where x, y are lists w. test data (x model matrices and y outcome vectors)
    if (length(m) != length(x) | length(m) != length(y))
        stop("Lists not equally long.")

    type <- unique(sapply(m, function(x) x$modelType))

    gen_preds <- function(m, x, y) {
        # generates a prediction data to be used in caret summary function
        if (m$modelType == "Regression") {
            return(data.frame(obs = y, pred = predict(m, x)))
        }
        if (m$modelType == "Classification") {
            return(cbind(data.frame(obs = y, pred = predict(m, x)),
                         predict(m, x, type = "prob")))
        }
    }
    get_opt_thresh <- function(m) {
        # get opt threshold from training data
        suppressPackageStartupMessages(require(pROC))
        df <- merge(m$pred, m$bestTune)
        roc <- roc(response = df$obs, predictor = df$replicated)
        cutoff <- coords(roc, "best",
                         #best.method="closest.topleft", # closest to topleft
                         best.method = "youden", # furthest away from identity
                         #best.weights = # add weights due to class imbalance?
                         ret = c("threshold")
                         )
        # coords("best") may return multiple values, return the first
        return(cutoff[1])
    }

    out <- mapply(gen_preds, m, x, y, SIMPLIFY = FALSE)

    if (type == "Regression") out <- mapply(resultfunc, out, SIMPLIFY = FALSE)
    else if (type == "Classification") {
        out <- mapply(resultfunc, out, ifelse(opt_thresh,
                                              lapply(m, get_opt_thresh), 0.5),
                      SIMPLIFY = FALSE)
    }
    out <- do.call(rbind, out)
    return(out)
}

merge_results <- function(r) {
    # r is list of result data frames generated by gen_results
    # names is list of model names
    # generates data frame like the one generated by caret::resamples$x
    out <- mapply(function(x, names) {
        colnames(x) <- paste0(names, "~", colnames(x))
        out <- data.frame(Resample = rownames(x), x, check.names = FALSE)
        rownames(out) <- NULL
        return(out)
    }, r, names(r), SIMPLIFY = FALSE)
    return(Reduce(merge, out))
}

###############################################################################
# PLOTTING AND PRINTING
###############################################################################

# GGPLOT2 THEMING
pr_theme <- function() {
    require(ggplot2)
    mytheme <- theme_bw() +
        theme(axis.title = element_text(size = 16, face = "bold"),
            legend.title = element_text(hjust = 0, size = 16, face = "bold"),
            legend.text = element_text(size = rel(1)),
            panel.grid = element_blank()
            )
    return(mytheme)
}
# pr_colors <- function() {
#     colors <- scale_color_brewer(palette="Set1")
#     return(colors)
# }
pr_theme_paper <- function() {
    # return(list(pr_theme(), pr_colors()))
    return(pr_theme())
}
pr_theme_presentation <- function() {
    # return(list(pr_theme(), pr_colors()))
    return(pr_theme())
}

# Plot ROC with classification models
rocplot <- function(rocs, names, caption="ROC", ...) {
    if (!exists("bw")) bw <- FALSE  # default to color
    cols <- brewer.pal(length(names), ifelse(bw, "Greys", "Set1"))
    library(scales)

    par(cex.lab=1.4)

    if (length(rocs) != length(names)) { error("Name not provided for all models.")}
    for (i in length(rocs):1) {
            plot.roc(rocs[[i]], add = ifelse(i != length(rocs), TRUE, FALSE),
                     print.thres = TRUE,
                     print.thres.pattern = "%.2f",
                     print.thres.adj = c(-0.07, 1.4),
                     col = scales::alpha(cols[i], 0.7),
                     lty = 2, # line type
                     lwd = 1.7, # line width
                     identity.lwd = 1,
                     identity.col = rgb(0.7,0.7,0.7),
                     xlab="True Negative Rate", ylab="True Positive Rate",
                     ...
                     )
    }
    labs <-
    legend("bottomright", paste0(names, " (AUC: ",
           round(sapply(rocs, function(x) pROC::auc(x)[[1]]), 2), ")" ),
           lty = 2, lwd = 1.7, col = cols, adj = c(0, 0.7), cex = 1.2)
    title(caption, line = 2.5, adj = 0.5)
}

# Print confusion matrix
printmat <- function(model, x, y, prob.cutoff=0.5) {
        lvls <- levels(y)
        probs <- predict(model, newdata=x, type = "prob")
        preds <- factor(ifelse(probs[, 2] >= prob.cutoff, lvls[2], lvls[1]))
        return(confusionMatrix(preds, y, positive = lvls[2]))
}

# Get the AUC for a model on new data
get_AUC <- function(model, x, y) {
    suppressPackageStartupMessages(require(ROCR))
    probs <- predict(model, newdata = x, type = "prob")
    pred <- prediction(probs$replicated, labels = (y == "replicated"))
    return(attributes(performance(pred, 'auc'))$y.value[[1]])
}

# My own plots for resamples and variable importance
PR_dotplot <- function (x, model.order, metric.order, metric.names = NULL,
                        type = NULL, conf.level = 0.95) {
    # Create plot data and convert to long format to apply t.test/IQR
    # x is object created by caret::resamples
    suppressPackageStartupMessages(require(reshape))
    suppressPackageStartupMessages(require(ggplot2))
    suppressPackageStartupMessages(require(RColorBrewer))
    if (is.null(type)) { warning("No type provided, assuming IQR."); type <- "iqr"}
    if (!exists("bw")) bw <- FALSE  # default to color

    # get correct and enough colors so that first model is red
    n <- length(model.order)
    color_palette <- if(bw) {
        brewer.pal(n+1, ifelse(bw, "Greys", "Set1"))[2:(n+1)]
    } else rev(c(brewer.pal(9, ifelse(bw, "Greys", "Set1"))[-6],
                 if (n > 8) rep("#000000", n - 8))[1:n])

    # creates a list of long data frames, one for each model-metric combo
    plotData <- data.table(reshape::melt(x$values, id.vars = "Resample"))
    tmp <- strsplit(as.character(plotData$variable), "~", fixed = TRUE)
    plotData$model <- unlist(lapply(tmp, function(x) x[1]))
    plotData$metric <- unlist(lapply(tmp, function(x) x[2]))

    # Include only those models and metrics that are specified
    plotData <- plotData[metric %in% metric.order]
    plotData <- plotData[model %in% model.order]

    # Split into list
    plotData$variable <- factor(as.character(plotData$variable))
    plotData <- split(plotData, plotData$variable)

    if (type == "ci") {
        # then apply t test to each of these list elements:
        results <- lapply(plotData, function(x, cl) {
            ttest <- t.test(x$value, conf.level = cl)
            out <- c(ttest$estimate, ttest$conf.int[1], ttest$conf.int[2])
            names(out) <- c("center", "ymin", "ymax")
            return(out)
        }, cl = conf.level)
    } else if (type == "iqr") {
        results <- lapply(plotData, function(x) {
            q <- quantile(x$value, type = 7) # using type 7, default
            out <- c(q[3], q[2], q[4])
            names(out) <- c("center", "ymin", "ymax")
            return(out)
        })
    }
    results <- as.data.frame(do.call("rbind", results))
    tmp <- strsplit(as.character(rownames(results)), "~", fixed = TRUE)
    # for some reason plot wants reversed order (from bottom up)
    results$model <- factor(unlist(lapply(tmp, function(x) x[1])),
                            levels = rev(model.order))
    results$metric <- factor(unlist(lapply(tmp, function(x) x[2])),
                             levels = metric.order)
    if (!is.null(metric.names)) { levels(results$metric) <- metric.names }
    rownames(results) <- NULL

    g <- ggplot(data = results,
                aes(x = model, y = center, color = model)) +
        geom_point() +
        coord_flip() +
        geom_errorbar(aes(ymin = ymin, ymax = ymax), width = 0.3) +
        scale_y_continuous(limits = c(0, 1)) +
        facet_grid(metric ~ ., switch = "y", scales = "free", space = "fixed") +
        scale_colour_manual(values = color_palette) +
        pr_theme_paper() %+replace%
        theme(axis.title = element_blank(), axis.text.y = element_blank(),
              axis.ticks.y = element_blank(),
              strip.text = element_text(size = 9),
              panel.grid.major.x = element_line(color = "gray"),
              legend.position="bottom", legend.title = element_blank()) +
        guides(color = guide_legend(title = "Model", reverse = TRUE),
               shape = guide_legend(title = NULL))
    return(g)
}

PR_boxplot <- function(x, model.order, metric.order, metric.names = NULL,
                       conf.level = 0.95) {
    suppressPackageStartupMessages(require(reshape))
    suppressPackageStartupMessages(require(ggplot2))
    suppressPackageStartupMessages(require(RColorBrewer))
    if (!exists("bw")) bw <- FALSE  # default to color
    color_palette <- if(bw) {
        brewer.pal(n+1, ifelse(bw, "Greys", "Set1"))[2:(n+1)]
    } else rev(c(brewer.pal(9, ifelse(bw, "Greys", "Set1"))[-6],
                 if (n > 8) rep("#000000", n - 8))[1:n])

    df <- reshape::melt(x$values, id.vars = "Resample")
    tmp <- strsplit(as.character(df$variable), "~", fixed = TRUE)
    df$model <- unlist(lapply(tmp, function(x) x[1]))
    df$metric <- unlist(lapply(tmp, function(x) x[2]))
    df$variable <- factor(as.character(df$variable))
    df <- split(df, df$variable)

    df <- as.data.frame(do.call("rbind", df))
    df <- data.frame(metric = factor(df$metric, levels = metric.order),
                     model = factor(df$model, levels = rev(model.order)),
                     y = df$value)
    if (!is.null(metric.names)) { levels(df$metric) <- metric.names }

    g <- ggplot(data = df,
                aes(x = model, y = y, color = model)) +
            geom_boxplot(notch = TRUE) +
            coord_flip() +
            facet_grid(metric ~ ., switch = "y",
                       scales = "free", space = "fixed") +
            scale_colour_manual(values = color_palette) +
            pr_theme_paper() %+replace%
            theme(axis.title = element_blank(), axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  panel.grid.major.x = element_line(color = "gray"),
                  legend.position="bottom", legend.title = element_blank()) +
            guides(color = guide_legend(title = "Model", reverse = TRUE),
                   shape = guide_legend(title = NULL))
    return(g)
}

gen_varimp_df <- function(rf_models, lm_models, ame = NULL) {
    # Generates a variable importance plot
    require(caret)

    # Create variable importance tables for all random forest models
    vi <- Map(varImp, rf_models)

    # Generate a joined df with data from RF and LM
    types = c("replicated", "relative_es")

    if (length(vi) != length(lm_models)) stop("Unequal argument lengths (vi,lm_models).")
    if (!is.null(ame)) if (length(vi) != length(ame)) stop("Not enough AMEs")
    if (!is.list(vi) | !is.list(lm_models)) stop("Arguments should be lists.")

    out <- vector("list", length = length(vi))

    # Loop over all models, creating varImp objects
    for (i in 1:length(vi)) {
        if (class(vi[[i]]) != "varImp.train" & !("lm" %in%
                                                    class(lm_models[[i]])))
            stop("Arguments classes incorrect.")
        # caret::sortImp() returns Imp object as sorted df
        df <- sortImp(vi[[i]], dim(vi[[i]]$importance)[1])[, 1,
            drop = FALSE]
        df$Feature <- gsub("`", "", rownames(df)) # column with varnames
        rownames(df) = NULL
        df <- df[, c(2,1)] # order columns
        names(df) <- c("Feature", "Importance")
        lm.df <- as.data.frame(coef(summary(lm_models[[i]]))[-1,c(1,4)])
        lm.df$Feature <- gsub("`", "", rownames(lm.df))
        rownames(lm.df) <- NULL
        colnames(lm.df) <- c("Coef", "p", "Feature")
        if (!is.null(ame)) {
            require(margins) # can't read object if library is not loaded
            # replace coefficients with _average_ marginal effects
            lm.df$Coef <- summary(ame[[i]])$AME[lm.df$Feature]
        }
        stars <- ifelse(lm.df$p < 0.1,
                        ifelse(lm.df$p < 0.05,
                                ifelse(lm.df$p < 0.01, "***", "**"), "*"), "")
        lm.df$label <- paste0(sprintf("%2.2f", lm.df$Coef), stars)
        lm.df <- lm.df[, c(3, 4)]

        # merge data frame
        df <- merge(df, lm.df,
                    by = "Feature", all = TRUE, sort = FALSE)
        df$type <- types[i]
        out[[i]] <- df
    }

    # wide return format
    # df <- Reduce(function(x, y) merge(x, y, by = "Feature"), out)

    # long return format
    df <- do.call(rbind, out)

    return(df)
}

###############################################################################
# FETCHING AUTHOR DATA
###############################################################################

# This function searches google scholar for a name
# if it finds a unique results it outputs GS_id
# if multiple results it just opens the search window
# Not really useful as GS blocks robots :(
gs_author <- function(name) {
    library(rvest)
    data <- read_html(paste0('http://scholar.google.com/scholar?q=%22', gsub(" ", "+", name, fixed=TRUE),'%22'))

    results <- data %>% html_nodes("div#gs_ccl")

    if (grepl(paste0('Your search - "', name, '" - did not match any articles'), results %>% html_text())) {
        print("No results!")
        return(NULL)
    } else {
        results <- results %>% html_nodes("div.gs_r") %>% .[[1]]
        # check so that result is actually author
        if (!grepl('<div class="gs_ri">', results, fixed=TRUE)) {
            results <- results %>% html_nodes("td") %>% .[[-1]] %>%
                       html_nodes("a") %>% html_attr("href") %>%
                       gsub("^.*user=([a-zA-Z0-9]*)&.*$", "\\1", .)

            if(length(results) == 1) { return(results) }
            else {
                print(paste0(name, ": Multiple results, opening search in browser."))
                browseURL(paste0('https://scholar.google.com/scholar?q="', name,'"'), encodeIfNeeded=TRUE)
            }
        }
    }
    return(NULL)
}
get_gs_citations <- function(id) {
    library(scholar) # https://cran.r-project.org/web/packages/scholar/
    get_publications(id)
}

###############################################################################
# APPENDIX: UPPER BOUND SIMULATION
###############################################################################

# run_experiment() runs a number of "experiments" and outputs results
run_experiment <- function(beta, N, mean=0, sd=2, alpha=0.05) {
    # If vectors are provided call recursively function with mc()
    if (length(beta) > 1) {
        if (length(N) == 1) { N <- rep(N, length(beta)) }
        else if (length(beta) != length(N)) {stop("N, beta not same lenght")}
        return(data.table(beta = beta, N = N,
                          do.call(rbind, mcmapply(run_experiment, beta = beta,
                                                  N = N, SIMPLIFY = FALSE))))
    }
    else if (length(N) > 1) { stop("Trying to run with one beta and many N!") }
    # Otherwise, for singleton beta/n run experiment
    else {
        # X is just randomly assigned treatment (1) or not (0) in equal groups
        x <- c(rep(0, N / 2), rep(1, N / 2)) # faster not to randomize
        epsilon <- rnorm(n = N, mean = mean, sd = sd) # normal noise
        if (length(x) != length(epsilon)) { stop("Wrong length of x/epsilon!") }
        y <- beta * x + epsilon # generate outcomes with noise

        # Return p-value from means ttest and Cohen's d effect size
        tt <- t.test(y[x==1], y[x==0], var.equal = TRUE,
                     alternative = "two.sided", conf.level = 1 - alpha)

        d <- (mean(y[x==1]) - mean(y[x==0])) /
             sqrt( ( (N/2 - 1) * var(y[x==1]) + (N/2 - 1) * var(y[x==0]) ) /
                  (N/2 + N/2 - 2) )

        return(c(p = tt$p.value, d = d))
    }
}
