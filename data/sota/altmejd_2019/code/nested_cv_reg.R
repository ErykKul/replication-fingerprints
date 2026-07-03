# This code is designed to be loaded from "nested_cv.R" and ran on AWS
# Produces .RData file
# ==============================================================================
if (!exists("start.time")) start.time <- Sys.time()
require(data.table)
require(caret)
source("functions.R", local = TRUE)
load("../data/nested_cv_reg_data.RData")

#########################
# CREATE AND SAVE ALL CV FOLDS
set.seed(1234)
outer_folds <- unlist(replicate(repeats,
                                createFolds(reg_data$project, k = folds),
                                simplify = FALSE), recursive = FALSE)
listnames <- sprintf(paste0("Fold%0", 1 + floor(log10(folds)), "d.Rep%0",
                            1 + floor(log10(repeats)), "d"),
                     rep(1:folds, repeats), sort(rep(1:repeats, folds)))
names(outer_folds) <- listnames

#########################
# LIST TO SAVE MODELS IN
gen_m_list <- function(folds, repeats, names) {
    l <- vector("list", length = folds * repeats)
    names(l) <- names
    return(l)
}

fit.ref_disc <- gen_m_list(folds, repeats, listnames)
fit.ref_eso <- gen_m_list(folds, repeats, listnames)
fit.lm <- gen_m_list(folds, repeats, listnames)
fit.lasso <- gen_m_list(folds, repeats, listnames)
fit.rf <- gen_m_list(folds, repeats, listnames)
fit.gbm <- gen_m_list(folds, repeats, listnames)
fit.svm <- gen_m_list(folds, repeats, listnames)
fit.rf_basic <- gen_m_list(folds, repeats, listnames)
fit.rf_nodisc <- gen_m_list(folds, repeats, listnames)
fit.rf_no_rep <- gen_m_list(folds, repeats, listnames)

#########################
# TUNING GRIDS
gbm.grid <- expand.grid(interaction.depth = c(5,10), n.trees = 1001,
                        shrinkage = c(0.05), n.minobsinnode = c(2,5))
rf.grid <- expand.grid(mtry = seq(3, 10, 1))
svm.grid <- expand.grid(sigma = 0.02, C = seq(0.25, 1.5, 0.25))
lassogrid <- expand.grid(alpha = 1, lambda = seq(0.005, 0.1, 0.001))

# ==============================================================================
# NESTED CV START

for (i in 1:length(outer_folds)) {
    end.time <- Sys.time()
    time.taken <- end.time - start.time
    if (verbose) print(paste0(round(time.taken[[1]], 2), " ", units(time.taken),
                              ": Reg ", names(outer_folds)[i]))

    df.train <- model.frame(f_reg, reg_data[-outer_folds[[i]], ])
    x <- model.matrix(f_reg, df.train)
    y <- df.train$relative_es

    ##################
    # Reference models
    fit.ref_disc[[i]] <-
        run_reg_model(model.matrix(relative_es ~ 0 + discipline, df.train), y,
                      method = "lm")
    fit.ref_eso[[i]] <-
        run_reg_model(model.matrix(relative_es ~ 0 + effect_size.o, df.train),
                      y, method = "lm")

    ##################
    # Actual models
    fit.lm[[i]] <- run_reg_model(x, y, method = "lm")

    fit.lasso[[i]] <- run_reg_model(x, y, method = "glmnet",
                                    tuneGrid = lassogrid)

    fit.rf[[i]] <- run_reg_model(x, y, method = "rf", ntree = 1001,
                                 tuneGrid = rf.grid, importance = TRUE)

    fit.gbm[[i]] <- run_reg_model(x, y, method = "gbm", tuneGrid = gbm.grid,
                                  verbose = FALSE)

    fit.svm[[i]] <- run_reg_model(x, y, method = "svmRadial",
                                   tuneGrid = svm.grid,
                                   preProc = c("center", "scale"))

    ##################
    # Extra Models
    fit.rf_basic[[i]] <- run_reg_model(model.matrix(f_reg_basic, df.train), y,
                                       method = "rf", ntree = 1001,
                                       tuneGrid = rf.grid, importance = TRUE)

    fit.rf_nodisc[[i]] <-
        run_reg_model(model.matrix(update(f_reg, .~. - discipline), df.train),
                      y, method = "rf", ntree = 1001, tuneGrid = rf.grid,
                      importance = TRUE)

    fit.rf_no_rep[[i]] <- run_reg_model(model.matrix(f_reg_no_rep, df.train), y,
                                        method = "rf", ntree = 1001,
                                        tuneGrid = rf.grid, importance = TRUE)
}

# ==============================================================================
# Summarize results

df.test <- lapply(outer_folds,
                  function(index, data) model.frame(f_reg, data[index, ]),
                  data = reg_data)
x_test <- lapply(df.test, function(x) model.matrix(f_reg, x))
y_test <- lapply(df.test, function(x) x$relative_es)

reg_results <- list(
    "Ref. Discipline Dummies" = gen_results(fit.ref_disc,
        lapply(df.test, function(x) model.matrix(relative_es ~ 0 + discipline,
                                                 x)), y_test, my_reg_summary),
    "Ref. Original Effect Size" = gen_results(fit.ref_eso,
        lapply(df.test,
               function(x) model.matrix(relative_es ~ 0 + effect_size.o, x)),
        y_test, my_reg_summary),
    "LM (OLS or Logit)" = gen_results(fit.lm, x_test, y_test, my_reg_summary),
    "LASSO" = gen_results(fit.lasso, x_test, y_test, my_reg_summary),
    "Random Forest" = gen_results(fit.rf, x_test, y_test, my_reg_summary),
    "GBM" = gen_results(fit.gbm, x_test, y_test, my_reg_summary),
    "SVM" = gen_results(fit.svm, x_test, y_test, my_reg_summary),
    "Only Basic Features" = gen_results(fit.rf_basic,
        lapply(df.test, function(x) model.matrix(f_reg_basic, x)),
        y_test, my_reg_summary),
    "No Discipline" = gen_results(fit.rf_nodisc,
        lapply(df.test, function(x) model.matrix(update(f_reg,
                                                        .~. - discipline), x)),
        y_test, my_reg_summary),
    "No Replication Features" = gen_results(fit.rf_no_rep,
        lapply(df.test, function(x) model.matrix(f_reg_no_rep, x)),
        y_test, my_reg_summary)
)

# ==============================================================================
# Save image with models and results

save(list = c("outer_folds",
              "fit.ref_disc", "fit.ref_eso", "fit.lm",
              "fit.lasso", "fit.rf", "fit.gbm", "fit.svm", "fit.rf_basic",
              "fit.rf_nodisc", "fit.rf_no_rep"),
     file = "../data/nested_cv_reg_models.RData")

save(list = c("reg_results"),
     file = "../data/nested_cv_reg_results.RData")

# Code is run in a local environment so we can just remove all
rm(list=ls())
