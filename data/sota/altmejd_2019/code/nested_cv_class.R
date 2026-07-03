# This code is designed to be loaded from "nested_cv.R" and ran on AWS
# Produces .RData file
# ==============================================================================
if (!exists("start.time")) start.time <- Sys.time()
require(data.table)
require(caret)
source("functions.R", local = TRUE)
load("../data/nested_cv_class_data.RData")

#########################
# CREATE AND SAVE ALL CV FOLDS
set.seed(1234)
outer_folds <- unlist(replicate(repeats,
                                createFolds(interaction(class_data$replicated,
                                                        class_data$project),
                                            k = folds),
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

fit.ref_logit <- gen_m_list(folds, repeats, listnames)
fit.ref_disc <- gen_m_list(folds, repeats, listnames)
fit.glm <- gen_m_list(folds, repeats, listnames)
fit.lasso <- gen_m_list(folds, repeats, listnames)
fit.rf <- gen_m_list(folds, repeats, listnames)
fit.gbm <- gen_m_list(folds, repeats, listnames)
fit.svm <- gen_m_list(folds, repeats, listnames)
fit.rf_basic <- gen_m_list(folds, repeats, listnames)
fit.rf_nodisc <- gen_m_list(folds, repeats, listnames)
fit.rf_no_rep <- gen_m_list(folds, repeats, listnames)

#########################
# TUNING GRIDS
gbm.grid <- expand.grid(interaction.depth = 1:10, n.trees = 1001,
                        shrinkage = 0.1, n.minobsinnode = c(10))
rf.grid <- expand.grid(mtry = seq(3, 10, 1))
svm.grid <- expand.grid(sigma = 0.02, C = seq(0.25, 1.5, 0.25))
lassogrid <- expand.grid(alpha = 1, lambda = seq(0.005, 0.1, 0.001))

# ==============================================================================
# NESTED CV START

for (i in 1:length(outer_folds)) {
    end.time <- Sys.time()
    time.taken <- end.time - start.time
    if (verbose) print(paste0(round(time.taken[[1]], 2), " ", units(time.taken),
                              ": Class ", names(outer_folds)[i]))

    df.train <- model.frame(f_class, class_data[-outer_folds[[i]], ])
    x <- model.matrix(f_class, df.train)
    y <- df.train$replicated



    ##################
    # Reference model
    fit.ref_disc[[i]] <-
        run_class_model(model.matrix(replicated ~ 0 + discipline, df.train), y,
                        method = "glm", family = "binomial")

    ##################
    # Actual models
    fit.glm[[i]] <- run_class_model(x, y, method = "glm", family = "binomial")

    fit.lasso[[i]] <- run_class_model(x, y, method = "glmnet",
                                      family = "binomial", tuneGrid = lassogrid)

    fit.rf[[i]] <- run_class_model(x, y, method = "rf", ntree = 1001,
                                   tuneGrid = rf.grid, importance = TRUE)

    fit.gbm[[i]] <- run_class_model(x, y, method = "gbm", tuneGrid = gbm.grid,
                                    verbose = FALSE)

    fit.svm[[i]] <- run_class_model(x, y, method = "svmRadial",
                                    tuneGrid = svm.grid,
                                    preProc = c("center", "scale"))

    # Adding extra RF models to test variations
    fit.rf_basic[[i]] <- run_class_model(model.matrix(f_class_basic, df.train),
                                         y, method = "rf", ntree = 1001,
                                         tuneGrid = rf.grid, importance = TRUE)

    fit.rf_nodisc[[i]] <- run_class_model(model.matrix(update(f_class,
                                                              .~. - discipline),
                                                       df.train),
                                          y, method = "rf", ntree = 1001,
                                          tuneGrid = rf.grid, importance = TRUE)

    fit.rf_no_rep[[i]] <- run_class_model(model.matrix(f_class_no_rep,
                                                       df.train),
                                          y, method = "rf", ntree = 1001,
                                          tuneGrid = rf.grid, importance = TRUE)
}

# ==============================================================================
# Summarize results

df.test <- lapply(outer_folds,
                  function(index, data) model.frame(f_class, data[index, ]),
                  data = class_data)
x_test <- lapply(df.test, function(x) model.matrix(f_class, x))
y_test <- lapply(df.test, function(x) x$replicated)

class_results <- list(
    "Ref. Discipline Dummies" = gen_results(fit.ref_disc,
        lapply(df.test, function(x) model.matrix(replicated ~ 0 + discipline,
                                                 x)), y_test, my_class_summary),
    "LM (OLS or Logit)" = gen_results(fit.glm, x_test, y_test,
                                      my_class_summary),
    "LASSO" = gen_results(fit.lasso, x_test, y_test, my_class_summary),
    "Random Forest" = gen_results(fit.rf, x_test, y_test, my_class_summary),
    "GBM" = gen_results(fit.gbm, x_test, y_test, my_class_summary),
    "SVM" = gen_results(fit.svm, x_test, y_test, my_class_summary),
    "Only Basic Features" = gen_results(fit.rf_basic, lapply(df.test,
        function(x) model.matrix(f_class_basic, x)), y_test, my_class_summary),
    "No Discipline" = gen_results(fit.rf_nodisc, lapply(df.test,
        function(x) model.matrix(update(f_class, .~. - discipline), x)),
        y_test, my_class_summary),
    "No Replication Features" = gen_results(fit.rf_no_rep, lapply(df.test,
        function(x) model.matrix(f_class_no_rep, x)), y_test, my_class_summary)
)

# ==============================================================================
# Save image

save(list = c("outer_folds",
              "fit.ref_logit", "fit.ref_disc", "fit.glm", "fit.lasso",
              "fit.rf", "fit.gbm", "fit.svm", "fit.rf_basic", "fit.rf_nodisc",
              "fit.rf_no_rep"),
     file = "../data/nested_cv_class_models.RData")

save(list = c("class_results"),
     file = "../data/nested_cv_class_results.RData")

# Code is run in a local environment so we can just remove all
rm(list=ls())
