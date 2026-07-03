# Train the full model on the whole data set, save as .RData file
# Source the Nested CV models
# Meant to be run on AWS
source("functions.R", local = TRUE)
load("../data/nested_cv_class_data.RData")
load("../data/nested_cv_reg_data.RData")
set.seed(1234)

################################################################################
# Train new models on full data set

x_class <- model.matrix(f_class, model.frame(f_class, class_data))
y_class <- factor(model.frame(f_class, class_data)$replicated)
levels(y_class) <- c("not_replicated", "replicated")
colnames(x_class) <- make.names(colnames(x_class))

x_reg <- model.matrix(f_reg, model.frame(f_reg, reg_data))
y_reg <- model.frame(f_reg, reg_data)$relative_es
colnames(x_reg) <- make.names(colnames(x_reg))

# Classification
fit_class.full <- run_class_model(x_class, y_class, method = "rf",
                                     ntree = 1001, tuneLength = 15,
                                     importance = TRUE)

# Regression
fit_reg.full <- run_reg_model(x_reg, y_reg, method = "rf", ntree = 1001,
                                 tuneLength = 15, importance = TRUE)

# LASSO and LM models for varImp
# Glmnet standardizes all variables so we don't do it
lassogrid <- expand.grid(alpha = 1, lambda = seq(0.001, 0.2, 0.001))
fit_class.lasso <- run_class_model(x_class, y_class, method = "glmnet",
                                   family = "binomial", tuneGrid = lassogrid)
fit_reg.lasso <- run_reg_model(x_reg, y_reg, method = "glmnet",
                               family = "gaussian", tuneGrid = lassogrid)

x_class_lasso <- x_class[, coef(fit_class.lasso$finalModel,
    fit_class.lasso$bestTune$lambda)@i]
x_reg_lasso <- x_reg[, coef(fit_reg.lasso$finalModel,
    fit_reg.lasso$bestTune$lambda)@i]
# We then reestimate regular Logit with only regularized vars

fit_class.lasso_lm <- run_class_model(x_class_lasso, y_class, method = "glm",
                                      family = "binomial")$finalModel
fit_reg.lasso_lm <- run_reg_model(x_reg_lasso, y_reg, method = "glm",
                                  family = "gaussian")$finalModel

# And calculate marginal effects
# type = "response" to use scale of y-var
require(margins)
ame_class <- margins(fit_class.lasso_lm, as.data.frame(x_class_lasso),
                     type = "response")
ame_reg <- margins(fit_reg.lasso_lm, as.data.frame(x_reg_lasso),
                   type = "response")

################################################################################
# Save models
save(list = c("fit_class.full", "fit_reg.full",
              "fit_class.lasso", "fit_reg.lasso",
              "fit_class.lasso_lm", "fit_reg.lasso_lm",
              "ame_class", "ame_reg",
              "f_class", "f_reg"),
     file = "../data/full_models.RData")

# Code is run in a local environment so we can just remove all
rm(list=ls())
