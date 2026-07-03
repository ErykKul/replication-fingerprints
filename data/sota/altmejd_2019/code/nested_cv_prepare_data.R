# This script loads the data and prepares it for training
# It creates the training formulas and the correct model matrices
# and keep them in memory
require(data.table)
source("functions.R", local = TRUE)

# ==============================================================================
# PREPARE DATA FOR MODEL FITTING
data <- readRDS("../data/data.rds")
# We only use the aggregated ML data
data <- data[aggregated == TRUE | is.na(aggregated)]
data <- data[drop == FALSE] # droplist
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
levels(data$replicated) <- c("not_replicated", "replicated")

# ==============================================================================
# Models
# ==============================================================================

# Remove post-replication variables
remove_cols(data, c("id", "effect_size.r", "p_value.r", "n.r", "power.r",
    "transactions", "trading_volume"))

class_data <- copy(data)
reg_data <- copy(data)

# Remove outcome varibles
class_data[, relative_es := NULL]
reg_data[, replicated := NULL]

# The Full Model
# no intercept as this is added by model later
f_class <- terms(replicated ~ 0 + ., data = class_data)
f_reg <- terms(relative_es ~ 0 + ., data = reg_data)
# Remove some features that do not work properly:
# - "endprice" because lots of missing
# - "seniority.o" becuase too little variation
# - "author_citations_avg.o/r" because only want to use author_citations_max
# - "planned_power.r" because we use "es_80power" instead
# - n_planned.r not needed together with es_80power
# - "project" because meta-variable (and captured by discipline)
# - "pub_year" because proxy for replicability
exclude_vars <- c("endprice", "seniority.o", "author_citations_avg.o",
    "author_citations_avg.r", "power_planned.r", "n_planned.r",
    "project", "pub_year")

f_class <- update(f_class,
    as.formula(paste(".~. - ", paste(exclude_vars, collapse = "-"))))
f_reg <- update(f_reg,
    as.formula(paste(".~. - ", paste(exclude_vars, collapse = "-"))))

basic_vars <- c("effect_size.o", "p_value.o")
f_class_basic <- as.formula(paste("replicated ~ 0 + ",
    paste(basic_vars, collapse = "+")))
f_reg_basic <- as.formula(paste("relative_es ~ 0 + ",
    paste(basic_vars, collapse = "+")))

rep_vars <- c("es_80power", "compensation.r", "n_authors.r",
    "seniority.r", "author_citations_max.r", "authors_male.r",
    "same_country", "same_language", "same_online", "same_subjects", "us_lab.r")
f_class_no_rep <- update(f_class,
    as.formula(paste(".~. - ", paste(rep_vars, collapse = "-"))))
f_reg_no_rep <- update(f_reg,
    as.formula(paste(".~. - ", paste(rep_vars, collapse = "-"))))

# Save datasets
save(list = c("class_data", "f_class", "f_class_basic", "f_class_no_rep"),
     file = "../data/nested_cv_class_data.RData")
save(list = c("reg_data", "f_reg", "f_reg_basic", "f_reg_no_rep"),
     file = "../data/nested_cv_reg_data.RData")

# Code is run in a local environment so we can just remove all
rm(list=ls())
