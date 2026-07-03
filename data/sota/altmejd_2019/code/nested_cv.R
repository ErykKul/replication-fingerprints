#!/usr/bin/env Rscript

logfile_output <- file("nested_cv_output.log.txt", open = "wt")
sink(logfile_output, type = "output")
logfile_message <- file("nested_cv_message.log.txt", open = "wt")
sink(logfile_message, type = "message")

# OPTIONS
options(warn=1) # display warnings as they occur (save to log file)
verbose <- TRUE
folds <- 5 # Outer CV folds (i.e. train with 80%, test on 20%)
repeats <- 20 # Repeat the CV 20 times for a total of 100 resamples

# ==============================================================================
# Load packages and functions

start.time <- Sys.time()

# Load all packages directly to not get errors later
require(doMC)
registerDoMC(cores = detectCores()) # svmR has some trouble with this
options("mc.cores" = detectCores())
require(caret)
require(data.table)
require(plyr)
require(reshape)

# Machine Learning Models
require(glmnet)
require(randomForest)
require(gbm)
require(survival)
require(kernlab)
require(margins)
require(pROC)
require(ROCR)
require(ggplot2)
require(RColorBrewer)
require(pwr)

# Training functions, etc
source("functions.R")

# Prepare data set for CV
if (verbose) print(paste0("Started Nested CV process at: ", date()))

# ==============================================================================
# Run nested CV

time.taken <- Sys.time() - start.time
if (verbose) print(paste0(round(time.taken[[1]], 2), " ", units(time.taken),
                          ": Preparing data"))
source("nested_cv_prepare_data.R", local = new.env())

time.taken <- Sys.time() - start.time
if (verbose) print(paste0(round(time.taken[[1]], 2), " ", units(time.taken),
                          ": Starting Nested CV Class"))
source("nested_cv_class.R", local = new.env())

time.taken <- Sys.time() - start.time
if (verbose) print(paste0(round(time.taken[[1]], 2), " ", units(time.taken),
                          ": Starting Nested CV Reg"))
source("nested_cv_reg.R", local = new.env())

time.taken <- Sys.time() - start.time
if (verbose) print(paste0(round(time.taken[[1]], 2), " ", units(time.taken),
                          ": Starting Full Model Training"))
source("nested_cv_final_models.R", local = new.env())

# ==============================================================================
# END
if (verbose) print(paste0("Process ended at: ", date()))
