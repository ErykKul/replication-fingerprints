# This script takes all the data files and merges them to one.
# Output
#     - "data.rds" full merged dataset to use in analysis
#     - "data.csv" full dataset in csv format
#     - "authors.csv" list of all authors in dataset to use with authordata

source("functions.R")
library(data.table)
library(pwr)
options(stringsAsFactors=FALSE)
sourcedir <- "../local/data_sources/"

###############################################################################
# DATA LOADING

data.rpp <- as.data.table(read.csv(paste0(sourcedir, "rpp/rpp_data_updates.csv"),
                                   fileEncoding="ISO-8859-1")[1:167, ])
colnames(data.rpp)[1] <- "id"
data.rpp$id <- paste0("rpp.", data.rpp$id)
data.rpp <- merge(data.rpp, read.csv(paste0(sourcedir, "rpp/rpp_extra_data.csv"), sep = ","),
                  by = "id", all=TRUE)

data.ee <- read.csv(paste0(sourcedir, "ee/studydetails.csv"))
data.ee <- as.data.table(cbind(data.frame(id=paste0("ee.", 1:nrow(data.ee))),
                     data.ee))
data.ee <- merge(data.ee, read.csv(paste0(sourcedir, "ee/ee_extra_data.csv"), sep = ";"), by = "id")

data.ml1 <- as.data.table(read.csv(paste0(sourcedir, "ml1/ml1_data.csv"), sep = ";"))
# data.ml2 <- as.data.table(read.csv(paste0(sourcedir, "ml2/ml2_data.csv"), sep = ";"))
# data.ml2 <- merge(data.ml2, read.csv(paste0(sourcedir, "ml2/ml2_replication_data.csv")),
#                   by="ml2_id", all.x=TRUE)
data.ml3 <- as.data.table(read.csv(paste0(sourcedir, "ml3/ml3_data.csv"), sep = ";"))

#########################
## Prediction Market data (when available)
data.rpp <- merge(data.rpp, read.csv(paste0(sourcedir, "rpp/rpp_market_data.csv")),
                  by = "id", all=TRUE)

market_data.ee <- read.csv(paste0(sourcedir, "ee/marketsurveysummary.csv"))
# just take the first row for each study (throw away trader data)
market_data.ee <- market_data.ee[!duplicated(market_data.ee$study), ]
data.ee <- merge(data.ee, market_data.ee, by = "study", sort = FALSE,
                 suffixes = c("", ".mkt"))
market_data.ml2 <- read.csv(paste0(sourcedir, "ml2/trades_report.csv"))
survey_data.ml2 <- read.csv(paste0(sourcedir, "ml2/survey.csv"))

#########################
# BIBLIOGRAPHY DATA
bib.rpp <- data.table(read.csv(paste0(sourcedir, "rpp/rpp_bibdata.csv")))
bib.ee <- data.table(read.csv(paste0(sourcedir, "ee/ee_bibdata.csv")))
bib.ml1 <- data.table(read.csv(paste0(sourcedir, "ml1/ml1_bibdata.csv")))
bib.ml2 <- data.table(read.csv(paste0(sourcedir, "ml2/ml2_bibdata.csv")))
bib.ml3 <- data.table(read.csv(paste0(sourcedir, "ml3/ml3_bibdata.csv")))

###############################################################################
# DATA CLEANING

#########################
# Merge cleaned bibliography data
data.rpp <- merge(data.rpp, bib_clean(bib.rpp),
                  by = "id", suffixes = c("", ".bib"))
data.ee <- merge(data.ee, bib_clean(bib.ee),
                  by = "id", suffixes = c("", ".bib"))
data.ml1 <- merge(data.ml1, bib_clean(bib.ml1),
                  by = "id", suffixes = c("", ".bib"), all = TRUE)
# data.ml2 <- merge(data.ml2, bib_clean(bib.ml2),
#                   by = "id", suffixes = c("", ".bib"), all = TRUE)
data.ml3 <- merge(data.ml3, bib_clean(bib.ml3),
                  by = "id", suffixes = c("", ".bib"), all = TRUE)

# Fix utf-8 error
data.rpp["rpp.145"]$title <- '"In-group love" and "out-group hate" as motives for individual participation in intergroup conflict: a new game paradigm'

# 67 of the studies in RPP were never actually done - we exclude those
data.rpp <- data.rpp[!is.na(T_sign_O) & !is.na(T_sign_R),]
# 3 studies had significant original results, not really comparable so dropped already here
data.rpp <- data.rpp[T_sign_O != 0]

# Add all studies that will not be used to droplist variable to be excluded later
# If study has insignificant original result we drop it (3 studies)
# 3 studies excluded because of missing ES data
droplist <- c(data.rpp[is.na(T_r..O.) | is.na(T_r..R.), id])

# RPP.77 is problematic: email is sent out to 768703 but only 15 respond
# the replication authors themselves claim replication is not trustworty
# data.rpp["rpp.77"]$Planned.Sample <- 303
# data.rpp["rpp.77"]$N..R. <- 768703
# "rpp.46" is not really an experiment, we drop it
# same for "rpp.73" -- observational study
# "rpp.82" is a correlational study with no treatment
# correlation between oblivious student assessed power level of CEO and company performance
# "rpp.154" is a correlational cross-country study using available data
droplist <- c(droplist, c("rpp.77", "rpp.46", "rpp.73", "rpp.82", "rpp.154"))
# a number of prediction market prices were traded on the wrong hypothesis
data.rpp[id %in% c("rpp.103", "rpp.104", "rpp.108"), endprice := NA]

#########################
# Dependent variables

###
# Effect sizes and P-values
# Many labs projects do not report r-effect sizes. So we need to transform.

###
# ML1

# Effect size of ml1.12 and 14 are reported with the wrong sign
data.ml1["ml1.14"]$statistic.o <- data.ml1["ml1.14"]$statistic.o * -1
data.ml1["ml1.12"]$statistic.o <- data.ml1["ml1.12"]$statistic.o * -1
data.ml1$effect_size.r <- with(data.ml1, mapply(get_es, statistic.r, test.r, df, n.r))
data.ml1$effect_size.o <- with(data.ml1, mapply(get_es, statistic.o, test.o, param1.o, param2.o))
data.ml1$p_value.o <- with(data.ml1, mapply(get_p, statistic.o, test.o, param1.o, param2.o))
data.ml1$p_value.r <- with(data.ml1, mapply(get_p, statistic.r, test.r, df, n.r))

###
# # ML2
# data.ml2$test.type <- gsub(data.ml2$test.type, pattern="X2", replace="chi2")
# data.ml2$test.type <- gsub(data.ml2$test.type, pattern="lm.Z", replace="z")
# data.ml2$test.parameter[33] <- data.ml2$test.parameter1[33]
# # Something seems very strange with p-value data for ml2.8 (nr 33).
# # data.ml2$test.p.value[33] is different from my calculations
# # Does not work (statistic.o not available!)
# #data.ml2$effect_size.o <- with(data.ml2, mapply(get_es, statistic.o, test.o, param1.o, param2.o))
# data.ml2$effect_size.r <- with(data.ml2, mapply(get_es, test.statistic, test.type, test.parameter, test.parameter2))
# # data.ml2$p_value.o <- with(data.ml2, mapply(get_p, statistic.o, test.o, param1.o, param2.o))
# data.ml2$p_value.r <- with(data.ml2, mapply(get_p, test.statistic, test.type, test.parameter, test.parameter2))

###
# ML3
data.ml3$effect_size.o <- with(data.ml3, mapply(get_es, statistic.o, test.o, param1.o, param2.o))
data.ml3$effect_size.r <- with(data.ml3, mapply(get_es, statistic.r, test.r, param1.r, param2.r))
data.ml3$p_value.o <- with(data.ml3, mapply(get_p, statistic.o, test.o, param1.o, param2.o))
data.ml3$p_value.r <- with(data.ml3, mapply(get_p, statistic.r, test.r, param1.r, param2.r))

###
# Add replication status variable
data.ee$significant.o <- ifelse(data.ee$porig <= 0.05, 1, 0)
data.ee$significant.r <- ifelse(data.ee$prep <= 0.05, 1, 0)
# ee.2 has p=0.057 but is considered significant
# ee.18 has p=0.07 but is considered significant
data.ee[id %in% c("ee.2", "ee.18"), significant.o := 1]

data.ml1$significant.o <- ifelse(data.ml1$p_value.o <= 0.05, 1, 0)
data.ml1$significant.r <- ifelse(data.ml1$p_value.r <= 0.05, 1, 0)
# ml1.1 has p=0.09935133 but is considered significant
data.ml1[id=="ml1.1", significant.o := 1]
# ml1.9, ml1.10, ml1.11 don't report p-values but are considered significant
data.ml1[id=="ml1.9", significant.o := 1]
data.ml1[id=="ml1.10", significant.o := 1]
data.ml1[id=="ml1.11", significant.o := 1]

# data.ml2$significant.o <- ifelse(data.ml2$p_value.o <= 0.05, 1, 0)
# data.ml2$significant.r <- ifelse(data.ml2$p_value.r <= 0.05, 1, 0)

data.ml3$significant.o <- ifelse(data.ml3$p_value.o <= 0.05, 1, 0)
data.ml3$significant.r <- ifelse(data.ml3$p_value.r <= 0.05, 1, 0)

# An original insignificant effect should not be counted as a repociation
data.ml1$replicated <- ifelse(data.ml1$significant.o == 0, NA, data.ml1$significant.r)
# data.ml2$replicated <- ifelse(data.ml2$significant.o == 0, NA, data.ml2$significant.r)
data.ml3$replicated <- ifelse(data.ml3$significant.o == 0, NA, data.ml3$significant.r)

###############################################################################
# DATA MERGING
# Create one dataset holding all relevant features from all 5 studies.

## RPP
data <- data.table(id = data.rpp$id,
                   title = data.rpp$Study.Title..O.,
                   authors.o = data.rpp$author,
                   pub_year = data.rpp$year,
                   journal = data.rpp$Journal..O.,
                   volume = data.rpp$Volume..O.,
                   issue = data.rpp$Issue..O.,
                   pages = data.rpp$pages,
                   discipline = data.rpp$Discipline..O.,
                   length = NA_integer_,
                   citations = data.rpp$Citation.count..paper..O.,
                   effect_size.o = data.rpp$T_r..O.,
                   p_value.o = data.rpp$T_pval_USE..O.,
                   n.o = data.rpp$T_N_O_for_tables, # df/N instead #n.o = data.rpp$N..O.,
                   effect_type = data.rpp$Type.of.effect..O., # interaction/main/etc
                   effect_size.r = data.rpp$T_r..R.,
                   p_value.r = data.rpp$T_pval_USE..R.,
                   n_planned.r = data.rpp$Planned.Sample, # not available for most data
                   n.r = data.rpp$T_N_R_for_tables, # df/N instead # n.r = data.rpp$N..R.,
                   power.o = NA_real_,
                   power.r = data.rpp$Power..R.,
                   power_planned.r = data.rpp$Planned.Power,
                   experiment_country.o = data.rpp$experiment_country.o,
                   experiment_country.r = data.rpp$experiment_country.r,
                   experiment_language.o = data.rpp$experiment_language.o,
                   experiment_language.r = data.rpp$experiment_language.r,
                   online.o = as.integer(data.rpp$online.o),
                   online.r = as.integer(data.rpp$online.r),
                   compensation.o = data.rpp$incentive.o,
                   compensation.r = data.rpp$incentive.r,
                   subjects.o = data.rpp$subjects.o,
                   subjects.r = data.rpp$subjects.r,
                   endprice = data.rpp$endprice/100,
                   transactions = data.rpp$transactions,
                   trading_volume = data.rpp$volume,
                   significant.o = data.rpp$T_sign_O,
                   significant.r = data.rpp$T_sign_R,
                   replicated = ifelse(data.rpp$T_sign_O == 0, NA, data.rpp$T_sign_R)
                   )

## EXP ECON
data <- rbind(data,
              data.table(id = data.ee$id,
                         title = data.ee$title,
                         authors.o = data.ee$author,
                         pub_year = data.ee$year,
                         journal = data.ee$journal,
                         volume = data.ee$volume.bib,
                         issue = data.ee$number,
                         pages = data.ee$pages,
                         discipline = "Economics",
                         length = NA_integer_,
                         citations = data.ee$citations,
                         effect_size.o = data.ee$eorig,
                         p_value.o = data.ee$porig,
                         n.o = data.ee$norig,
                         effect_type = data.ee$effect_type,
                         effect_size.r = data.ee$erep,
                         p_value.r = data.ee$prep,
                         n_planned.r = data.ee$nrep_plan,
                         n.r = data.ee$nrep_act,
                         power.o = NA_real_,
                         power.r = data.ee$powrep_act,
                         power_planned.r = data.ee$powrep_plan,
                         experiment_country.o = data.ee$experiment_country.o,
                         experiment_country.r = data.ee$experiment_country.r,
                         experiment_language.o = data.ee$experiment_language.o,
                         experiment_language.r = data.ee$experiment_language.r,
                         online.o = as.integer(data.ee$online.o),
                         online.r = as.integer(data.ee$online.r),
                         compensation.o = data.ee$incentive.o,
                         compensation.r = data.ee$incentive.r,
                         subjects.o = data.ee$subjects.o,
                         subjects.r = data.ee$subjects.r,
                         endprice = data.ee$endprice,
                         transactions = data.ee$transactions,
                         trading_volume = data.ee$volume,
                         significant.o = data.ee$significant.o,
                         significant.r = data.ee$significant.r,
                         replicated = data.ee$result
                         ), fill = TRUE, use.names = TRUE)

## Many Labs 1
data <- rbind(data,
              data.table(id = data.ml1$id,
                         title = data.ml1$title.bib,
                         authors.o = data.ml1$author,
                         pub_year = data.ml1$year,
                         journal = data.ml1$journal.bib,
                         volume = data.ml1$volume.bib,
                         issue = data.ml1$number,
                         pages = data.ml1$pages.bib,
                         discipline = data.ml1$discipline,
                         length = NA_integer_,
                         citations = data.ml1$citations,
                         effect_size.o = data.ml1$effect_size.o,
                         p_value.o = data.ml1$p_value.o,
                         n.o = data.ml1$n.o,
                         effect_type = data.ml1$effect_type,
                         effect_size.r = data.ml1$effect_size.r,
                         p_value.r = data.ml1$p_value.r,
                         n_planned.r = NA_real_,
                         n.r = data.ml1$n.r,
                         power.o = NA_real_,
                         power.r = NA_real_,
                         power_planned.r = NA,
                         significant.o = data.ml1$significant.o,
                         significant.r = data.ml1$significant.r,
                         replicated = data.ml1$replicated,
                         experiment_country.o = data.ml1$experiment_country.o,
                         experiment_language.o = data.ml1$experiment_language.o,
                         online.o = data.ml1$online.o,
                         compensation.o = data.ml1$compensation.o,
                         subjects.o = data.ml1$subjects.o
                         ), fill = TRUE, use.names = TRUE)

## Many Labs 2
# data <- rbind(data,
#               data.table(id = data.ml2$id,
#                          title = data.ml2$title,
#                          authors.o = data.ml2$author,
#                          pub_year = data.ml2$year,
#                          journal = data.ml2$journal,
#                          volume = data.ml2$volume,
#                          issue = data.ml2$number,
#                          pages = data.ml2$pages,
#                          discipline = data.ml2$discipline,
#                          length = NA_integer_,
#                          citations = data.ml2$citations,
#                          effect_size.o = data.ml2$effect_size.o,
#                          p_value.o = data.ml2$p_value.o,
#                          n.o = data.ml2$n.o,
#                          effect_type = NA,
#                          effect_size.r = NA,
#                          p_value.r = data.ml2$p_value.r,
#                          n_planned.r = NA_real_,
#                          n.r = data.ml2$stat.N,
#                          power.o = NA_real_,
#                          power.r = NA_real_,
#                          power_planned.r = NA,
#                          endprice = NA,
#                          transactions = NA,
#                          trading_volume = NA,
#                          significant.o = data.ml2$significant.o,
#                          significant.r = data.ml2$significant.r,
#                          replicated = data.ml2$replicated,
#                          experiment_country.o = NA,
#                          experiment_country.r = NA, # need to get this from lab level (see below)
#                          experiment_language.o = NA,
#                          experiment_language.r = NA, # need to get this from lab level (see below)
#                          online.o = NA_integer_,
#                          online.r = NA_integer_, # need to get this from lab level (see below)
#                          compensation.o = NA,
#                          compensation.r = NA, # need to get this from lab level (see below)
#                          subjects.o = NA,
#                          subjects.r = NA # need to get this from lab level (see below)
#                          ), fill = TRUE, use.names = TRUE)

## Many Labs 3
data <- rbind(data,
              data.table(id = data.ml3$id,
                         title = data.ml3$title,
                         authors.o = data.ml3$author,
                         pub_year = data.ml3$year,
                         journal = data.ml3$journal,
                         volume = data.ml3$volume,
                         issue = data.ml3$number,
                         pages = data.ml3$pages,
                         discipline = data.ml3$discipline,
                         length = NA_integer_,
                         citations = data.ml3$citations,
                         effect_size.o = data.ml3$effect_size.o,
                         p_value.o = data.ml3$p_value.o,
                         n.o = data.ml3$n.o,
                         effect_type = data.ml3$effect_type,
                         effect_size.r = data.ml3$effect_size.r,
                         p_value.r = data.ml3$p_value.r,
                         n_planned.r = data.ml3$n_planned.r,
                         n.r = data.ml3$n_actual.r,
                         power.o = NA_real_,
                         power.r = NA_real_,
                         power_planned.r = NA,
                         significant.o = data.ml3$significant.o,
                         significant.r = data.ml3$significant.r,
                         replicated = data.ml3$replicated,
                         experiment_country.o = data.ml3$experiment_country.o,
                         experiment_language.o = data.ml3$experiment_language.o,
                         online.o = data.ml3$online.o,
                         compensation.o = data.ml3$compensation.o,
                         subjects.o = data.ml3$subjects.o
                         ), fill = TRUE, use.names = TRUE)

data$pub_year <- as.numeric(data$pub_year)
data$citations <- as.numeric(data$citations)
data$discipline <- as.factor(data$discipline)
data$effect_size.o <- as.numeric(data$effect_size.o)
data$effect_size.r <- as.numeric(data$effect_size.r)
data$n.o <- as.numeric(data$n.o)
data$n.r <- as.numeric(data$n.r)
data$effect_size.o <- as.numeric(data$effect_size.o)
data$effect_size.r <- as.numeric(data$effect_size.r)
data$p_value.o <- as.numeric(data$p_value.o)
data$p_value.r <- as.numeric(data$p_value.r)
data$power_planned.r <- suppressWarnings(as.numeric(data$power_planned.r)) # NAs introduced
data$power.r <- suppressWarnings(as.numeric(data$power.r)) # NAs introduced
# Data table keyed by id
setkey(data, id)

################################################################################
# GENERATE VARIABLES
# Generate some extra variables that are the same for all 5 projects

# project dummy variable
data$project <- as.factor(gsub(data$id, pattern="\\..*$", replace=""))

# add length variable using regular expression matching pages style "123--345"
# add 1 because both first and last page should be included
data$length <- 1 + as.numeric(gsub(data$pages, pattern="^\\d*[-–]{1,2}",
                                   replace="")) - as.numeric(gsub(data$pages,
                                   pattern="[-–]{1,2}\\d*$", replace=""))

# Relative effect size (before sign corrections!)
data$relative_es <- data$effect_size.r / data$effect_size.o

###############################################################################
# Load author data and match

# authors.csv are keyed by author-study pairs and can be used for
# matching. One author can of course be on many studies
# We use 'author_order' (when available) here to make sure that the authors are
# always in the correct order
authors <- data.table(read.csv("../data/authors.csv"))
setkey(authors, study_id, study_type, author_order)

# `author_data.csv` includes info information on all authors (citations etc)
author_data <- data.table(read.csv("../data/author_data.csv", sep = ";",
                                   fileEncoding = "macroman"))
setkey(author_data, author_id) # one row per author

# Add variables for number of authors
n_authors.o <- unique(authors[study_type=="o", .(study_id, n_authors)])
names(n_authors.o) <- c("id", "n_authors.o")
n_authors.r <- unique(authors[study_type=="r", .(study_id, n_authors)])
names(n_authors.r) <- c("id", "n_authors.r")
data <- merge(data, merge(n_authors.o, n_authors.r, by="id"), by="id")

# For each study, find the relevant authors
for (id in data$id) {

    # loop over replication and original studies
    for (type in c("o", "r")) {

        study_authors <- authors[study_id == id & study_type == type, author_id]
        if (length(study_authors) > 0 & study_authors[1] != "") {
            # Instead of saving individual author info in the data file do the aggregation directly

            # Average citations
            data[id, paste0("author_citations_avg.", type) :=
                 mean(author_data[study_authors, citations], na.rm=TRUE)]

            # Citations of most cited author
            data[id, paste0("author_citations_max.", type) :=
                 max(author_data[study_authors, citations])]

            # Gender ratio (percent male)
            data[id, paste0("authors_male.", type) :=
                 mean(author_data[study_authors, gender] == "M", na.rm=TRUE)]

            # Highest seniority (highest seniority on project)
            seniority_order <- c("Professor", "Associate Professor",
                                 "Assistant Professor", "Researcher",
                                 "Lecturer", "Assistant", "Other")
            data[id, paste0("seniority.", type) :=
                 seniority_order[seniority_order %in%
                                 author_data[study_authors, position]][1]]
        }
    }
}


###############################################################################
# GENERATE LAB LEVEL DATA (ML1 and ML3)

# Many Labs aggregated data marked as true (but EE and RPP are NA because no aggregation)
data$aggregated <- as.logical(NA)
data$aggregated[data$project %in% c("ml1", "ml2", "ml3")] <- TRUE

# Load per lab data
lab_data.ml1 <- data.table(read.csv(paste0(sourcedir, "ml1/ml1_replication_data_per_lab.csv"), sep = ";", fileEncoding = "macroman"))
lab_data.ml3 <- data.table(read.csv(paste0(sourcedir, "ml3/ml3_replication_data_per_lab.csv"), sep = ";", fileEncoding = "macroman"))

# fix es
lab_data.ml1$r[is.na(lab_data.ml1$r)] <- with(lab_data.ml1, mapply(get_es, statistic, test, df, n))[is.na(lab_data.ml1$r)]
lab_data.ml3$r[is.na(lab_data.ml3$r)] <- with(lab_data.ml3, mapply(get_es, statistic, test, df, n))[is.na(lab_data.ml3$r)]

# fix p values
lab_data.ml1$p <- with(lab_data.ml1, mapply(get_p, statistic, test, df, n))
lab_data.ml3$p[is.na(lab_data.ml3$p)] <- with(lab_data.ml3, mapply(get_p, statistic, test, df, n))[is.na(lab_data.ml3$p)]

# create one data table for merging with all disaggregated data
cols <- c("study", "lab_code", "country", "language", "online", "incentive",
          "subjects", "n", "r", "p")
lab_data <- data.table(rbind(lab_data.ml1[, cols, with = FALSE],
                             lab_data.ml3[, cols, with = FALSE]))
names(lab_data)[1:2] <- c("id", "lab_id")
setkey(lab_data, id, lab_id)
# merge() creates dataset with only disaggregated entries for ml1 and ml3
# we then re add the old aggregated ML1/3 by rbind
data <- rbind(merge(data, lab_data, all = TRUE),
              data[project %in% c("ml1", "ml3"),], fill = TRUE)
data[!is.na(lab_id), aggregated := FALSE]
setkey(data, id, lab_id)

# add data to existing columns instead of keeping in new cols
data[aggregated == FALSE, effect_size.r := r]
data[aggregated == FALSE, p_value.r := p]
data[aggregated == FALSE, n.r := as.numeric(n)]
data[aggregated == FALSE, n_planned.r := NA_real_] # we dont have planned n for any DA data
data[aggregated == FALSE, relative_es := effect_size.r / effect_size.o]
data[aggregated == FALSE, significant.r := ifelse(p <= 0.05, 1, 0)]
data[aggregated == FALSE & significant.o == 1, replicated := ifelse(significant.r == 1 & relative_es > 0, 1, 0)]
data[aggregated == FALSE & significant.o != 1, replicated := NA_integer_]
data[aggregated == FALSE, experiment_country.r := country]
data[aggregated == FALSE, experiment_language.r := language]
data[aggregated == FALSE, online.r := as.integer(online)]
data[aggregated == FALSE, compensation.r := incentive]
data[aggregated == FALSE, subjects.r := subjects]

data[, c("country", "language", "online", "incentive", "subjects", "n", "r", "p") := NULL]

# get correct lab authors
for (i in 1:nrow(data[aggregated == FALSE,])) {
    study <- data[aggregated==FALSE, id][i]
    lab <- data[aggregated==FALSE, lab_id][i]
    project <- data[aggregated==FALSE, project][i]

    # first fetch all authors for study (not per lab)
    study_authors <- authors[study_id == study & study_type == "r", author_id]
    # from these find lab authors
    lab_authors <- author_data[author_id %in% study_authors &
                               get(paste0(toupper(project), "_lab")) == lab,
                               author_id]

    if (length(lab_authors) > 0 & lab_authors[1] != "") {
        # N authors
        data[id == study & lab_id == lab, n_authors.r := length(lab_authors)]

        # Average citations
        data[id == study & lab_id == lab,
             author_citations_avg.r :=
             mean(author_data[lab_authors, citations], na.rm = TRUE)]

        # Citations of most cited author
        data[id == study & lab_id == lab,
             author_citations_max.r :=
             max(author_data[lab_authors, citations])]

        # Gender ratio (percent male)
        data[id == study & lab_id == lab,
             authors_male.r :=
             mean(author_data[lab_authors, gender] == "M", na.rm = TRUE)]
    }
}

###############################################################################
# Calculate and fix some variables

# Instead of using different power calculations we recalculate power for all
# Original power is just power given n.o and effect size
# Planned power is based on original effect size and planned n
# Impute planned n from replication n when not available
data[is.na(n_planned.r), n_planned.r := n.r]

for (i in 1:nrow(data)) {
    if(!is.na(data[i, n.o]) & !is.na(data[i, effect_size.o])) {
        set(data, i = i, j = "power.o",
            value = pwr.r.test(n = data[i, n.o], r = data[i, effect_size.o],
                               sig.level = 0.05, power = NULL)$power)
    }

    if(!is.na(data[i, n_planned.r]) & !is.na(data[i, effect_size.o])) {
        set(data, i = i, j = "power_planned.r",
            value = pwr.r.test(n = data[i, n_planned.r],
                               r = data[i, effect_size.o],
                               sig.level = 0.05, power = NULL)$power)
    }

    if(!is.na(data[i, n.r]) & !is.na(data[i, effect_size.r])) {
        set(data, i = i, j = "power.r",
            value = pwr.r.test(n = data[i, n.r],
                               r = data[i, effect_size.r],
                               sig.level = 0.05, power = NULL)$power)
    }

    if(!is.na(data[i, n_planned.r])) {
        # We also calculate an alternative metric to use instead of planned power:
        # The effect size required to have 80% power given the planned sample size
        set(data, i = i, j = "es_80power",
            value = pwr.r.test(n = data[i, n_planned.r],
                               r = NULL, power = 0.8, sig.level = 0.05)$r)
    }
}

# From lab level ML data we can calculate the most common (mode) characteristic
# to use for aggregates
# Exclude ML2 since its aggregated but we dont have the lab-level data yet
data[aggregated == TRUE & project != "ml2", online.r :=
     data[aggregated == FALSE, .(Mode(online.r)), by=id][unique(id)]$V1]
data[aggregated == TRUE & project != "ml2", experiment_country.r :=
     data[aggregated == FALSE, .(Mode(experiment_country.r)), by=id][unique(id)]$V1]
data[aggregated == TRUE & project != "ml2", experiment_language.r :=
     data[aggregated == FALSE, .(Mode(experiment_language.r)), by=id][unique(id)]$V1]
data[aggregated == TRUE & project != "ml2", compensation.r :=
     data[aggregated == FALSE, .(Mode(compensation.r)), by=id][unique(id)]$V1]
data[aggregated == TRUE & project != "ml2", subjects.r :=
     data[aggregated == FALSE, .(Mode(subjects.r)), by=id][unique(id)]$V1]

# Same country and language
data[, same_country := ifelse(experiment_country.o == experiment_country.r, 1, 0)]
data[, same_language := ifelse(experiment_language.o == experiment_language.r, 1, 0)]
data[, same_online := ifelse(online.o == online.r, 1, 0)]
data[, same_subjects := ifelse(subjects.o == subjects.r, 1, 0)]

# US original/replication
data[, us_lab.o := ifelse(experiment_country.o == "United States", 1, 0)]
data[, us_lab.r := ifelse(experiment_country.r == "United States", 1, 0)]

# Effect types in RPP are in too many categories
data[effect_type %in% c("focused interaction contrast", "contrast"),
         effect_type := "interaction"]
data[effect_type %in% c("binomial test", "regression", "trend"), effect_type := "main effect"]

# Simplify categories for subject types
data[subjects.o %in% c("students+community"), subjects.o := "community"]
data[subjects.r %in% c("students+community"), subjects.r := "community"]
data[subjects.o %in% c("special"), subjects.o := "anyone"]
data[subjects.r %in% c("special"), subjects.r := "anyone"]

###############################################################################
# Data integrity checks

# Stop if some is marked as replicated but significant in wrong direction
if (any(data$relative_es < 0 & data$p_value.r <= 0.05 &
        data$replicated == 1, na.rm = TRUE)) {
    stop(paste("Rows", paste0(which(data$relative_es < 0 & data$p_value.r <= 0.05 &
                                    data$replicated == 1),
                              collapse=", "),
               "are marked as significant but go in wrong direction."))
}
if (any(data$significant.o == 0 & data$replicated == 1, na.rm = TRUE)) {
    stop(paste("Rows", paste0(which(data$significant.o == 0 & data$replicated == 1), collapse=", "),
               "have insignificant original results but are counted as replicated"))
}

###############################################################################
# Save data as csv for easy viewing and RDS for use in analysis

# Drop columns that are not useful
# we use "replicated" rather than significance dummies
data[, c("significant.r", "significant.o") := NULL]
# drop pages (we use length)
data[, pages := NULL]

# We are missing effect sizes and p-values for 3 ML1 studies
droplist <- append(droplist, c("ml1.9", "ml1.10", "ml1.11"))
data[, drop := FALSE]
data[id %in% droplist, drop := TRUE]

write.csv(data, "../data/data.csv", row.names = FALSE, na = "",
          fileEncoding = "UTF-8")
saveRDS(data, "../data/data.rds")
