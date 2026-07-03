# This file prepares all the data needed.
stop("This file will overwrite data files. Please do not run if you are not sure about what you are doing.")

library(httr)
library(RefManageR)
options(stringsAsFactors=FALSE)


# GET RPP DATA FROM OSF
# osf_data <- GET("https://osf.io/fgjvw/?action=download",
#                 write_disk("../data/rpp/rpp_data.csv", overwrite = TRUE))
# Use new updated dataset instead
osf_data <- GET("https://osf.io/yt3gq/?action=download",
                write_disk("../data/rpp/rpp_data_updates.csv", overwrite = TRUE))

# Fetch bib references for RPP papers
# file is encoded in windows ISO-8859-1 encoding
data.rpp <- read.csv("../data/rpp/rpp_data_updates.csv", fileEncoding="ISO-8859-1")[1:167, ]
colnames(data.rpp)[1] <- "id"
data.rpp$id <- paste0("rpp.", data.rpp$id)

papers <- data.frame(bibtype="article", title=data.rpp$Study.Title..O.,
                     author=data.rpp$Authors..O., journal=data.rpp$Journal..O.,
                     year="2008")
rownames(papers) <- data.rpp$id
papers <- as.BibEntry(papers)
# too many queries overloads server, split in two
dois1 <- GetDOIs(papers[1:100])
dois2 <- GetDOIs(papers[101:167])
dois <- append(dois1, dois2)

# Fixing incorrect DOI
dois$doi[20] <- "10.1037/0278-7393.34.2.408"
dois$doi[28] <- "10.1037/0278-7393.34.1.249"
dois$doi[31] <- "10.1037/0278-7393.34.3.533"
dois$doi[40] <- "10.1037/0278-7393.34.2.415"
dois$doi[62] <- "10.1037/0022-3514.94.1.32"
dois$doi[104] <- "10.1111/j.1467-9280.2008.02062.x"
dois$doi[146] <- "10.1111/j.1467-9280.2008.02070.x"
dois$doi[160] <- "10.1111/j.1467-9280.2008.02090.x"

bib <- GetBibEntryWithDOI(dois$doi, temp.file = "../data/rpp/rpp.bib",
                          delete.file = FALSE)
# Using the new bibliographic info we create a csv with everything we need.
bibdata <- cbind(data.frame(id=data.rpp$id), as.data.frame(bib))

# Fix some missing data

write.csv(bibdata, "../data/rpp/rpp_bibdata.csv", row.names=FALSE,
          na="", fileEncoding="UTF-8")


######
# Experimental Econ
# We use already saved bibdata from project.
bib <- as.data.frame(ReadBib("../data/ee/exp_econ_papers.bib"))
bib$bibref <- row.names(bib)
bib <- merge(bib, read.csv("../data/ee/ee_bibrefs.csv"), by="bibref")
write.csv(bib, "../data/ee/ee_bibdata.csv", row.names=FALSE,
          na="", fileEncoding="UTF-8")

######
# ML1
data.ml1 <- read.csv("../data/ml1/ml1_data.csv", sep=";")[-c(7,9),]
papers <- data.frame(bibtype="article", title=data.ml1$title,
                     author=data.ml1$authors.o, journal=data.ml1$journal,
                     year="2008")
rownames(papers) <- data.ml1$id
papers <- as.BibEntry(papers)
dois <- GetDOIs(papers)
bib <- GetBibEntryWithDOI(dois$doi, temp.file = "../data/ml1/ml1.bib",
                          delete.file = FALSE)
bib <- cbind(data.frame(id=data.ml1$id), as.data.frame(bib))
# need to add 7 and 9 manually
bib <- rbind(bib, read.csv("../data/ml1/ml1_missing_bibs.csv"))
write.csv(bib, "../data/ml1/ml1_bibdata.csv", row.names=FALSE,
          na="", fileEncoding="UTF-8")

######
# ML2
dois <- read.csv("../data/ml2/ml2_original_papers.csv", sep=";")[1:31,]
bib <- GetBibEntryWithDOI(dois$doi, temp.file = "../data/ml2/ml2.bib",
                          delete.file = FALSE)
bib <- cbind(data.frame(id=dois$id), as.data.frame(bib))
bib <- rbind(bib, read.csv("../data/ml2/ml2_missing_bibs.csv"))
write.csv(bib, "../data/ml2/ml2_bibdata.csv", row.names=FALSE,
          na="", fileEncoding="UTF-8")

# ML2 data is currently very messy, need to fetch results to make usable
ml2_data <- readRDS("../data/ml2/Manylabs2/testoutput/ML2_results_global")$aggregated
ml2_names <- names(ml2_data)
library(dplyr)
ml2_data <- rbind_all(ml2_data)
ml2_data <- cbind(ml2_names[-which(ml2_names == "Zhong.1")], ml2_data)
names(ml2_data)[1] <- "ml2_id"
write.csv(ml2_data, "../data/ml2/ml2_replication_data.csv", row.names=FALSE,
          na="", fileEncoding="UTF-8")

######
# ML3
dois <- read.csv("../data/ml3/ml3_original_papers.csv", sep=";")
bib <- GetBibEntryWithDOI(dois$doi, temp.file = "../data/ml3/ml3.bib",
                          delete.file = FALSE)
bib <- cbind(data.frame(id=dois$id), as.data.frame(bib))
write.csv(bib, "../data/ml3/ml3_bibdata.csv", row.names=FALSE,
          na="", fileEncoding="UTF-8")
