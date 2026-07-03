###############################################################################
#                                                                             #
# The purpose of this script is to create an "authors.csv" file which         #
# contains information to match authors with studies.                         #
#                                                                             #
# If the files do not already exist, this script creates "authors_o.csv"      #
# and "authors_r.csv" files using the author data that is already available   #
# from the studies. These files are used to match study_id and author_id      #
#                                                                             #
# IMPORTANT: Do not change author id's for any reason!                        #
#                                                                             #
# When the dataset is completely set up you need the following files to run   #
# this file:
#
#                                                                             #
###############################################################################

if(!file.exists("../data/data.rds")) { stop("Could not find 'data.rds'.")}
if(file.exists("../data/authors.csv")) { stop("Error: authors.csv already exists. Overwrite disabled.") }
library(data.table)
options(stringsAsFactors=FALSE)
sourcedir <- "../local/data_sources/"

###############################################################################
# Original authors

generate_author_o <- function() {
    data <- readRDS("../data/data.rds")
    authors <- data.table()

    # for each study, fetch all authors and create ID's
    for (i in data$id) {
        newauthors <- unlist(strsplit(data[i]$authors.o, " and ", fixed=TRUE))
        first <- c(TRUE, rep(FALSE, length(newauthors) - 1))
        last <- c(rep(FALSE, length(newauthors) - 1), TRUE)
        newauthors <- data.table(study_id = i,
                                 study_type = "o",
                                 author_id = NA_integer_,
                                 full_name = newauthors,
                                 first_name = NA,
                                 last_name = NA,
                                 author_order = 1:length(newauthors),
                                 first_author = first,
                                 last_author = last,
                                 n_authors = data[i]$n_authors.o
                                 )
        authors <- rbind(authors, newauthors)
    }

    ###
    # Manually fix duplicate with slight spelling differences
    authors[full_name == "D Kahneman"]$full_name <- "Daniel Kahneman"
    authors[full_name == "D. Kahneman"]$full_name <- "Daniel Kahneman"
    authors[full_name == "A. D. Galinsky"]$full_name <- "Adam D. Galinsky"
    authors[full_name == "J. Knobe"]$full_name <- "Joshua Knobe"
    authors[full_name == "Chris Mitchell"]$full_name <- "Christopher Mitchell"
    authors[full_name == "A Tversky"]$full_name <- "Amos Tversky"
    authors[full_name == "C. K. Hsee"]$full_name <- "Christopher K. Hsee"

    ###
    # Populate first/last name columns
    authors$first_name <- gsub(authors$full_name, pattern="^(.*)\\s.*$",
                               replacement="\\1")
    authors$last_name <- gsub(authors$full_name, pattern=".*\\s(.*$)",
                              replacement="\\1")

    # drop full name column (id matching possible by study_id)
    authors[, full_name := NULL]

    ###########################################################################

    ###
    # Add author IDs, unique to each author

    # Order authors accoring to last name to make it easier to find duplicates
    setkey(authors, last_name, first_name)
    # http://stackoverflow.com/questions/13018696/data-table-key-indices-or-group-counter
    authors[, author_id := .GRP, by = key(authors)]
    authors[, author_id := paste0("author.", author_id)]


    ###########################################################################
    # Add RPP study info about authors

    data.rpp <- read.csv(paste0(sourcedir, "rpp/rpp_data_updates.csv"),
                         fileEncoding="ISO-8859-1")[1:167, ]
    colnames(data.rpp)[1] <- "id"
    data.rpp$id <- paste0("rpp.", data.rpp$id)

    for (i in 1:nrow(data.rpp)) {
        authors[study_id == data.rpp$id[i] & first_author == TRUE,
                c("first_author_name", "first_author_citations",
                  "first_author_institution") := list(
                  data.rpp[i, "X1st.author..O."],
                  data.rpp[i, "Citation.Count..1st.author..O."],
                  data.rpp[i, "Institution.1st.author..O."])
                  ]
        authors[study_id == data.rpp$id[i] & last_author == TRUE,
                c("last_author_name", "last_author_citations",
                  "last_author_institution") := list(
                  data.rpp[i, "Senior.author..O."],
                  data.rpp[i, "Citation.count..senior.author..O."],
                  data.rpp[i, "Institution.senior.author..O."])
                  ]
    }
    return(authors)
}

###############################################################################
# Replication authors

generate_author_r <- function() {
    data <- readRDS("../data/data.rds")
    authors <- data.table()

    # Load RPP (data has recorded replication authors)
    data.rpp <- as.data.table(read.csv(paste0(sourcedir,
                                              "rpp/rpp_data_updates.csv"),
                                       fileEncoding="ISO-8859-1")[1:167, ])
    colnames(data.rpp)[1] <- "id"
    data.rpp$id <- paste0("rpp.", data.rpp$id)
    setkey(data.rpp, id)

    # We can scrape the RPP author names
    osf_scrape <- function(url) {
        library(rvest)
        authors <- read_html(url) %>%
                   html_nodes("#contributorsList ol li") %>%
                   html_text() %>% gsub(pattern="^\\n\\s*\\n\\s*", replace="", .)
        return(authors)
    }

    # Loop through all studies
    for (i in data$id) {
        # If it's an RPP study, save first/senior author data to data table.
        # Call it last author to match the other data set but actually its
        # not necessarily the last author.
        if (i %in% data.rpp$id) {
            rpp_id <- which(data.rpp$id == i)
            rpp_authors <- osf_scrape(data.rpp[rpp_id, Project.URL])
            for (j in 1:length(rpp_authors)) {
                new <- data.table(study_id = data.rpp$id[rpp_id],
                                  study_type = "r",
                                  author_id = NA_integer_,
                                  first_name = sub(rpp_authors[j], pattern="^(.*)\\s.*$", replacement="\\1"),
                                  last_name = sub(rpp_authors[j], pattern=".*\\s(.*$)", replacement="\\1"),
                                  author_order = j,
                                  first_author = ifelse(j == 1, TRUE, FALSE),
                                  last_author = ifelse(j == length(rpp_authors), TRUE, FALSE),
                                  n_authors = length(rpp_authors),
                                  first_author_name = data.rpp[rpp_id, X1st.author..R.],
                                  first_author_citations = data.rpp[rpp_id, Citation.count..1st.author..R.],
                                  first_author_institution = data.rpp[rpp_id, Institution.1st.author..R.],
                                  last_author_name = data.rpp[rpp_id, Senior.author..R.],
                                  last_author_citations = data.rpp[rpp_id, Citation.count..senior.author..R.],
                                  last_author_institution = data.rpp[rpp_id, Institution.senior.author..R.]
                                  )
                authors <- rbind(authors, new, fill = TRUE)
            }
        }
        # Otherwise just save one empty entry with study id
        else {
            new <- data.table(study_id = i, study_type = "r")
            authors <- rbind(authors, new, fill = TRUE)
        }
    }
    return(authors)
}


###############################################################################
# Generate replication author data file using Scopus API

get_scopus_author_info <- function(first, last) {
    library(httr)
    query <- paste0("authfirst(", first, ") AND authlast(", last, ")")
    # field <- "dc:identifier"
    # we could get affiliation data here too
    field <- "prism:url,dc:identifier,subject-area,affiliation-name,affiliation-city,affiliation-country"

    session <- GET("http://api.elsevier.com/content/search/author",
                   add_headers("X-ELS-APIKey" = "33375c07ac4d0a7bc6fa0a918aa34bdc"),
                   query = list("query" = query, "field" = field)
    )
    if (content(session)$`search-results`$`opensearch:totalResults` > 1) {
        warning(paste("More than one result found for author:", first, last))
    }
    # Always pick first result (if zero result coerce null to character type)
    data <- content(session)$`search-results`$entry[[1]]

    url <- data$`prism:url`
    id <- gsub("AUTHOR_ID:", "",
               data$`dc:identifier`,
               fixed = TRUE)
    subject <- data$`subject-area`[[1]]$`@abbrev`
    affiliation <- data$`affiliation-current`$`affiliation-name`
    city <- data$`affiliation-current`$`affiliation-city`
    country <- data$`affiliation-current`$`affiliation-country`

    return(as.list(as.character(list(url, id, subject, affiliation, city, country))))
}

get_scopus_citation_data <- function(id) {
    # Fetches citation data from Scopus API
    # HOWEVER: datastore trails 1 month behind web, and does not include pre
    # 1996 data. So better to fetch manually.
    # ALSO: the code just looks at the best match, so if the actual author
    # is not the most famous author with that name, the match will be wrong.

    if (id == "" | id == "NULL") { return(list("","","")) }
    library(httr)

    url <- paste0("http://api.elsevier.com/content/author/author_id/", id)
    session <- GET(url,
                   add_headers("X-ELS-APIKey" = "33375c07ac4d0a7bc6fa0a918aa34bdc"),
                   query = list("view" = "metrics")
    )
    data <- content(session)
    citations <- data$`author-retrieval-response`[[1]]$coredata$`citation-count`
    h_index <- data$`author-retrieval-response`[[1]]$`h-index`
    co_authors <- data$`author-retrieval-response`[[1]]$`coauthor-count`
    return(as.list(as.character(list(citations, h_index, co_authors))))
}

if (!file.exists(paste0(sourcedir, "authors/author_data_r.csv"))) {
    print("Could not find replication author data, generating new file from Scopus.")

    a_data <- as.data.table(read.csv(
        paste0(sourcedir, "authors/author_data_r_initial.csv"),
        sep=";", fileEncoding="macroman"))

    for (i in 1:nrow(a_data)) {
        a_data[i, c("url", "scopus_id", "subject", "affiliation", "city",
                    "country") := get_scopus_author_info(a_data[i]$first_name,
                                                         a_data[i]$last_name)]

        a_data[i, c("citations", "h_index", "co_authors") :=
                get_scopus_citation_data(a_data[i]$scopus_id)]

        print(paste("Fetched scopus data for:", a_data[i]$first_name,
                    a_data[i]$last_name))
    }

    write.csv(a_data, paste0(sourcedir, "authors/author_data_r.csv"), row.names = FALSE,
              na="", fileEncoding="UTF-8")
}

###############################################################################
# Create new files if needed, otherwise just load the data

if (file.exists(paste0(sourcedir, "authors/authors_o.csv"))) {
    print("Using provided authors_o.csv.")
    authors.o <- as.data.table(read.csv(paste0(sourcedir, "authors/authors_o.csv")))
} else {
    print("Could not find authors_o.csv, generating a new original author list. (NEW IDs)")
    authors.o <- generate_author_o()
    write.csv(authors.o, paste0(sourcedir, "authors/authors_o.csv"), row.names=FALSE,
              na="", fileEncoding="UTF-8")
}

if (file.exists(paste0(sourcedir, "authors/authors_r_excel.csv"))) {
    print("Using provided authors_r_excel.csv (assuming Macroman encoding).")
    authors.r <- as.data.table(read.csv(paste0(sourcedir, "authors/authors_r_excel.csv"), sep=";", fileEncoding="macroman"))
} else if (file.exists(paste0(sourcedir, "authors/authors_r.csv"))) {
    print("Using provided authors_r.csv that has been generated previously (assuming UTF-8).")
    authors.r <- as.data.table(read.csv(paste0(sourcedir, "authors/authors_r.csv")))
} else {
    print("Could not find authors_r.csv, generating a new replication author list. (NEW IDs)")
    authors.r <- generate_author_r()
    write.csv(authors.r, paste0(sourcedir, "authors/authors_r.csv"), row.names=FALSE,
              na="", fileEncoding="UTF-8")
}

# Author lists are identified by unique study-author pairs
setkey(authors.o, study_id, author_id)
setkey(authors.r, study_id, author_id)

###############################################################################
# Create one author table with all authors

if (!file.exists("../data/authors.csv")) {
    # Original authors already have ID's that should not be changed

    # Some replication authors have manually added ID's, which we keep
    # all other replication authors get new ID's.

    # Get max author id from original authors/replication authors
    max_id <- max(max(as.numeric(gsub(authors.o$author_id, pattern="author.",
                                      replace="", fixed=TRUE))),
                  max(as.numeric(gsub(authors.r$author_id, pattern="author.",
                                      replace="", fixed=TRUE)), na.rm=TRUE))

    # Loop through replication authors to check if they are also original
    # authors, otherwise assign them a new id.
    for (i in 1:nrow(authors.r)) {
        # If the author does not have a pre-assigned ID
        if (authors.r[i]$author_id == "") {
            # We try to assign the same author id if author is also original author
            new_id <- unique(authors.o[first_name == authors.r[i]$first_name &
                      last_name == authors.r[i]$last_name, author_id])

            if (authors.r[i]$first_name == "" & authors.r[i]$last_name == "") {
                new_id <- ""
            } else if (length(new_id) == 0) {
                # All authors with same first & last name should get same ID
                # First time the author appears he/she needs new ID though
                if (authors.r[i]$first_name %in% authors.r[1:i]$first_name &
                    authors.r[i]$last_name %in% authors.r[1:i]$last_name &
                    i != which(authors.r$first_name == authors.r[i]$first_name &
                    authors.r$last_name == authors.r[i]$last_name)[1]) {
                    new_id <- authors.r[first_name == authors.r[i]$first_name &
                                        last_name == authors.r[i]$last_name,
                                        author_id][1]
                }
                else {
                    max_id <- max_id + 1
                    new_id <- paste0("author.", max_id)
                }
            } else if (length(new_id) > 1) {
                stop("Error identifying unique author to match with replication
                     author")
            }
            authors.r[i]$author_id <- new_id
        }
    }

    # Bind the two author files together
    authors <- rbind(authors.o, authors.r, fill = TRUE)

    # remove uneccesary columns from authors.csv (just key data)
    authors[, c("first_author", "last_author",  "first_author_name",
                "first_author_citations", "first_author_institution",
                "last_author_name", "last_author_citations",
                "last_author_institution") := NULL]
    # Last we write everything to an authors-file.
    write.csv(authors, "../data/authors.csv", row.names=FALSE,
              na="", fileEncoding="UTF-8")

    print(paste0("Generated a new author.csv with ", nrow(authors), " author-study pairs. The maximum author_id was ", max_id, "."))
}
