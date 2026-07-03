# Predicting the Reproducibility of Lab Experiments in the Social Sciences

## Code
The `code` folder contains all code. The code is designed to be ran on an Amazon AWS server by executing the file `nested_cv_initiate`. While it would take a substantial amount of time it is also possible to run the code on one's own computer. To do so, just source `nested_cv.R` in an `R` console.

To train the models on an Amazon AWS Compute instance you need to configure a number of things. First of all, amazon [`awscli`](https://aws.amazon.com/cli/) is required. With it installed, you need to edit the executable `nested_cv_initiate` with your own settings. Create an S3 bucket where to store the files, setup awscli and get a AWS key with the correct credentials. [This guide](https://github.com/adamaltmejd/r-aws-scripts) explains the essentials. When everything is configures you should just be able to run `./nested_cv_initiate` and the code should be executed. When it's finished you receive an email notification and can download the `.RData` files that include the different models. With the models available in the `data` folder, the file `generate_figures.R` can be sourced to generate all figures in the paper.

If training for some reason does not work contact the authors and we will try to provide you with pre trained model files.

## Data
All data files are available in the `data` directory. The main dataset is contained in `data.csv`, with supplementary information about authors and their connections to different studies in the two other files. In the data set, one observation is a experiment-replication pair. Variable names ending with `.o` refer to the original experiment while `.r` refers to the replication. See `variables.csv` for a description of all features.

### Notes

*   Planned sample size and power is based on actual sample size and the original effect size if no data is available
*   For aggregated data, the mode is used for lab specifics. I.e. most common compensation scheme, country, etc.
*   Because author citations were collected over a number of weeks (4) in March/April 2016, some counts could be slightly higher because of a later collection date.
*   Sometimes, information about country/language etc is not available in a paper or its replication protocol. Whenever that is the case, but the answer is quite obvious (e.g. a replication of a paper where the original was done in the US and the replication authors are affiliated to American institutions) we make an educated guess to avoid missing values. (and note it with a comment in the data file)
*   Many Labs 2 is currently not included in the dataset because the study is not finished.
