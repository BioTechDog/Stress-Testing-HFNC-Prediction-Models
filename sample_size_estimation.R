if(!require('pmsampsize')) {install.packages('pmsampsize'); library(pmsampsize)}
if(!require('rms')) {install.packages('rms'); library(rms)}

data <- read.csv("RENOVATE_data_filtered_features.csv")

model <- lrm(HFNO_failure ~ Age..y. + HR_T0..bpm. + RR_T0..bpm. +
               FiO2_T1.... + RR_T1..bpm. + S.F_T1 + SpO2_T1..bpm.,
             data = data)

colnames(data)
print(model)  # It will show R² and other metrics

num_predictors <- 7           # Number of predictors
rsq <- 0.242                  # From the lrm model output
prevalence <- 140 / 596       # Prevalence of HFNO_failure (140 Failures out of 596)
shrinkage <- 0.9              # Desired shrinkage factor to avoid overfitting

sample_size_result <- pmsampsize(type = "b",
                                 csrsquared = cs_rsq,
                                 parameters = num_predictors,
                                 prevalence = prev,
                                 shrinkage = shrink)

print(sample_size_result)
