# Ensure the pmsampsize package is installed and loaded
if(!require('pmsampsize')) {install.packages('pmsampsize'); library(pmsampsize)}
# Ensure required packages are installed
if(!require('rms')) {install.packages('rms'); library(rms)}

# Load your data
data <- read.csv("RENOVATE_data_filtered_features.csv")

# Fit logistic model (assuming HFNO_failure is coded as 0 = success, 1 = failure)
model <- lrm(HFNO_failure ~ Age..y. + HR_T0..bpm. + RR_T0..bpm. +
               FiO2_T1.... + RR_T1..bpm. + S.F_T1 + ROX_T1,
             data = data)

# View actual column names
colnames(data)
# View Cox-Snell R²
print(model)  # It will show R² and other metrics

# Define parameters based on your dataset
num_predictors <- 7           # Number of predictors
rsq <- 0.242                  # From the lrm model output
prevalence <- 140 / 596       # Prevalence of HFNO_failure (140 Failures out of 596)
shrinkage <- 0.9              # Desired shrinkage factor to avoid overfitting

# Run the sample size calculation
sample_size_result <- pmsampsize(type = "b",
                                 csrsquared = cs_rsq,
                                 parameters = num_predictors,
                                 prevalence = prev,
                                 shrinkage = shrink)

# Print the result
print(sample_size_result)