Overview
This repository contains scripts to build and evaluate post-fire debris flow prediction models using 2- and 3-feature combinations under different algorithms, rainfall intensities, and weighting schemes. Below are detailed instructions for running each script and the required input files.

In order to run 2_feature_models.py, you need to download data_rained.xlsx and run data_2_feature_matrices.py, saving the outputs as called in 2_feature_models.py:
i10_2, i15_2, i30_2, i60_2.

In order to run 3_feature_models.py, you need to download data_rained.xlsx and run data_3_feature_matrices.py, saving the outputs as called in 3_feature_models.py:
i10_3, i15_3, i30_3, i60_3.

To run either 2_feature_models.py or 3_feature_models.py, you also need to download train_4k.csv and test_4k.csv.

2_feature_models.py and 3_feature_models.py are single-intensity, single-algorithm, and single-weight scripts, resulting in 80 and 400 models, respectively. You have to set the intensity, algorithm, and weight you wish to compute — all options are available in the code as comments.

Since threat score (TS) is the metric used to determine postfire debris-flow likelihood, the results for each script are the mean TS from the 4-fold, and the median TS from the 10-permutation run, ranked in decreasing order. Next to each TS is a number from 0 to 79 (or 0 to 399), indicating the feature combination. You can find the feature combinations in variables.xlsx. Remember: Python indexes from 0, so if you want to investigate a feature combination further, subtract 1 when looking it up in Excel.

two_2_feature_models.py, as the name says, only runs two 2-feature models — the ones selected for the SWA and SWB models. It also requires train_4k.csv, test_4k.csv, and the output from data_2_feature_matrices.py, saved as i10_2, i15_2, i30_2, i60_2 (same as 2_feature_models.py). The SWA and SWB models use logistic regression, i15 intensity, and square root weighting by default, but the code can be edited to use other algorithms, intensities, or weights. The result includes different accuracy metrics for just these two models.

SWA_SWB_models.py runs the final models selected for i15 and i30 using the SWA and SWB features. It outputs the three model coefficients for each case. It also runs SHAP (Shapley) values and PDP (partial dependence plots), so this file includes the models created for the four algorithms at i15, along with their interpretation graphs.

