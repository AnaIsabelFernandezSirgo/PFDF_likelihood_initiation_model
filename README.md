Overview

This repository contains scripts to build and evaluate post-fire debris flow prediction models using 2- and 3-feature combinations under different algorithms, rainfall intensities, and weighting schemes. Below are detailed instructions for running each script and the required input files.

Instructions

In order to run 2_feature_models.py, you need to download data_rained.xlsx and run data_2_feature_matrices.py, saving the outputs as called in 2_feature_models.py:
i10_2, i15_2, i30_2, i60_2.

In order to run 3_feature_models.py, you need to download data_rained.xlsx and run data_3_feature_matrices.py, saving the outputs as called in 3_feature_models.py:
i10_3, i15_3, i30_3, i60_3.

To run either 2_feature_models.py or 3_feature_models.py, you also need to download train_4k.csv and test_4k.csv.

2_feature_models.py and 3_feature_models.py are single-intensity, single-algorithm, and single-weight scripts, resulting in 80 and 400 models, respectively. You have to set the intensity, algorithm, and weight you wish to compute — all options are available in the code as comments.

Since threat score (TS) is the metric used to determine postfire debris-flow likelihood, the results for each script are the mean TS from the 4-fold, and the median TS from the 10-permutation run, ranked in decreasing order. Next to each TS is a number from 0 to 79 (or 0 to 399), indicating the feature combination. You can find the feature combinations in variables.xlsx. Remember: Python indexes from 0, so if you want to investigate a feature combination further, subtract 1 when looking it up in Excel.

two_2_feature_models.py, as the name suggests, runs only two 2-feature models. These models use the feature combinations selected for the SWA and SWB models. To run this script, you need to download train_4k.csv and test_4k.csv, and run data_2_feature_matrices.py, saving the outputs as expected by two_2_feature_models.py:
i10_2, i15_2, i30_2, and i60_2.

By default, the SWA and SWB models use logistic regression (LR), the i15 rainfall intensity, and square root weighting. However, the code can be easily modified to run with any other algorithm, weighting method, or rainfall intensity. The output includes multiple accuracy metrics, but only for these two specific models — one algorithm, one intensity, one weighting.

 

