In order to run 2_feature_models.py you need to download data_rained.xlsx and run data_2_feture_matrices.py and save the output as called in 2_feature_models.py (i10_2, i15_2, i30_2, i60_2) 

In order to run 3_feature_models.py you need to download data_rained.xlsx and run data_3_feture_matrices.py and save the output as called in 3_feature_models.py (i10_3, i15_3, i30_3, i60_3)

To run 2_feature_models.py and  3_feature_models.py  you also need to download train_4k.csv and test_4k.cvs files.  


2_feature_models.py and  3_feature_models.py are single intensity, single algorithm, and single weight,  resulting in 80 and 400 models, respectively. You have to set what intensity, algorithm, and weight you wish to compute. All are available in code as comments. Since threat score (TS) is what we use to determine the likelihood of a post fire debris flow, these models' results are the median TS of the 4_fold, 10 permutation run in decreasing order. Next to it are numbers from 0 to 79, indicating the feature combination. You may see these feature combinations by opening variables.xlsx. Remember that Python counts from 0, so if you wish to investigate a feature combination further, you must subtract 1. 


two_2_feature_models.py is, as the name says, only 2 2_ferature models. The features that are set to run are the ones selected for the SWA model and SWB model. This file also needs you to download train_4k.csv and test_4k.cvs files and run data_2_feture_matrices.py and save the output as called in two_2_feature_models.py (i10_2, i15_2, i30_2, i60_2). SWA and SWB use LR, i15, and sqrt weighting, but the code can be manipulated to run for any algorithm, weight, and intensity. The result is different accuracy metrics but for two models, one algorithm, one rain intensity, and 1 weighting.

If you run and save the results of two_2_feature_models.py with different algorithms (same weight, same intensity to be able to compare), you can run metric_acuracy_tables. It is set for the four algorithms for two features and two rain intensities. You can also graph the single threshold AUC ROC curve and  AUC PR curve.

SWA_ SWB_models are the models selected for i15 and i30, which will give the three coefficients for the model. Shapley values and PDP are run on the models, so the file has the models created with SWA and SWB features for the four algorithms with i15 and their graphs. 

To run the watershed threshold, download con01.xlsx and data_clean_xlsx and run the watershed initiation value. It will give the values at which SWA and SWB would trigger a debris flow for 3 watersheds. We trained with Pinal Fire Watershed 651 South, Tadpole Fire Watershed D, and Contreras Fire, which we did not train on; we tested  Watershed CON 01. 

