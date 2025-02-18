In order to run 2_feature_models.py you need to download data_rained.xlsx and run data_2_feture_matrices.py and save the output as called in 2_feature_models.py (i10_2, i15_2, i30_2, i60_2) 

In order to run 3_feature_models.py you need to download data_rained.xlsx and run data_3_feture_matrices.py and save the output as called in 3_feature_models.py (i10_3, i15_3, i30_3, i60_3)

To run 2_feature_models.py and  3_feature_models.py  you also neeed to download train_4k.csv and test_4k.cvs files. 


2_feature_models.py and  3_feature_models.py are single intensity, single algorithm, single wheight,  resulting on 80 and 400 models, respectivly. You have to set what intensity what algorithm and what weight you wish to compute. All are available in code as comments.  Since Threat Score is what we use to determine the likelihood of a post fire debireflow the result from these models are the median TS of the 4_fold 10 permutation run; next to it are numbers from 0 to 79 indicating the feature combination. You may see what these feature combinations are by opening 


tow_2_feature_models.py is, as is named says, only 2 2_ferature model. The features that are set to run to are the ones selected for SWA model and SWB model. This file also needs for you to download train_4k.csv and test_4k.cvs files and run data_2_feture_matrices.py and save the output as called in two_2_feature_models.py (i10_2, i15_2, i30_2, i60_2). SWA and SWB uses LR, i15 and, sqrt weghting, but the code can be manipulated to run for any algorithm, any wight, any intensity. The result are different accuracy metrics but for 2 models, 1 algoriuthm, 1 rain intensity, 1 weighting.

