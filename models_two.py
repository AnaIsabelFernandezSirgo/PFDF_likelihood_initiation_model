# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 06:40:34 2025

@author: 15206
"""

import pandas as pd
import numpy as np
from itertools import product
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_recall_curve, auc,  average_precision_score

 # SWA and SWB features  all algorithms i 15 
#DOWNLOADING DATA MULTIPLIED BY RAIN 
data_rained=pd.read_excel(r'D:/PFDF_08_2025/two_feature_c.xlsx', sheet_name=[ 'Sheet2','Sheet3','Sheet4' ,'Sheet5','Sheet6','Sheet7'])

i2_s=data_rained['Sheet2']
i5_s=data_rained['Sheet3']
i10_s=data_rained['Sheet4']
i15_s=data_rained['Sheet5']
i30_s=data_rained['Sheet6']
i60_s=data_rained['Sheet7']



#SEPARATING DATA INTO CATEGORIES

response=i2_s[['Response']].copy()

number=i2_s[['No']].copy()
vtype=i2_s[['VT']].copy()



re=response.to_numpy()


topography_col=['Ruggedness_S',	'Slope_mean','s23'	,'s23MH',	's23H'	,'a2000s15','a2000s15MH'	,'a2000s15H',	'a2000s23',	'a2000s23MH',	'a2000s23H'	,'MeanSlopeLMH',	'MeanSlopeMH'	,'S23LMH'	,'S30LMH',	'S30MH']







t_i2=i2_s[topography_col].to_numpy(copy=True)
t_i5=i5_s[topography_col].to_numpy(copy=True)
t_i10=i10_s[topography_col].to_numpy(copy=True)
t_i15=i15_s[topography_col].to_numpy(copy=True)
t_i30=i30_s[topography_col].to_numpy(copy=True)
t_i60 = i60_s[topography_col].to_numpy(copy=True)



fire_col=['mdNBR_1000'	,'MH'	,'L_2_MH'	,'L_3_MH',	'L_4_MH']



f_i2=i2_s[fire_col].to_numpy(copy=True)
f_i5= i5_s[fire_col].to_numpy(copy=True)
f_i10= i10_s[fire_col].to_numpy(copy=True)
f_i15= i15_s[fire_col].to_numpy(copy=True)
f_i30= i30_s[fire_col].to_numpy(copy=True)
f_i60= i60_s[fire_col].to_numpy(copy=True)


number_data,number_terrain=t_i10.shape
number_fire=f_i10.shape[1]


number_combo=number_fire*number_terrain

#CREATING THE MATRICES WITH THE DIFFERENT COMBINATIONS

i2=np.empty((number_data, 5,number_combo))
i5=np.empty((number_data, 5,number_combo))
i10=np.empty((number_data, 5,number_combo))
i15=np.empty((number_data, 5,number_combo))
i30=np.empty((number_data, 5,number_combo))
i60=np.empty((number_data, 5,number_combo))
s=0;


for s, (j, k) in enumerate(product(range(number_terrain), range(number_fire))):
    i2[:, :, s] = np.column_stack((re[:, 0], t_i2[:, j], f_i2[:, k], number, vtype))
    i5[:, :, s] = np.column_stack((re[:,0],t_i5[:,j],f_i5[:,k],number,vtype))
    i10[:, :, s]= np.column_stack((re[:,0],t_i10[:,j],f_i10[:,k],number,vtype))
    i15[:, :, s]= np.column_stack((re[:,0],t_i15[:,j],f_i15[:,k],number,vtype))
    i30[:, :, s]= np.column_stack((re[:,0],t_i30[:,j],f_i30[:,k],number,vtype))
    i60[:, :, s]= np.column_stack((re[:,0],t_i60[:,j],f_i60[:,k],number,vtype))


# STRATIFIED SAMPLING 75% YES 75% NO: TRAIN.  25% YES 25% NO: TEST
np.random.seed(52)
no_total=re.size-np.sum(re)
percent=.75
yes_total=int(np.sum(re))
no_train=round(no_total*percent)
yes_train=round(np.sum(re)*percent)
no_test=int(no_total-no_train)
yes_test=int(yes_total-yes_train)
number_permutation=10

matrix_no=np.array([np.random.permutation(no_total) for _ in range(number_permutation)]).T
matrix_yes = np.array([np.random.permutation(np.arange(no_total, re.size)) for _ in range(number_permutation)]).T

matrix_train_no_f1=matrix_no[no_test:no_total,:]
matrix_train_no_f2=np.vstack((matrix_no[0:no_test],matrix_no[2*no_test:no_total]))
matrix_train_no_f3=np.vstack((matrix_no[0:2*no_test],matrix_no[3*no_test:no_total]))
matrix_train_no_f4=matrix_no[0:3*no_test-1]
#ignaoring a point that goes to test


matrix_test_no_f1=matrix_no[0:no_test]
matrix_test_no_f2=matrix_no[no_test:2*no_test]
matrix_test_no_f3=matrix_no[2*no_test:3*no_test]
matrix_test_no_f4=matrix_no[3*no_test-1:no_total]
# repeated point 1st f4 last f3 

matrix_train_yes_f1=matrix_yes[yes_test:yes_total,:]
matrix_train_yes_f2=np.vstack((matrix_yes[0:yes_test],matrix_yes[2*yes_test:yes_total]))
matrix_train_yes_f3=np.vstack((matrix_yes[0:2*yes_test],matrix_yes[3*yes_test:yes_total]))
matrix_train_yes_f4=matrix_yes[0:3*yes_test+1]
#there is a point repeating so it can match number of rows last of f3 1st of f4 

matrix_test_yes_f1=matrix_yes[0:yes_test]
matrix_test_yes_f2=matrix_yes[yes_test:2*yes_test]
matrix_test_yes_f3=matrix_yes[2*yes_test:3*yes_test]
matrix_test_yes_f4=matrix_yes[3*yes_test+1:yes_total]
# we are ignoring a point but it went to the repeated point on train 

matrix_train_f1=np.vstack((matrix_train_no_f1,matrix_train_yes_f1))
matrix_train_f2=np.vstack((matrix_train_no_f2,matrix_train_yes_f2))
matrix_train_f3=np.vstack((matrix_train_no_f3,matrix_train_yes_f3))
matrix_train_f4=np.vstack((matrix_train_no_f4,matrix_train_yes_f4))

matrix_test_f1=np.vstack((matrix_test_no_f1,matrix_test_yes_f1))
matrix_test_f2=np.vstack((matrix_test_no_f2,matrix_test_yes_f2))
matrix_test_f3=np.vstack((matrix_test_no_f3,matrix_test_yes_f3))
matrix_test_f4=np.vstack((matrix_test_no_f4,matrix_test_yes_f4))



matrix_test = np.stack(
    [matrix_test_f1, matrix_test_f2, matrix_test_f3, matrix_test_f4],
    axis=2
)

matrix_train = np.stack(
    [matrix_train_f1, matrix_train_f2, matrix_train_f3, matrix_train_f4],
    axis=2
)


length_test=matrix_test.shape[0]
length_train=matrix_train.shape[0]


number_kfolds=4 # kfold
number_fire=5 # fire
number_terrain=16 # terrrain
number_rain=6 # rain 
#SWA is index 7 SWB index 5
vv=[5,7]
number_combo_chosen=len(vv)

test=np.empty((length_test,5,number_combo_chosen,number_permutation,number_kfolds))
train=np.empty((length_train,5,number_combo_chosen,number_permutation,number_kfolds));



for cc in range (number_combo_chosen):
    for p in range (number_permutation):
        for k in range (number_kfolds):
                ver=i15[:,:,vv[cc]]
               

                test[:,:,cc,p,k]=ver[matrix_test[:,p,k],:];
                train[:,:,cc,p,k]=ver[matrix_train[:,p,k],:];

#allocating

lm_fold_ts_k=np.empty((number_combo_chosen,number_permutation,number_kfolds ))
lm_fold_tp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_fold_fn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_fold_fp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_fold_tn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))


lm_all_info=np.empty((number_combo_chosen, number_permutation, number_kfolds, length_test,6))
lm_info=np.empty((length_test,6))
lm_RandA=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_pr_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_roc_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_ap=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lm_ap_one=np.empty((number_combo_chosen,number_permutation,number_kfolds))


lm_p_ts_p=np.empty((number_combo_chosen,number_permutation))
lm_p_tp_p=np.empty((number_combo_chosen,number_permutation))
lm_p_fn_p=np.empty((number_combo_chosen,number_permutation))
lm_p_fp_p=np.empty((number_combo_chosen,number_permutation))
lm_p_tn_p=np.empty((number_combo_chosen,number_permutation))
lm_p_RandA=np.empty((number_combo_chosen,number_permutation))
lm_p_pr_auc=np.empty((number_combo_chosen,number_permutation))
lm_p_roc_auc=np.empty((number_combo_chosen,number_permutation))
lm_p_ap=np.empty((number_combo_chosen,number_permutation))
lm_p_ap_one=np.empty((number_combo_chosen,number_permutation))

lm_pre=np.empty((length_test,number_combo_chosen,number_permutation,number_kfolds ))

#LDA

lda_fold_ts_k=np.empty((number_combo_chosen,number_permutation,number_kfolds ))
lda_fold_tp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_fold_fn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_fold_fp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_fold_tn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))

lda_all_info=np.empty((number_combo_chosen, number_permutation, number_kfolds, length_test,6))
lda_info=np.empty((length_test,6))
lda_RandA=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_pr_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_roc_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_ap=np.empty((number_combo_chosen,number_permutation,number_kfolds))
lda_ap_one=np.empty((number_combo_chosen,number_permutation,number_kfolds))


lda_p_ts_p=np.empty((number_combo_chosen,number_permutation))
lda_p_tp_p=np.empty((number_combo_chosen,number_permutation))
lda_p_fn_p=np.empty((number_combo_chosen,number_permutation))
lda_p_fp_p=np.empty((number_combo_chosen,number_permutation))
lda_p_tn_p=np.empty((number_combo_chosen,number_permutation))
lda_p_RandA=np.empty((number_combo_chosen,number_permutation))
lda_p_pr_auc=np.empty((number_combo_chosen,number_permutation))
lda_p_roc_auc=np.empty((number_combo_chosen,number_permutation))
lda_p_ap=np.empty((number_combo_chosen,number_permutation))
lda_p_ap_one=np.empty((number_combo_chosen,number_permutation))


lda_pre=np.empty((length_test,number_combo_chosen,number_permutation,number_kfolds ))



#RF

rf_fold_ts_k=np.empty((number_combo_chosen,number_permutation,number_kfolds ))
rf_fold_tp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_fold_fn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_fold_fp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_fold_tn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))

rf_all_info=np.empty((number_combo_chosen, number_permutation, number_kfolds, length_test,6))
rf_info=np.empty((length_test,6))
rf_RandA=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_pr_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_roc_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_ap=np.empty((number_combo_chosen,number_permutation,number_kfolds))
rf_ap_one=np.empty((number_combo_chosen,number_permutation,number_kfolds))


rf_p_ts_p=np.empty((number_combo_chosen,number_permutation))
rf_p_tp_p=np.empty((number_combo_chosen,number_permutation))
rf_p_fn_p=np.empty((number_combo_chosen,number_permutation))
rf_p_fp_p=np.empty((number_combo_chosen,number_permutation))
rf_p_tn_p=np.empty((number_combo_chosen,number_permutation))
rf_p_RandA=np.empty((number_combo_chosen,number_permutation))
rf_p_pr_auc=np.empty((number_combo_chosen,number_permutation))
rf_p_roc_auc=np.empty((number_combo_chosen,number_permutation))
rf_p_ap=np.empty((number_combo_chosen,number_permutation))
rf_p_ap_one=np.empty((number_combo_chosen,number_permutation))


rf_pre=np.empty((length_test,number_combo_chosen,number_permutation,number_kfolds ))


#XGB

xgb_fold_ts_k=np.empty((number_combo_chosen,number_permutation,number_kfolds ))
xgb_fold_tp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_fold_fn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_fold_fp_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_fold_tn_k=np.empty((number_combo_chosen,number_permutation,number_kfolds))


xgb_all_info=np.empty((number_combo_chosen, number_permutation, number_kfolds, length_test,6))
xgb_info=np.empty((length_test,6))
xgb_RandA=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_pr_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_roc_auc=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_ap=np.empty((number_combo_chosen,number_permutation,number_kfolds))
xgb_ap_one=np.empty((number_combo_chosen,number_permutation,number_kfolds))


xgb_p_ts_p=np.empty((number_combo_chosen,number_permutation))
xgb_p_tp_p=np.empty((number_combo_chosen,number_permutation))
xgb_p_fn_p=np.empty((number_combo_chosen,number_permutation))
xgb_p_fp_p=np.empty((number_combo_chosen,number_permutation))
xgb_p_tn_p=np.empty((number_combo_chosen,number_permutation))
xgb_p_RandA=np.empty((number_combo_chosen,number_permutation))
xgb_p_pr_auc=np.empty((number_combo_chosen,number_permutation))
xgb_p_roc_auc=np.empty((number_combo_chosen,number_permutation))
xgb_p_ap=np.empty((number_combo_chosen,number_permutation))
xgb_p_ap_one=np.empty((number_combo_chosen,number_permutation))


xgb_pre=np.empty((length_test,number_combo_chosen,number_permutation,number_kfolds ))


lm_ra=np.empty((number_combo_chosen,1))
lm_roc_auc_m=np.empty((number_combo_chosen,1))
lm_ap_m=np.empty((number_combo_chosen,1))
lm_ap_one_m=np.empty((number_combo_chosen,1))
lm_ts_median=np.empty((number_combo_chosen,1)) 
lm_ts_mean=np.empty((number_combo_chosen,1))
lm_ts_m=np.empty((number_combo_chosen,3)) 
lm_tp_m=np.empty((number_combo_chosen,1))
lm_fp_m=np.empty((number_combo_chosen,1))
lm_fn_m=np.empty((number_combo_chosen,1))
lm_tn_m=np.empty((number_combo_chosen,1))
lm_pi_m=np.empty((number_combo_chosen,1))
lm_positive_predictive_value_or_precision=np.empty((number_combo_chosen,1))
lm_false_positive_rate=np.empty((number_combo_chosen,1))
lm_true_negative_rate_or_specificity=np.empty((number_combo_chosen,1))
lm_positive_predictive_value=np.empty((number_combo_chosen,1))
lm_negative_predicted_value=np.empty((number_combo_chosen,1))
lm_recall_or_sensitivity_or_true_positive_rate=np.empty((number_combo_chosen,1))
lm_f1_score=np.empty((number_combo_chosen,1))
lm_ts_median_diviation_fold= np.empty((number_combo_chosen,1))  
lm_pr2= np.zeros((number_combo_chosen,1)) 


#LDA
lda_ra=np.empty((number_combo_chosen,1))
lda_roc_auc_m=np.empty((number_combo_chosen,1))
lda_ap_m=np.empty((number_combo_chosen,1))
lda_ap_one_m=np.empty((number_combo_chosen,1))
lda_ts_median=np.empty((number_combo_chosen,1)) 
lda_ts_mean=np.empty((number_combo_chosen,1))
lda_ts_m=np.empty((number_combo_chosen,3)) 
lda_tp_m=np.empty((number_combo_chosen,1))
lda_fp_m=np.empty((number_combo_chosen,1))
lda_fn_m=np.empty((number_combo_chosen,1))
lda_tn_m=np.empty((number_combo_chosen,1))
lda_pi_m=np.empty((number_combo_chosen,1))
lda_positive_predictive_value_or_precision=np.empty((number_combo_chosen,1))
lda_false_positive_rate=np.empty((number_combo_chosen,1))
lda_true_negative_rate_or_specificity=np.empty((number_combo_chosen,1))
lda_positive_predictive_value=np.empty((number_combo_chosen,1))
lda_negative_predicted_value=np.empty((number_combo_chosen,1))
lda_recall_or_sensitivity_or_true_positive_rate=np.empty((number_combo_chosen,1))
lda_f1_score=np.empty((number_combo_chosen,1))
lda_ts_median_diviation_fold= np.empty((number_combo_chosen,1))  
lda_pr2= np.zeros((number_combo_chosen,1))

#RF
rf_ra=np.empty((number_combo_chosen,1))
rf_roc_auc_m=np.empty((number_combo_chosen,1))
rf_ap_m=np.empty((number_combo_chosen,1))
rf_ap_one_m=np.empty((number_combo_chosen,1))
rf_ts_median=np.empty((number_combo_chosen,1)) 
rf_ts_mean=np.empty((number_combo_chosen,1))
rf_ts_m=np.empty((number_combo_chosen,3)) 
rf_tp_m=np.empty((number_combo_chosen,1))
rf_fp_m=np.empty((number_combo_chosen,1))
rf_fn_m=np.empty((number_combo_chosen,1))
rf_tn_m=np.empty((number_combo_chosen,1))
rf_pi_m=np.empty((number_combo_chosen,1))
rf_positive_predictive_value_or_precision=np.empty((number_combo_chosen,1))
rf_false_positive_rate=np.empty((number_combo_chosen,1))
rf_true_negative_rate_or_specificity=np.empty((number_combo_chosen,1))
rf_positive_predictive_value=np.empty((number_combo_chosen,1))
rf_negative_predicted_value=np.empty((number_combo_chosen,1))
rf_recall_or_sensitivity_or_true_positive_rate=np.empty((number_combo_chosen,1))
rf_f1_score=np.empty((number_combo_chosen,1))
rf_ts_median_diviation_fold= np.empty((number_combo_chosen,1))  
rf_pr2= np.zeros((number_combo_chosen,1))

#XGB
xgb_ra=np.empty((number_combo_chosen,1))
xgb_roc_auc_m=np.empty((number_combo_chosen,1))
xgb_ap_m=np.empty((number_combo_chosen,1))
xgb_ap_one_m=np.empty((number_combo_chosen,1))
xgb_ts_median=np.empty((number_combo_chosen,1)) 
xgb_ts_mean=np.empty((number_combo_chosen,1))
xgb_ts_m=np.empty((number_combo_chosen,3)) 
xgb_tp_m=np.empty((number_combo_chosen,1))
xgb_fp_m=np.empty((number_combo_chosen,1))
xgb_fn_m=np.empty((number_combo_chosen,1))
xgb_tn_m=np.empty((number_combo_chosen,1))
xgb_pi_m=np.empty((number_combo_chosen,1))
xgb_positive_predictive_value_or_precision=np.empty((number_combo_chosen,1))
xgb_false_positive_rate=np.empty((number_combo_chosen,1))
xgb_true_negative_rate_or_specificity=np.empty((number_combo_chosen,1))
xgb_positive_predictive_value=np.empty((number_combo_chosen,1))
xgb_negative_predicted_value=np.empty((number_combo_chosen,1))
xgb_recall_or_sensitivity_or_true_positive_rate=np.empty((number_combo_chosen,1))
xgb_f1_score=np.empty((number_combo_chosen,1))
xgb_ts_median_diviation_fold= np.empty((number_combo_chosen,1))  
xgb_pr2= np.zeros((number_combo_chosen,1))





c=-1;
p=0
cc=0
k=0

# The other weights are available it is set for sqrt SWA is LR index 7; SWB is LR index 5. 

#Model options
 
#None weight:
    
# lm_model = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000, fit_intercept=True)

# lda_model = LinearDiscriminantAnalysis( ) 
# #RF
# rf_model= RandomForestClassifier(n_estimators=100, max_features=None, random_state=0,max_depth=6,  n_jobs=-1, bootstrap=True ) 
# #XGB
# n_features =2
# mono = [1, 1] + [0]*(n_features-2)
# xgb_model = xgb.XGBClassifier(
#     booster='gbtree',
#     objective='binary:logistic',
#     learning_rate=0.05,
#     max_depth=5,               # shallower = less variance on ~1k rows
#     n_estimators=100,         # big cap 3000
#     min_child_weight=10,       # block brittle splits
#     gamma=2,                   # penalty on weak gains
#     reg_lambda=12,             # a bit more L2
#     subsample=1.0,
#     colsample_bytree=1.0,
#     colsample_bylevel=1.0,
#     colsample_bynode=1.0,
#     monotone_constraints=tuple(mono),  # must match X’s column order/length
#     random_state=0,
#     n_jobs=-1
# )

  
# Balanced weight 
# lm_model = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000,class_weight='balanced', fit_intercept=True)
# #  LDA convert that into class priors
# w = (no_train/yes_train)
# pi_pos = (w * yes_train) / (no_train + w * yes_train)
# pi_neg = 1.0 - pi_pos
# lda_model = LinearDiscriminantAnalysis( priors=[pi_neg, pi_pos]) 
# #RF
# rf_model= RandomForestClassifier(n_estimators=100, max_features=None, random_state=0,max_depth=6, class_weight='balanced', n_jobs=-1, bootstrap=True ) 
# #XGB
# n_features =2
# mono = [1, 1] + [0]*(n_features-2)
# xgb_model = xgb.XGBClassifier(
#     booster='gbtree',
#     objective='binary:logistic',
#     learning_rate=0.05,
#     max_depth=5,               # shallower = less variance on ~1k rows
#     n_estimators=100,         # big cap 3000
#     min_child_weight=10,       # block brittle splits
#     gamma=2,                   # penalty on weak gains
#     reg_lambda=12,             # a bit more L2
#     subsample=1.0,
#     colsample_bytree=1.0,
#     colsample_bylevel=1.0,
#     colsample_bynode=1.0,
#     scale_pos_weight=(no_train/yes_train),
#     monotone_constraints=tuple(mono),  # must match X’s column order/length
#     random_state=0,
#     n_jobs=-1
# )


    
#sqrt weight

## LR

lm_model = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000,class_weight={0:1, 1:np.sqrt(no_train/yes_train)}, fit_intercept=True)

# LDA


w = np.sqrt(no_train/yes_train)
# LDA convert that into class priors
pi_pos = (w * yes_train) / (no_train + w * yes_train)
pi_neg = 1.0 - pi_pos
lda_model = LinearDiscriminantAnalysis( priors=[pi_neg, pi_pos]) 

# RF
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=3,
   min_samples_split=10,
   min_samples_leaf=5,
    max_features=2,
    bootstrap=True,
    max_samples=.9,         # sub-sample rows per tree
    oob_score=True,
    class_weight={0:1, 1:np.sqrt(no_train/yes_train)},  # keep your scheme
    random_state=0,
    n_jobs=-1
)


# XGB 
n_features =2
mono = [1, 1] + [0]*(n_features-2)
xgb_model = xgb.XGBClassifier(
    booster='gbtree',
    objective='binary:logistic',
    learning_rate=0.05,
    max_depth=3,               # shallower = less variance on ~1k rows
    n_estimators=150,         # big cap 3000
    min_child_weight=10,       # block brittle splits
    gamma=2,                   # penalty on weak gains
    reg_lambda=12,             # a bit more L2
    subsample=1.0,
    colsample_bytree=1.0,
    colsample_bylevel=1.0,
    colsample_bynode=1.0,
    scale_pos_weight=np.sqrt(no_train/yes_train),
    monotone_constraints=tuple(mono),  # must match X’s column order/length
    random_state=0,
    n_jobs=-1
)

 
baseline =yes_test/(yes_test+no_test)
for cc in range (number_combo_chosen):
    c=c+1;
    #np.random.seed(13)
    for p in range (number_permutation):
                    
                    
                    for k in range (number_kfolds):
                     
                        
                        Test=test[:,:,cc,p,k]
                        Train=train[:,:,cc,p,k]
                        #same order of 1s different data np.random.seed(13+p*11)
                        
                        np.random.seed(13+k*11+p*17)
                        Test=Test[np.random.permutation(length_test),:]
                        Train=Train[np.random.permutation(length_train),:]
                        c_test=Test[:,0];
                        c_train=Train[:,0];
           

                        t_1=Test[:,1:3]
                        t_2=Train[:,1:3]
                        vt_test=Test[:,4]
                        number_test=Test[:,3]
                        
                        #LM
                        lm_s=lm_model.fit(t_2,c_train)
                        lm_probs = lm_s.predict_proba(t_1)
                        lm_fpr, lm_tpr, roc_thresh = roc_curve(c_test, lm_probs[:,1])
                        lm_roc_auc[cc,p,k]= auc(lm_fpr, lm_tpr)
                        lm_prec, lm_rec, pr_thresh = precision_recall_curve(c_test, lm_probs[:,1])
                        lm_ap[cc,p,k] = average_precision_score(c_test, lm_probs[:,1])
                        lm_pre[:,cc,p,k] = lm_s.predict(t_1)
                        lm_fold_tn_k[cc,p,k], lm_fold_fp_k[cc,p,k], lm_fold_fn_k[cc,p,k], lm_fold_tp_k[cc,p,k] = confusion_matrix(c_test, lm_pre[:,cc,p,k], labels=[0, 1]).ravel()
                        for w in range (len(lm_pre)):
                            if (lm_pre[w,cc,p,k]==c_test[w]):
                                if (((lm_pre[w,cc,p,k]==1)&(c_test[w]==1))):
                                    
                                    lm_info[w,0]=1 #TP
                                else:
                                    
                                    lm_info[w,1]=1 #TN
                            else:
                                if ((c_test[w]==1)):
                                    
                                    lm_info[w,2]=1 #FN
                                else:
                                    
                                    lm_info[w,3]=1 #FP 
                            
                        lm_info[:,4]=number_test
                        lm_info[:,5]=vt_test
                        lm_all_info[cc,p,k,:,:]=lm_info
                        lm_RandA[cc,p,k]=roc_auc_score(c_test, lm_pre[:,cc,p,k])
                        lm_prec2, lm_rec2, _ = precision_recall_curve(c_test, lm_pre[:,cc,p,k])
                        lm_pr_auc[cc,p,k] = auc(lm_rec2, lm_prec2)   # trapezoidal PR AUC
                        lm_ap_one [cc,p,k] = average_precision_score(c_test, lm_pre[:,cc,p,k])
                        lm_fold_ts_k[cc,p,k]=lm_fold_tp_k[cc,p,k]/(lm_fold_tp_k[cc,p,k]+lm_fold_fp_k[cc,p,k]+lm_fold_fn_k[cc,p,k])
                        
                        #LDA
                        
                        
                        
                        lda_s=lda_model.fit(t_2, c_train)
                        #lda_s=lda_model.fit(t_2,c_train)
                        lda_probs = lda_s.predict_proba(t_1)
                        lda_fpr, lda_tpr, roc_thresh = roc_curve(c_test, lda_probs[:,1])
                        lda_roc_auc[cc,p,k]= auc(lda_fpr, lda_tpr)
                        lda_prec, lda_rec, pr_thresh = precision_recall_curve(c_test, lda_probs[:,1])
                        lda_ap[cc,p,k] = average_precision_score(c_test, lda_probs[:,1])
                        lda_pre[:,cc,p,k] = lda_s.predict(t_1)
                        lda_fold_tn_k[cc,p,k], lda_fold_fp_k[cc,p,k], lda_fold_fn_k[cc,p,k], lda_fold_tp_k[cc,p,k] = confusion_matrix(c_test, lda_pre[:,cc,p,k]).ravel()
                        for w in range (len(lda_pre)):
                            if (lda_pre[w,cc,p,k]==c_test[w]):
                                if (((lda_pre[w,cc,p,k]==1)&(c_test[w]==1))):
                                    
                                    lda_info[w,0]=1 #TP
                                else:
                                    
                                    lda_info[w,1]=1 #TN
                            else:
                                if ((c_test[w]==1)):
                                    
                                    lda_info[w,2]=1 #FN
                                else:
                                    
                                    lda_info[w,3]=1 #FP 
                            
                        lda_info[:,4]=number_test
                        lda_info[:,5]=vt_test
                        lda_all_info[cc,p,k,:,:]=lda_info
                        lda_RandA[cc,p,k]=roc_auc_score(c_test, lda_pre[:,cc,p,k])
                        lda_prec2, lda_rec2, _ = precision_recall_curve(c_test, lda_pre[:,cc,p,k])
                        lda_pr_auc[cc,p,k] = auc(lda_rec2, lda_prec2)   # trapezoidal PR AUC
                        lda_ap_one [cc,p,k] = average_precision_score(c_test, lda_pre[:,cc,p,k])
                        lda_fold_ts_k[cc,p,k]=lda_fold_tp_k[cc,p,k]/(lda_fold_tp_k[cc,p,k]+lda_fold_fp_k[cc,p,k]+lda_fold_fn_k[cc,p,k]);
                        
                        #RF
                        rf_s=rf_model.fit(t_2,c_train)
                        rf_probs = rf_s.predict_proba(t_1)
                        rf_fpr, rf_tpr, roc_thresh = roc_curve(c_test, rf_probs[:,1])
                        rf_roc_auc[cc,p,k]= auc(rf_fpr, rf_tpr)
                        rf_prec, rf_rec, pr_thresh = precision_recall_curve(c_test, rf_probs[:,1])
                        rf_ap[cc,p,k] = average_precision_score(c_test, rf_probs[:,1])
                        rf_pre[:,cc,p,k] = rf_s.predict(t_1)
                        rf_fold_tn_k[cc,p,k], rf_fold_fp_k[cc,p,k], rf_fold_fn_k[cc,p,k], rf_fold_tp_k[cc,p,k] = confusion_matrix(c_test, rf_pre[:,cc,p,k]).ravel()
                        for w in range (len(rf_pre)):
                            if (rf_pre[w,cc,p,k]==c_test[w]):
                                if (((rf_pre[w,cc,p,k]==1)&(c_test[w]==1))):
                                    
                                    rf_info[w,0]=1 #TP
                                else:
                                    
                                    rf_info[w,1]=1 #TN
                            else:
                                if ((c_test[w]==1)):
                                    
                                    rf_info[w,2]=1 #FN
                                else:
                                    
                                    rf_info[w,3]=1 #FP 
                            
                        rf_info[:,4]=number_test
                        rf_info[:,5]=vt_test
                        rf_all_info[cc,p,k,:,:]=rf_info
                        rf_RandA[cc,p,k]=roc_auc_score(c_test, rf_pre[:,cc,p,k])
                        rf_prec2, rf_rec2, _ = precision_recall_curve(c_test, rf_pre[:,cc,p,k])
                        rf_pr_auc[cc,p,k] = auc(rf_rec2, rf_prec2)   # trapezoidal PR AUC
                        rf_ap_one [cc,p,k] = average_precision_score(c_test, rf_pre[:,cc,p,k])
                        rf_fold_ts_k[cc,p,k]=rf_fold_tp_k[cc,p,k]/(rf_fold_tp_k[cc,p,k]+rf_fold_fp_k[cc,p,k]+rf_fold_fn_k[cc,p,k]);
                        
                        #XGB
                        xgb_s=xgb_model.fit(t_2,c_train)
                        xgb_probs = xgb_s.predict_proba(t_1)
                        xgb_fpr, xgb_tpr, roc_thresh = roc_curve(c_test, xgb_probs[:,1])
                        xgb_roc_auc[cc,p,k]= auc(xgb_fpr, xgb_tpr)
                        xgb_prec, xgb_rec, pr_thresh = precision_recall_curve(c_test, xgb_probs[:,1])
                        xgb_ap[cc,p,k] = average_precision_score(c_test, xgb_probs[:,1])
                        xgb_pre[:,cc,p,k] = xgb_s.predict(t_1)
                        xgb_fold_tn_k[cc,p,k], xgb_fold_fp_k[cc,p,k], xgb_fold_fn_k[cc,p,k], xgb_fold_tp_k[cc,p,k] = confusion_matrix(c_test, xgb_pre[:,cc,p,k]).ravel()
                        
                        for w in range (len(xgb_pre)):
                            if (xgb_pre[w,cc,p,k]==c_test[w]):
                                if (((xgb_pre[w,cc,p,k]==1)&(c_test[w]==1))):
                                    
                                    xgb_info[w,0]=1 #TP
                                else:
                                    
                                    xgb_info[w,1]=1 #TN
                            else:
                                if ((c_test[w]==1)):
                                    
                                    xgb_info[w,2]=1 #FN
                                else:
                                    
                                    xgb_info[w,3]=1 #FP 
                            
                        xgb_info[:,4]=number_test
                        xgb_info[:,5]=vt_test
                        xgb_all_info[cc,p,k,:,:]=xgb_info
                        xgb_RandA[cc,p,k]=roc_auc_score(c_test, xgb_pre[:,cc,p,k])
                        xgb_prec2, xgb_rec2, _ = precision_recall_curve(c_test, xgb_pre[:,cc,p,k])
                        xgb_pr_auc[cc,p,k] = auc(xgb_rec2, xgb_prec2)   # trapezoidal PR AUC
                        xgb_ap_one [cc,p,k] = average_precision_score(c_test, xgb_pre[:,cc,p,k])
                        xgb_fold_ts_k[cc,p,k]=xgb_fold_tp_k[cc,p,k]/(xgb_fold_tp_k[cc,p,k]+xgb_fold_fp_k[cc,p,k]+xgb_fold_fn_k[cc,p,k]);
                        
                        
                        
                
                      
                
    
                    #LM
            
                    lm_p_RandA[cc,p]=statistics.mean(lm_RandA[cc,p, :])
                    lm_p_pr_auc[cc,p] =statistics.mean(lm_pr_auc[cc,p, :])
                    lm_p_ap_one[cc,p]=statistics.mean(lm_ap_one[cc,p,:])
                    lm_p_roc_auc[cc,p]=statistics.mean(lm_roc_auc[cc,p, :])
                    lm_p_ap[cc,p] =statistics.mean(lm_ap[cc,p, :])
                    lm_p_ts_p[cc,p]=statistics.mean(lm_fold_ts_k[cc,p, :])
                    lm_p_tp_p[cc,p]=statistics.mean(lm_fold_tp_k[cc,p, :])
                    lm_p_fn_p[cc,p]=statistics.mean(lm_fold_fn_k[cc,p, :])
                    lm_p_fp_p[cc,p]=statistics.mean(lm_fold_fp_k[cc,p, :])
                    lm_p_tn_p[cc,p]=statistics.mean(lm_fold_tn_k[cc,p, :])
                    
                    #LDA 
                    lda_p_RandA[cc,p]=statistics.mean(lda_RandA[cc,p, :])
                    lda_p_pr_auc[cc,p] =statistics.mean(lda_pr_auc[cc,p, :])
                    lda_p_ap_one[cc,p]=statistics.mean(lda_ap_one[cc,p,:])
                    lda_p_roc_auc[cc,p]=statistics.mean(lda_roc_auc[cc,p, :])
                    lda_p_ap[cc,p] =statistics.mean(lda_ap[cc,p, :])
                    lda_p_ts_p[cc,p]=statistics.mean(lda_fold_ts_k[cc,p, :])
                    lda_p_tp_p[cc,p]=statistics.mean(lda_fold_tp_k[cc,p, :])
                    lda_p_fn_p[cc,p]=statistics.mean(lda_fold_fn_k[cc,p, :])
                    lda_p_fp_p[cc,p]=statistics.mean(lda_fold_fp_k[cc,p, :])
                    lda_p_tn_p[cc,p]=statistics.mean(lda_fold_tn_k[cc,p, :])
                    
                    
                    #RF
                    rf_p_RandA[cc,p]=statistics.mean(rf_RandA[cc,p, :])
                    rf_p_pr_auc[cc,p] =statistics.mean(rf_pr_auc[cc,p, :])
                    rf_p_ap_one[cc,p]=statistics.mean(rf_ap_one[cc,p,:])
                    rf_p_roc_auc[cc,p]=statistics.mean(rf_roc_auc[cc,p, :])
                    rf_p_ap[cc,p] =statistics.mean(rf_ap[cc,p, :])
                    rf_p_ts_p[cc,p]=statistics.mean(rf_fold_ts_k[cc,p, :])
                    rf_p_tp_p[cc,p]=statistics.mean(rf_fold_tp_k[cc,p, :])
                    rf_p_fn_p[cc,p]=statistics.mean(rf_fold_fn_k[cc,p, :])
                    rf_p_fp_p[cc,p]=statistics.mean(rf_fold_fp_k[cc,p, :])
                    rf_p_tn_p[cc,p]=statistics.mean(rf_fold_tn_k[cc,p, :])
                    
                    
                    #XGB
                    xgb_p_RandA[cc,p]=statistics.mean(xgb_RandA[cc,p, :])
                    xgb_p_pr_auc[cc,p] =statistics.mean(xgb_pr_auc[cc,p, :])
                    xgb_p_ap_one[cc,p]=statistics.mean(xgb_ap_one[cc,p,:])
                    xgb_p_roc_auc[cc,p]=statistics.mean(xgb_roc_auc[cc,p, :])
                    xgb_p_ap[cc,p] =statistics.mean(xgb_ap[cc,p, :])
                    xgb_p_ts_p[cc,p]=statistics.mean(xgb_fold_ts_k[cc,p, :])
                    xgb_p_tp_p[cc,p]=statistics.mean(xgb_fold_tp_k[cc,p, :])
                    xgb_p_fn_p[cc,p]=statistics.mean(xgb_fold_fn_k[cc,p, :])
                    xgb_p_fp_p[cc,p]=statistics.mean(xgb_fold_fp_k[cc,p, :])
                    xgb_p_tn_p[cc,p]=statistics.mean(xgb_fold_tn_k[cc,p, :])
                    
                    

    lm_ts_median[cc]=statistics.median(lm_p_ts_p[cc,:])
    lm_ts_mean[cc]=statistics.mean(lm_p_ts_p[cc,:])
    lm_tp_m[cc]=statistics.median(lm_p_tp_p[cc,:])
    lm_fp_m[cc]=statistics.median(lm_p_fp_p[cc,:])
    lm_fn_m[cc]=statistics.median(lm_p_fn_p[cc,:])
    lm_tn_m[cc]=statistics.median(lm_p_tn_p[cc,:])
    lm_pi_m[cc]=(lm_tp_m[cc]+lm_fn_m[cc])/(lm_tp_m[cc]+lm_fn_m[cc] +lm_tn_m[cc]+lm_fp_m[cc])
    lm_ra[cc]=statistics.median(lm_p_RandA[cc,:])
    lm_roc_auc_m[cc]=statistics.median(lm_p_roc_auc[cc,:])
    lm_ap_m[cc]=statistics.median(lm_p_ap[cc,:])
    lm_ap_one_m[cc]=statistics.median(lm_p_ap_one[cc,:])
    lm_pr2[cc]=statistics.median(lm_p_pr_auc[cc,:])
    lm_positive_predictive_value_or_precision[cc]=lm_tp_m[cc]/(lm_tp_m[cc]+lm_fp_m[cc])
    lm_false_positive_rate[cc]=lm_fp_m[cc]/(lm_fp_m[cc]+lm_tn_m[cc])
    lm_true_negative_rate_or_specificity[cc]=lm_tn_m[cc]/(lm_tn_m[cc]+lm_fp_m[cc])
    lm_positive_predictive_value[cc]=lm_tp_m[cc]/(lm_tp_m[cc]+lm_fp_m[cc])
    lm_negative_predicted_value[cc]=lm_tn_m[cc]/(lm_tn_m[cc]+lm_fn_m[cc])
    lm_recall_or_sensitivity_or_true_positive_rate[cc]=lm_tp_m[cc]/(lm_tp_m[cc]+lm_fn_m[cc])
    lm_f1_score[cc]=(2*lm_recall_or_sensitivity_or_true_positive_rate[cc]*lm_positive_predictive_value_or_precision[cc])/(lm_positive_predictive_value_or_precision[cc]+lm_recall_or_sensitivity_or_true_positive_rate[cc])




    #LDA
    lda_ts_median[cc]=statistics.median(lda_p_ts_p[cc,:])
    lda_ts_mean[cc]=statistics.mean(lda_p_ts_p[cc,:])
    lda_tp_m[cc]=statistics.median(lda_p_tp_p[cc,:])
    lda_fp_m[cc]=statistics.median(lda_p_fp_p[cc,:])
    lda_fn_m[cc]=statistics.median(lda_p_fn_p[cc,:])
    lda_tn_m[cc]=statistics.median(lda_p_tn_p[cc,:])
    lda_pi_m[cc]=(lda_tp_m[cc]+lda_fn_m[cc])/(lda_tp_m[cc]+lda_fn_m[cc] +lda_tn_m[cc]+lda_fp_m[cc])
    lda_ra[cc]=statistics.median(lda_p_RandA[cc,:])
    lda_roc_auc_m[cc]=statistics.median(lda_p_roc_auc[cc,:])
    lda_ap_m[cc]=statistics.median(lda_p_ap[cc,:])
    lda_ap_one_m[cc]=statistics.median(lda_p_ap_one[cc,:])
    lda_pr2[cc]=statistics.median(lda_p_pr_auc[cc,:])
    lda_positive_predictive_value_or_precision[cc]=lda_tp_m[cc]/(lda_tp_m[cc]+lda_fp_m[cc])
    lda_false_positive_rate[cc]=lda_fp_m[cc]/(lda_fp_m[cc]+lda_tn_m[cc])
    lda_true_negative_rate_or_specificity[cc]=lda_tn_m[cc]/(lda_tn_m[cc]+lda_fp_m[cc])
    lda_positive_predictive_value[cc]=lda_tp_m[cc]/(lda_tp_m[cc]+lda_fp_m[cc])
    lda_negative_predicted_value[cc]=lda_tn_m[cc]/(lda_tn_m[cc]+lda_fn_m[cc])
    lda_recall_or_sensitivity_or_true_positive_rate[cc]=lda_tp_m[cc]/(lda_tp_m[cc]+lda_fn_m[cc])
    lda_f1_score[cc]=(2*lda_recall_or_sensitivity_or_true_positive_rate[cc]*lda_positive_predictive_value_or_precision[cc])/(lda_positive_predictive_value_or_precision[cc]+lda_recall_or_sensitivity_or_true_positive_rate[cc])


    #RF
    rf_ts_median[cc]=statistics.median(rf_p_ts_p[cc,:])
    rf_ts_mean[cc]=statistics.mean(rf_p_ts_p[cc,:])
    rf_tp_m[cc]=statistics.median(rf_p_tp_p[cc,:])
    rf_fp_m[cc]=statistics.median(rf_p_fp_p[cc,:])
    rf_fn_m[cc]=statistics.median(rf_p_fn_p[cc,:])
    rf_tn_m[cc]=statistics.median(rf_p_tn_p[cc,:])
    rf_pi_m[cc]=(rf_tp_m[cc]+rf_fn_m[cc])/(rf_tp_m[cc]+rf_fn_m[cc] +rf_tn_m[cc]+rf_fp_m[cc])
    rf_ra[cc]=statistics.median(rf_p_RandA[cc,:])
    rf_roc_auc_m[cc]=statistics.median(rf_p_roc_auc[cc,:])
    rf_ap_m[cc]=statistics.median(rf_p_ap[cc,:])
    rf_ap_one_m[cc]=statistics.median(rf_p_ap_one[cc,:])
    rf_pr2[cc]=statistics.median(rf_p_pr_auc[cc,:])
    rf_positive_predictive_value_or_precision[cc]=rf_tp_m[cc]/(rf_tp_m[cc]+rf_fp_m[cc])
    rf_false_positive_rate[cc]=rf_fp_m[cc]/(rf_fp_m[cc]+rf_tn_m[cc])
    rf_true_negative_rate_or_specificity[cc]=rf_tn_m[cc]/(rf_tn_m[cc]+rf_fp_m[cc])
    rf_positive_predictive_value[cc]=rf_tp_m[cc]/(rf_tp_m[cc]+rf_fp_m[cc])
    rf_negative_predicted_value[cc]=rf_tn_m[cc]/(rf_tn_m[cc]+rf_fn_m[cc])
    rf_recall_or_sensitivity_or_true_positive_rate[cc]=rf_tp_m[cc]/(rf_tp_m[cc]+rf_fn_m[cc])
    rf_f1_score[cc]=(2*rf_recall_or_sensitivity_or_true_positive_rate[cc]*rf_positive_predictive_value_or_precision[cc])/(rf_positive_predictive_value_or_precision[cc]+rf_recall_or_sensitivity_or_true_positive_rate[cc])

    #XGB
    xgb_ts_median[cc]=statistics.median(xgb_p_ts_p[cc,:])
    xgb_ts_mean[cc]=statistics.mean(xgb_p_ts_p[cc,:])
    xgb_tp_m[cc]=statistics.median(xgb_p_tp_p[cc,:])
    xgb_fp_m[cc]=statistics.median(xgb_p_fp_p[cc,:])
    xgb_fn_m[cc]=statistics.median(xgb_p_fn_p[cc,:])
    xgb_tn_m[cc]=statistics.median(xgb_p_tn_p[cc,:])
    xgb_pi_m[cc]=(xgb_tp_m[cc]+xgb_fn_m[cc])/(xgb_tp_m[cc]+xgb_fn_m[cc] +xgb_tn_m[cc]+xgb_fp_m[cc])
    xgb_ra[cc]=statistics.median(xgb_p_RandA[cc,:])
    xgb_roc_auc_m[cc]=statistics.median(xgb_p_roc_auc[cc,:])
    xgb_ap_m[cc]=statistics.median(xgb_p_ap[cc,:])
    xgb_ap_one_m[cc]=statistics.median(xgb_p_ap_one[cc,:])
    xgb_pr2[cc]=statistics.median(xgb_p_pr_auc[cc,:])
    xgb_positive_predictive_value_or_precision[cc]=xgb_tp_m[cc]/(xgb_tp_m[cc]+xgb_fp_m[cc])
    xgb_false_positive_rate[cc]=xgb_fp_m[cc]/(xgb_fp_m[cc]+xgb_tn_m[cc])
    xgb_true_negative_rate_or_specificity[cc]=xgb_tn_m[cc]/(xgb_tn_m[cc]+xgb_fp_m[cc])
    xgb_positive_predictive_value[cc]=xgb_tp_m[cc]/(xgb_tp_m[cc]+xgb_fp_m[cc])
    xgb_negative_predicted_value[cc]=xgb_tn_m[cc]/(xgb_tn_m[cc]+xgb_fn_m[cc])
    xgb_recall_or_sensitivity_or_true_positive_rate[cc]=xgb_tp_m[cc]/(xgb_tp_m[cc]+xgb_fn_m[cc])
    xgb_f1_score[cc]=(2*xgb_recall_or_sensitivity_or_true_positive_rate[cc]*xgb_positive_predictive_value_or_precision[cc])/(xgb_positive_predictive_value_or_precision[cc]+xgb_recall_or_sensitivity_or_true_positive_rate[cc])

lm=np.column_stack((lm_ts_median, lm_recall_or_sensitivity_or_true_positive_rate,lm_false_positive_rate, lm_true_negative_rate_or_specificity,lm_positive_predictive_value_or_precision,     lm_f1_score, lm_pi_m, lm_ap_m, lm_ap_one_m, lm_ra, lm_roc_auc_m))
lda=np.column_stack((lda_ts_median, lda_recall_or_sensitivity_or_true_positive_rate,lda_false_positive_rate, lda_true_negative_rate_or_specificity,lda_positive_predictive_value_or_precision,     lda_f1_score, lda_pi_m, lda_ap_m, lda_ap_one_m, lda_ra, lda_roc_auc_m))
rf=np.column_stack((rf_ts_median, rf_recall_or_sensitivity_or_true_positive_rate,rf_false_positive_rate, rf_true_negative_rate_or_specificity,rf_positive_predictive_value_or_precision,     rf_f1_score, rf_pi_m, rf_ap_m, rf_ap_one_m, rf_ra, rf_roc_auc_m))
xgb=np.column_stack((xgb_ts_median, xgb_recall_or_sensitivity_or_true_positive_rate,xgb_false_positive_rate, xgb_true_negative_rate_or_specificity,xgb_positive_predictive_value_or_precision,     xgb_f1_score, xgb_pi_m, xgb_ap_m, xgb_ap_one_m, xgb_ra, xgb_roc_auc_m))
                   

