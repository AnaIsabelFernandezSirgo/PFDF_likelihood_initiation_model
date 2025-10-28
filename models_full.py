# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 06:34:34 2025

@author: Ana Isabel Fernandez Sirgo 
"""

import pandas as pd
import numpy as np
from itertools import product
import statistics
from sklearn.linear_model import LogisticRegression
#they will get used when you switch out model 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")

# # set for logistic regresion sqrt weight i15


#DOWNLOADING DATA MULTIPLIED BY RAIN 
data_rained=pd.read_excel(r'DATA', sheet_name=[ 'Sheet2','Sheet3','Sheet4' ,'Sheet5','Sheet6','Sheet7'])

i2_s=data_rained['Sheet2']
i5_s=data_rained['Sheet3']
i10_s=data_rained['Sheet4']
i15_s=data_rained['Sheet5']
i30_s=data_rained['Sheet6']
i60_s=data_rained['Sheet7']


#SEPARATING DATA INTO CATEGORIES

response=i2_s[['Response']].copy()
vtype=i2_s[['No']].copy()
number=i2_s[['VT']].copy()



re=response.to_numpy()

topography_col=['Ruggedness_S', 'Slope_mean', 's23', 's23MH', 's23H','a2000s15', 
                     'a2000s15MH', 'a2000s15H', 'a2000s23', 'a2000s23MH','a2000s23H','MeanSlopeLMH', 'MeanSlopeMH',
                     'S23LMH', 'S30LMH' ,'S30MH']

t_i2=i2_s[topography_col].to_numpy(copy=True)
t_i5=i5_s[topography_col].to_numpy(copy=True)
t_i10=i10_s[topography_col].to_numpy(copy=True)
t_i15=i15_s[topography_col].to_numpy(copy=True)
t_i30=i30_s[topography_col].to_numpy(copy=True)
t_i60 = i60_s[topography_col].to_numpy(copy=True)

fire_col=['mdNBR_1000', 'MH', 'L_2_MH', 'L_3_MH', 'L_4_MH']

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
yes_total=np.sum(re)
no_train=round(no_total*percent)
yes_train=round(np.sum(re)*percent)
no_test=no_total-no_train
yes_test=yes_total-yes_train
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


test=np.empty((length_test,5,number_combo,number_permutation,number_kfolds))
train=np.empty((length_train,5,number_combo,number_permutation,number_kfolds));
 


#create the test train matrices with different (extra: i2, i5) i10, i15, i30 or i60

for cc in range (number_combo):
    for p in range (number_permutation):
        for k in range (number_kfolds):
                ver=i15[:,:,cc]
                

                test[:,:,cc,p,k]=ver[matrix_test[:,p,k],:];
                train[:,:,cc,p,k]=ver[matrix_train[:,p,k],:];
                


fold_ts_k=np.empty((number_combo,number_permutation,number_kfolds ))
fold_tp_k=np.empty((number_combo,number_permutation,number_kfolds))
fold_fn_k=np.empty((number_combo,number_permutation,number_kfolds))
fold_fp_k=np.empty((number_combo,number_permutation,number_kfolds))
fold_tn_k=np.empty((number_combo,number_permutation,number_kfolds))
fold_number=np.empty((number_combo,number_permutation,number_kfolds))
fold_recall_k=np.empty((number_combo,number_permutation,number_kfolds))
fold_precision_k=np.empty((number_combo,number_permutation,number_kfolds))
fold_f1_k=np.empty((number_combo,number_permutation,number_kfolds))
all_info=np.empty((number_combo, number_permutation, number_kfolds, length_test,6))
info=np.empty((length_test,6))


p_ts_p=np.empty((number_combo,number_permutation))
p_tp_p=np.empty((number_combo,number_permutation))
p_fn_p=np.empty((number_combo,number_permutation))
p_fp_p=np.empty((number_combo,number_permutation))
p_tn_p=np.empty((number_combo,number_permutation))
p_f1_p=np.empty((number_combo,number_permutation))
p_precision_p=np.empty((number_combo,number_permutation))
p_recall_p=np.empty((number_combo,number_permutation))




p_ts_p_m=np.empty((number_combo,number_permutation))
p_tp_p_m=np.empty((number_combo,number_permutation))
p_fn_p_m=np.empty((number_combo,number_permutation))
p_fp_p_m=np.empty((number_combo,number_permutation))
p_tn_p_m=np.empty((number_combo,number_permutation))
p_f1_m=np.empty((number_combo,number_permutation))
p_precision_m=np.empty((number_combo,number_permutation))
p_recall_m=np.empty((number_combo,number_permutation))




standard_fold=np.empty((number_combo,number_permutation))
ts_median=np.empty((number_combo,1)) 
ts_mean=np.empty((number_combo,1))
 
f1_median=np.empty((number_combo,1)) 
f1_mean=np.empty((number_combo,1))
precision_median=np.empty((number_combo,1)) 
precision_mean=np.empty((number_combo,1))
recall_median=np.empty((number_combo,1)) 
recall_mean=np.empty((number_combo,1))

ts_m=np.empty((number_combo,8))
ts_median_diviation_fold= np.empty((number_combo,1)) 

tp_mean=np.empty((number_combo,1))
fp_mean=np.empty((number_combo,1))
fn_mean=np.empty((number_combo,1))
tn_mean=np.empty((number_combo,1))
tp_median=np.empty((number_combo,1))
fp_median=np.empty((number_combo,1))
fn_median=np.empty((number_combo,1))
tn_median=np.empty((number_combo,1))


tp=0;
fn=0;
fp=0;
tn=0;

c=-1;
p=0
cc=0
k=0
pre=np.empty((length_test,number_combo,number_permutation,number_kfolds ))


# Model options
# # Choose only one option it will give 80 models on each  differnent weight and different algorithm  12 options 
# # set for logistic regresion sqrt weight i15

#none weight:
    
#model = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000, fit_intercept=True)

#model = LinearDiscriminantAnalysis( ) 


# #RF
# model = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=3,
#    min_samples_split=10,
#    min_samples_leaf=5,
#     max_features=2,
#     bootstrap=True,
#     max_samples=.9,         # sub-sample rows per tree
#     oob_score=True,
    
#     random_state=0,
#     n_jobs=-1
# )

# #XGB
# n_features =2
# mono = [1, 1] + [0]*(n_features-2)
# model = xgb.XGBClassifier(
#     booster='gbtree',
#     objective='binary:logistic',
# learning_rate=0.05,
# max_depth=3,               # shallower = less variance on ~1k rows
# n_estimators=150,         # big cap 3000
# min_child_weight=10,       # block brittle splits
# gamma=2,                   # penalty on weak gains
# reg_lambda=12,             # a bit more L2
# subsample=1.0,
# colsample_bytree=1.0,
# colsample_bylevel=1.0,
# colsample_bynode=1.0,
#     monotone_constraints=tuple(mono),  # must match X’s column order/length
#     random_state=0,
#     n_jobs=-1
# )






   
#balanced weight 
# LR
#model = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000,class_weight='balanced', fit_intercept=True)
# #  LDA convert that into class priors
# w = (no_train/yes_train)
# pi_pos = (w * yes_train) / (no_train + w * yes_train)
# pi_neg = 1.0 - pi_pos
# model = LinearDiscriminantAnalysis( priors=[pi_neg, pi_pos]) 
# #RF
# model = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=3,
#    min_samples_split=10,
#    min_samples_leaf=5,
#     max_features=2,
#     bootstrap=True,
#     max_samples=.9,         # sub-sample rows per tree
#     oob_score=True,
#     class_weight={0:1, 1:(no_train/yes_train)},  # keep your scheme
#     random_state=0,
#     n_jobs=-1
# )
# #XGB
# n_features =2
# mono = [1, 1] + [0]*(n_features-2)
#model = xgb.XGBClassifier(
#     booster='gbtree',
#     objective='binary:logistic',
# learning_rate=0.05,
# max_depth=3,               # shallower = less variance on ~1k rows
# n_estimators=150,         # big cap 3000
# min_child_weight=10,       # block brittle splits
# gamma=2,                   # penalty on weak gains
# reg_lambda=12,             # a bit more L2
# subsample=1.0,
# colsample_bytree=1.0,
# colsample_bylevel=1.0,
# colsample_bynode=1.0,
#     scale_pos_weight=(no_train/yes_train),
#     monotone_constraints=tuple(mono),  # must match X’s column order/length
#     random_state=0,
#     n_jobs=-1
# )







    
#sqrt weight



# #LR
model = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000,class_weight={0:1, 1:np.sqrt(no_train/yes_train)}, fit_intercept=True)

# # LDA
# #your weighting factor
# w = np.sqrt(no_train/yes_train)

# LDA convert that into class priors
# pi_pos = (w * yes_train) / (no_train + w * yes_train)
# pi_neg = 1.0 - pi_pos
#model = LinearDiscriminantAnalysis( priors=[pi_neg, pi_pos]) 
# # RF
# model = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=3,
#    min_samples_split=10,
#    min_samples_leaf=5,
#     max_features=2,
#     bootstrap=True,
#     max_samples=.9,         # sub-sample rows per tree
#     oob_score=True,
#     class_weight={0:1, 1:np.sqrt(no_train/yes_train)},  # keep your scheme
#     random_state=0,
#     n_jobs=-1
# )

# XGB

# n_features =2
# mono = [1, 1] + [0]*(n_features-2)
# model = xgb.XGBClassifier(
#     booster='gbtree',
#     objective='binary:logistic',
#     learning_rate=0.05,
#     max_depth=3,               # shallower = less variance on ~1k rows
#     n_estimators=150,         # big cap 3000
#     min_child_weight=10,       # block brittle splits
#     gamma=2,                   # penalty on weak gains
#     reg_lambda=12,             # a bit more L2
#     subsample=1.0,
#     colsample_bytree=1.0,
#     colsample_bylevel=1.0,
#     colsample_bynode=1.0,
#     scale_pos_weight=np.sqrt(no_train/yes_train),
#     monotone_constraints=tuple(mono),  # must match X’s column order/length
#     random_state=0,
#     n_jobs=-1
# )



# # set for logistic regresion sqrt weight i15

for cc in range (number_combo):
    c=c+1;
    #np.random.seed(13)
    for p in range (number_permutation):
                    
                    
                    for k in range (number_kfolds):
                     
                        
                        Test=test[:,:,cc,p,k]
                        Train=train[:,:,cc,p,k]
                        # Test2=test2[:,:,cc,p,k]
                        # Train2=train2[:,:,cc,p,k]
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
                        
                        
                        s=model.fit(t_2,c_train)
                        pre[:,cc,p,k] = s.predict(t_1)
                        
                        
                        for w in range (len(pre)):
                            if (pre[w,cc,p,k]==c_test[w]):
                                if (((pre[w,cc,p,k]==1)&(c_test[w]==1))):
                                    tp= tp+1
                                    info[w,0]=1
                                else:
                                    tn=tn+1
                                    info[w,1]=1
                            else:
                                if ((c_test[w]==1)):
                                    fn=fn+1
                                    info[w,2]=1
                                else:
                                    fp=fp+1
                                    info[w,3]=1
                        info[:,4]=number_test
                        info[:,5]=vt_test
                        fold_tp_k[cc,p,k]=tp;
                        fold_fn_k[cc,p,k]=fn;
                        fold_fp_k[cc,p,k]=fp;
                        fold_tn_k[cc,p,k]=tn;
                        all_info[cc,p,k,:,:]=info
                        
                        
                        
                        tp=0;
                        fn=0;
                        fp=0;
                        tn=0;

                        fold_ts_k[cc,p,k]=fold_tp_k[cc,p,k]/(fold_tp_k[cc,p,k]+fold_fp_k[cc,p,k]+fold_fn_k[cc,p,k]);
                        fold_recall_k[cc,p,k]=fold_tp_k[cc,p,k]/(fold_tp_k[cc,p,k]+fold_fn_k[cc,p,k])
                        fold_precision_k[cc,p,k]=fold_tp_k[cc,p,k]/(fold_tp_k[cc,p,k]+fold_fp_k[cc,p,k])
                        fold_f1_k[cc,p,k]=(2*fold_precision_k[cc,p,k]*fold_recall_k[cc,p,k])/(fold_recall_k[cc,p,k]+fold_precision_k[cc,p,k])
                        
                    standard_fold[cc,p]=statistics.stdev(fold_ts_k[cc,p,:])
                    p_ts_p[cc,p]=sum(fold_ts_k[cc,p, :])/number_kfolds
                    p_f1_p[cc,p]=sum(fold_f1_k[cc,p, :])/number_kfolds
                    p_precision_p[cc,p]=sum(fold_precision_k[cc,p, :])/number_kfolds
                    p_recall_p[cc,p]=sum(fold_recall_k[cc,p, :])/number_kfolds
                    p_tp_p[cc,p]=sum(fold_tp_k[cc,p, :])/number_kfolds
                    p_fn_p[cc,p]=sum(fold_fn_k[cc,p, :])/number_kfolds
                    p_fp_p[cc,p]=sum(fold_fp_k[cc,p, :])/number_kfolds
                    p_tn_p[cc,p]=sum(fold_tn_k[cc,p, :])/number_kfolds
                    
                    p_ts_p_m[cc,p]=statistics.median(fold_ts_k[cc,p, :])
                    p_f1_m[cc,p]=statistics.median(fold_f1_k[cc,p, :])
                    p_precision_m[cc,p]=statistics.median(fold_precision_k[cc,p, :])
                    p_recall_m[cc,p]=statistics.median(fold_recall_k[cc,p, :])
                    p_tp_p_m[cc,p]=statistics.median(fold_tp_k[cc,p, :])
                    p_fn_p_m[cc,p]=statistics.median(fold_fn_k[cc,p, :])
                    p_fp_p_m[cc,p]=statistics.median(fold_fp_k[cc,p, :])
                    p_tn_p_m[cc,p]=statistics.median(fold_tn_k[cc,p, :])


    ts_median_diviation_fold[cc]=statistics.median(standard_fold[cc,:]);
    ts_median[cc]=statistics.median(p_ts_p[cc,:])
    ts_mean[cc]=statistics.mean(p_ts_p[cc,:])
    f1_mean[cc]=statistics.mean(p_f1_p[cc,:])
    f1_median[cc]=statistics.median(p_f1_p[cc,:])
    precision_mean[cc]=statistics.mean(p_precision_p[cc,:])
    precision_median[cc]=statistics.median(p_precision_p[cc,:])
    recall_mean[cc]=statistics.mean(p_recall_p[cc,:])
    recall_median[cc]=statistics.median(p_recall_p[cc,:])
    
    
    
    tp_median[cc]=statistics.median(p_tp_p[cc,:])
    fp_median[cc]=statistics.median(p_fp_p[cc,:])
    fn_median[cc]=statistics.median(p_fn_p[cc,:])
    tn_median[cc]=statistics.median(p_tn_p[cc,:])
    tp_mean[cc]=statistics.mean(p_tp_p[cc,:])
    fp_mean[cc]=statistics.mean(p_fp_p[cc,:])
    fn_mean[cc]=statistics.mean(p_fn_p[cc,:])
    tn_mean[cc]=statistics.mean(p_tn_p[cc,:])
    
    
    
index_median=np.array(sorted(range(len(ts_median)), key=lambda k: ts_median[k]))
                
                
    
ts_sort_median=np.array(sorted(ts_median[:,0]))


TS=np.column_stack((ts_sort_median, index_median))
ts = TS[::-1]

df = pd.DataFrame(ts)