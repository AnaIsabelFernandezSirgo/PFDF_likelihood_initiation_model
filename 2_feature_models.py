# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:02:20 2025

@author: 15206

"""


import pandas as pd
import numpy as np


from xgboost import XGBClassifier

import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings("ignore")
#this needs to be saved by you use file data_2_feature_matrices 
#(check names) 
#options of data 
#i10_2.npy
#i15_2.npy
#i30_2.npy
#i60_2.npy

data_rained= np.load("i30_2.npy")
# permutation matrices seed
matrix_train=pd.read_csv(r"D:\Fire_Mar24\train_4k.csv",  header=None)
matrix_test=pd.read_csv(r"D:\Fire_Mar24\test_4k.csv",  header=None)

m_test=matrix_test.to_numpy()
m_train=matrix_train.to_numpy()
m_test=m_test-1
m_train=m_train-1


len_test=int(m_test.size/10)
len_train=int(m_train.size/10)
#preallocation
mk=4
mf=5
mt=16
mr=4
mp=10 #permutation
mc= mf*mt
test=np.zeros((len_test,3,mc,mp))
train=np.zeros((len_train,3,mc,mp));
z=np.zeros((975,1));
o=np.ones((122,1));
r_train=np.concatenate((z, o,z, o,z, o,z, o));
r_train=np.append(r_train, r_train[0])
z=np.zeros((325,1));
o=np.ones((41,1));



r_test=np.concatenate((z, o, z, o, z, o, z, o ));
r_test=np.append(r_test, r_test[0])
#data to use 
for cc in range (mc):
    for p in range (mp):

                ver=data_rained[:,:,cc]
               

                test[:,:,cc,p]=ver[m_test[:,p],:];
                train[:,:,cc,p]=ver[m_train[:,p],:];
                


fold_ts_k=np.zeros((mc,mp,mk))
fold_tp_k=np.zeros((mc,mp,mk))
fold_fn_k=np.zeros((mc,mp,mk))
fold_fp_k=np.zeros((mc,mp,mk))
fold_tn_k=np.zeros((mc,mp,mk))
p_ts_p=np.zeros((mc,mp))
p_tp_p=np.zeros((mc,mp))
p_fn_p=np.zeros((mc,mp))
p_fp_p=np.zeros((mc,mp))
p_tn_p=np.zeros((mc,mp))

p_ts_p_m=np.zeros((mc,mp))
p_tp_p_m=np.zeros((mc,mp))
p_fn_p_m=np.zeros((mc,mp))
p_fp_p_m=np.zeros((mc,mp))
p_tn_p_m=np.zeros((mc,mp))
standard_fold=np.zeros((mc,mp))

length_test=int((len_test+1)/mk)
length_train=int((len_train+1)/mk)

tp=0;
fn=0;
fp=0;
tn=0;

c=-1;
p=0
cc=0
k=0
prediction=np.zeros((length_test,mc,mp,mk ))
#Model options
 
#none weight:
    
# model = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150)
# model = LinearDiscriminantAnalysis()   
# model= RandomForestClassifier( n_estimators=100, random_state=0,max_depth=6)    
# model = xgb.XGBClassifier(booster='gbtree', 
# objective='binary:logistic',
#           use_label_encoder=False, 
#           eval_metric='error',
#           gamma=0,
#           learning_rate=0.05,
#           max_depth=6, 
#           reg_lambda=10,
#           subsample=.9,
#           colsample_bytree=.9, monotone_constraints = (1,1,1))    
#balanced weight 

#model = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150, max_iter= 150, class_weight='balanced' )  
#model= RandomForestClassifier( n_estimators=100, random_state=0,max_depth=6, max_iter= 150, class_weight='balanced' )   
#model = xgb.XGBClassifier(booster='gbtree', 
    # objective='binary:logistic',
    #           use_label_encoder=False, 
    #           eval_metric='error',
    #           gamma=0,
    #           learning_rate=0.05,
    #           max_depth=6, 
    #           reg_lambda=10,
    #           subsample=.9,
    #           scale_pos_weight=(975/122),
    #           colsample_bytree=.9, monotone_constraints = (1,1,1))
    
#sqrt weight
#model = LogisticRegression(penalty=None, solver = 'newton-cg', class_weight={0:1, 1:np.sqrt(975/122)}))   
#model= RandomForestClassifier( n_estimators=100, random_state=0,max_depth=6, class_weight={0:1, 1:np.sqrt(975/122)}) 
#model = xgb.XGBClassifier(booster='gbtree', 
    # objective='binary:logistic',
    #           use_label_encoder=False, 
    #           eval_metric='error',
    #           gamma=0,
    #           learning_rate=0.05,
    #           max_depth=6, 
    #           reg_lambda=10,
    #           subsample=.9,
    #           scale_pos_weight=np.sqrt(975/122),
    #           colsample_bytree=.9, monotone_constraints = (1,1,1))
        
        
model = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150)
                       

for cc in range (mc):
    c=c+1;
    np.random.seed(13)
    for p in range (mp):
                    Test_p=test[:,:,cc,p]
                    Train_p=train[:,:,cc,p]
                    test_resp_p=Test_p[:,0]
                    train_resp_p=Train_p[:,0]
                    for k in range (mk):
                     
                        start=(366*(k))
                        start_b=(length_train*(k))
                        l_train= length_train*(k+1);
                        l_test= length_test*(k+1);
                        
                    
                        
                        
                        Test=Test_p[start: l_test,:];
                        Train=Train_p[start_b:l_train,:];
                        Test=Test[np.random.permutation(len(Test)),:]
                        Train=Train[np.random.permutation(len(Train)),:]
                        c_test=Test[:,0];
                        c_train=Train[:,0];
           

                        t_1=Test[:,1:Test.shape[1]]
                        t_2=Train[:,1:Train.shape[1]]

                        
                        model = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150, class_weight={0:1, 1:(975/(122))})
                                                   
                        s=model.fit(t_2,c_train)
                        prediction[:,cc,p,k] = s.predict(t_1)
                        
                        
                        for w in range (len(prediction)):
                            if (prediction[w,cc,p,k]==c_test[w]):
                                if (((prediction[w,cc,p,k]==1)&(c_test[w]==1))):
                                    tp= tp+1
                                else:
                                    tn=tn+1
                                
                            else:
                                if ((c_test[w]==1)):
                                    fn=fn+1
                                else:
                                    fp=fp+1
                                

                        fold_tp_k[cc,p,k]=tp;
                        fold_fn_k[cc,p,k]=fn;
                        fold_fp_k[cc,p,k]=fp;
                        fold_tn_k[cc,p,k]=tn;


                        tp=0;
                        fn=0;
                        fp=0;
                        tn=0;

                        fold_ts_k[cc,p,k]=fold_tp_k[cc,p,k]/(fold_tp_k[cc,p,k]+fold_fp_k[cc,p,k]+fold_fn_k[cc,p,k]);

                
                    standard_fold[cc,p]=statistics.stdev(fold_ts_k[cc,p,:])
                    p_ts_p[cc,p]=sum(fold_ts_k[cc,p, :])/mk;
                    p_tp_p[cc,p]=sum(fold_tp_k[cc,p, :])/mk;
                    p_fn_p[cc,p]=sum(fold_fn_k[cc,p, :])/mk;
                    p_fp_p[cc,p]=sum(fold_fp_k[cc,p, :])/mk;
                    p_tn_p[cc,p]=sum(fold_tn_k[cc,p, :])/mk;
                    
                    p_ts_p_m[cc,p]=statistics.median(fold_ts_k[cc,p, :])
                    p_tp_p_m[cc,p]=statistics.median(fold_tp_k[cc,p, :])
                    p_fn_p_m[cc,p]=statistics.median(fold_fn_k[cc,p, :])
                    p_fp_p_m[cc,p]=statistics.median(fold_fp_k[cc,p, :])
                    p_tn_p_m[cc,p]=statistics.median(fold_tn_k[cc,p, :])
    
ts_median=np.zeros((mc,1)) 
ts_mean=np.zeros((mc,1))
ts_m=np.zeros((mc,3)) 
ts_median_diviation_fold= np.zeros((mc,1)) 

tp_mean=np.zeros((mc,1))
fp_mean=np.zeros((mc,1))
fn_mean=np.zeros((mc,1))
tn_mean=np.zeros((mc,1))

   
for cc in range (mc):
    ts_median_diviation_fold[cc]=statistics.median(standard_fold[cc,:]);
    ts_median[cc]=statistics.median(p_ts_p[cc,:])
    ts_mean[cc]=statistics.mean(p_ts_p[cc,:])
    tp_mean[cc]=statistics.median(p_tp_p[cc,:])
    fp_mean[cc]=statistics.median(p_fp_p[cc,:])
    fn_mean[cc]=statistics.median(p_fn_p[cc,:])
    tn_mean[cc]=statistics.median(p_tn_p[cc,:])
    
    
    
index_median=np.array(sorted(range(len(ts_median)), key=lambda k: ts_median[k]))
              
                
    
ts_sort_median=np.array(sorted(ts_median[:,0]))


ts_m=np.column_stack((ts_sort_median, index_median))
ts_and_index = ts_m[::-1]

            