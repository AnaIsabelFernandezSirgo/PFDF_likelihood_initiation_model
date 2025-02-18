# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:36:16 2025
@author: 15206
"""

import pandas as pd
import numpy as np
"""
These np.loads come from two_2_feature_models.py
you need to record them I only provide r15 and r30 for all algorithms  
"""
lda_15=np.load("results_lda_15.npy")
lda_30=np.load("results_lda_30.npy")
lm_15=np.load("results_lm_15.npy")
lm_30=np.load("results_lm_30.npy")
rf_15=np.load("results_rf_15.npy")
rf_30=np.load("results_rf_30.npy")
xgb_15=np.load("results_xgb_15.npy")
xgb_30=np.load("results_xgb_30.npy")
xgb_30=np.load("results_xgb_60.npy")

all_6=np.array((lm_15[0,:],lda_15[0,:],rf_15[0,:],xgb_15[0,:], lm_30[0,:],lda_30[0,:],rf_30[0,:],xgb_30[0,:]))
df_6=pd.DataFrame(all_6)

df_6.index=['LM  I15', 'LDA  I15', 'RF  I15', 'XGB  I15', 'LM  I30', 'LDA  I30', 'RF I30', 'XGB I30']
df_6.columns =['TS', "s", 'False Positive Rate', 'True Negative Rate or Specificity ', "Positive Predictive Value or Precision", 'Negative Predicted Value', 'Racall or Sensitivity' , "F1 Score", 'ROC and AUC Score']
df_6=df_6.drop('s',axis =1)
all_8=np.array((lm_15[1,:],lda_15[1,:],rf_15[1,:],xgb_15[1,:], lm_30[1,:],lda_30[1,:],rf_30[1,:],xgb_30[1,:]))
df_8=pd.DataFrame(all_8)
df_8.index=['LM I15', 'LDA I15', 'RF I15', 'XGB I15', 'LM I30', 'LDA I30', 'RF I30', 'XGB 10 I30']
df_8.columns =['TS',  's','False Positive Rate', 'True Negative Rate or Specificity ', "Positive predictive Value", 'Negative Predicted Value', 'Racall or Sensitivity' , "F1 Score", 'ROC and AUC Score']  
df_8=df_8.drop('s',axis =1)

pr=(all_8[:,6]+all_8[:,4])/2
pr_d=pd.DataFrame(pr)

all_15_8=np.array((lm_15[1,:],lda_15[1,:],rf_15[1,:],xgb_15[1,:]))

df1=pd.DataFrame(all_15_8)
df_swa=df1[df1.columns[[0,6,1,2,3,4,5,7,8]]]

df_swa.index=['LR', 'LDA', 'RF', 'XGB']
df_swa.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'ROC AUC Score']  


df_swa=df_swa.drop('s',axis =1)
df_swa=df_swa.drop('NPV',axis =1)
df_swa.insert(6, "PR AUC Score",pr[0:4],True)
#df_swa = pd.DataFrame(df_swa).T

all_30_8=np.array((lm_30[1,:],lda_30[1,:],rf_30[1,:],xgb_30[1,:]))

df1_30=pd.DataFrame(all_30_8)
df_swa_30=df1[df1_30.columns[[0,6,1,2,3,4,5,7,8]]]


df_swa_30.index=['LR', 'LDA', 'RF', 'XGB']
df_swa_30.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'AUC Score']  


df_swa_30=df_swa_30.drop('s',axis =1)
df_swa_30=df_swa_30.drop('NPV',axis =1)
df_swa_30.insert(6, "PR AUC Score",pr[4:8],True)
#df_swa_30 = pd.DataFrame(df_swa_30).T



all_15_6=np.array((lm_15[0,:],lda_15[0,:],rf_15[0,:],xgb_15[0,:]))
pr_6=(all_6[:,6]+all_6[:,4])/2
df2=pd.DataFrame(all_15_6)
df_swb=df2[df2.columns[[0,6,1,2,3,4,5,7,8]]]

df_swb.index=['LR', 'LDA', 'RF', 'XGB']
df_swb.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'AUC Score']  


df_swb=df_swb.drop('s',axis =1)
df_swb=df_swb.drop('NPV',axis =1)
df_swb.insert(6, "PR AUC Score",pr[0:4],True)



all_30_6=np.array((lm_30[0,:],lda_30[0,:],rf_30[0,:],xgb_30[0,:]))

df2_30=pd.DataFrame(all_30_6)
df_swb_30=df1[df2_30.columns[[0,6,1,2,3,4,5,7,8]]]


df_swb_30.index=['LR', 'LDA', 'RF', 'XGB']
df_swb_30.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'AUC Score']  


df_swb_30=df_swb_30.drop('s',axis =1)
df_swb_30=df_swb_30.drop('NPV',axis =1)
df_swb_30.insert(6, "PR AUC Score",pr[4:8],True)