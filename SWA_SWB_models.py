# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:52:04 2025

@author: 15206
"""




import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

#SWA SWB  are crfeated with r15  and LR
#i15_2 is created with data_2_feature_matrices.py
#i15_2.npy


i15= np.load("i15_2.npy")
 
#SWA
i15=i15[0:1463,:,:]

SWa_info=i15[:,:,7]


response_SWA=SWa_info[:,0]
data_SWA=SWa_info[:,1:3]
#SWA
model_lm_SWA_15 = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150,class_weight={0:1, 1:np.sqrt(1300/163)})
SWA=model_lm_SWA_15.fit(data_SWA,response_SWA)
SWA_coefficients= ( SWA.intercept_, SWA.coef_)
#beta0, beta1 beta 2


#SWB 

SWB_info=i15[:,:,5]


response_SWB=SWB_info[:,0]
data_SWB=SWB_info[:,1:3]

model_lm_6_15= LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150,class_weight={0:1, 1:np.sqrt(1300/163)})
s_lm_6_15=model_lm_6_15.fit(data_SWB,response_SWB)
SWB_coefficients=(s_lm_6_15.intercept_,s_lm_6_15.coef_)

#beta0, beta1 beta 2