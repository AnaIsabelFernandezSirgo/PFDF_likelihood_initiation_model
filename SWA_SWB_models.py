# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:52:04 2025

@author: 15206
"""




import numpy as np
from sklearn.linear_model import LogisticRegression


#SWA SWB  are crfeated with r15  and LR
#i15_2 and i30_2 are created with data_2_feature_matrices.py
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


#SWA_{30}
i30= np.load("i30_2.npy")
 

i30=i30[0:1463,:,:]

info_SWA_30=i30[:,:,7]

response_SWA_30=info_SWA_30[:,0]
data_SWA_30=info_SWA_30[:,1:3]

model_lm_8_30 = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150,class_weight={0:1, 1:np.sqrt(975/122)})
s_lm_8_30=model_lm_8_30.fit(data_SWA_30,response_SWA_30)
SWA_30_coefficients=(s_lm_8_30.intercept_,s_lm_8_30.coef_)

#SWB_{30}

info_SWB_30=i30[:,:,5]

response_SWB_30=info_SWB_30[:,0]
data_SWB_30=info_SWB_30[:,1:3]

model_lm_6_30= LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150,class_weight={0:1, 1:np.sqrt(1300/163)})
s_lm_6_30=model_lm_6_30.fit(data_SWB_30,response_SWB_30)
coefficients_SWB_30=print(s_lm_6_30.intercept_,s_lm_6_30.coef_)
