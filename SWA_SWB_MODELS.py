# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 19:08:42 2025

@author: 15206
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from itertools import product

data_rained=pd.read_excel(r'D:/PFDF_08_2025/two_feature_c.xlsx', sheet_name=[ 'Sheet2','Sheet3','Sheet4' ,'Sheet5','Sheet6','Sheet7'])


i15_s=data_rained['Sheet5']


#SEPARATING DATA INTO CATEGORIES

response=i15_s[['Response']].copy()

number=i15_s[['No']].copy()
vtype=i15_s[['VT']].copy()

re=response.to_numpy()

topography_col=['Ruggedness_S', 'Slope_mean', 's23', 's23MH', 's23H','a2000s15', 
                     'a2000s15MH', 'a2000s15H', 'a2000s23', 'a2000s23MH','a2000s23H','MeanSlopeLMH', 'MeanSlopeMH',
                     'S23LMH', 'S30LMH' ,'S30MH']

t_i15=i15_s[topography_col].to_numpy(copy=True)

fire_col=['mdNBR_1000', 'MH', 'L_2_MH', 'L_3_MH', 'L_4_MH']

f_i15= i15_s[fire_col].to_numpy(copy=True)

number_data,number_terrain=t_i15.shape
number_fire=f_i15.shape[1]

number_combo=number_fire*number_terrain

#CREATING THE MATRICES WITH THE DIFFERENT COMBINATIONS

i15=np.empty((number_data, 5,number_combo))
s=0;




for s, (j, k) in enumerate(product(range(number_terrain), range(number_fire))):
    i15[:, :, s]= np.column_stack((re[:,0],t_i15[:,j],f_i15[:,k],number,vtype))
    

data_i15_SWA_o=i15[:,:,7]

data_i15_SWB_o=i15[:,:,5]


length_test=re.size
np.random.seed(587)
SWA_data_i15=data_i15_SWA_o[np.random.permutation(length_test),:]
SWB_data_i15=data_i15_SWB_o[np.random.permutation(length_test),:]



SWA_response=SWA_data_i15[:,0]
SWA_data=SWA_data_i15[:,1:3]
SWB_response=SWB_data_i15[:,0]
SWB_data=SWB_data_i15[:,1:3]


no_total=re.size-np.sum(re)
yes_total=np.sum(re)




model_A = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000,class_weight={0:1, 1:np.sqrt(no_total/yes_total)}, fit_intercept=True)
SWA_s=model_A.fit(SWA_data, SWA_response)
SWA_beta0=print(SWA_s.intercept_)
SWA_beta2=print( SWA_s.coef_)

model_B = LogisticRegression(penalty="l2", C=3,solver="lbfgs",max_iter=1000,class_weight={0:1, 1:np.sqrt(no_total/yes_total)}, fit_intercept=True)
SWB_s=model_B.fit(SWB_data, SWB_response)

SWB_beta0=print(SWB_s.intercept_)
SWB_beta2=print( SWB_s.coef_)
