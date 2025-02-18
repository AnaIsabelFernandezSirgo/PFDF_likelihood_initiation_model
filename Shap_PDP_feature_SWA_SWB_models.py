# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:30:57 2025

@author: 15206
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import statistics
from sklearn.inspection import partial_dependence

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from itertools import cycle, islice  
m=1463

i15=np.load("i15_2.npy")
i30 = np.load("i30_2.npy")

data_SWA=i15[0:m,:,7]
data_SWA_30=i30[0:m,:,7]




c_train_30_8=data_SWA_30[:,0]
t_2_SWA_30=data_SWA_30[:,1:3]
c_train_SWA=data_SWA[:,0]
t_2_SWA=data_SWA[:,1:3]




model = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150, class_weight={0:1, 1:np.sqrt(975/122)})



s_lm_SWA=model.fit(t_2_SWA, c_train_SWA)
explainer_lm_SWA =shap.Explainer(s_lm_SWA.predict, t_2_SWA)
shap_values_lm_SWA = explainer_lm_SWA(t_2_SWA)
terrain_mean_shapley_lm_SWA=statistics.mean(abs(shap_values_lm_SWA.values[:,0]))
fire_mean_shapley_lm_SWA=statistics.mean(abs(shap_values_lm_SWA.values[:,1]))


model = LinearDiscriminantAnalysis()




s_lda_SWA = model.fit(t_2_SWA, c_train_SWA)
explainer_lda_SWA =shap.Explainer(s_lda_SWA.predict, t_2_SWA)
shap_values_lda_SWA = explainer_lda_SWA(t_2_SWA)
terrain_mean_shapley_lda_SWA=statistics.mean(abs(shap_values_lda_SWA.values[:,0]))
fire_mean_shapley_lda_SWA=statistics.mean(abs(shap_values_lda_SWA.values[:,1]))
#random_state=0,

model= RandomForestClassifier( n_estimators=100, max_depth=2,random_state=169, max_features=2,class_weight={0:1, 1:np.sqrt(975/122)})

s_rf_SWA=model.fit(t_2_SWA, c_train_SWA)
explainer_rf_SWA =shap.Explainer(s_rf_SWA.predict, t_2_SWA)
shap_values_rf_SWA = explainer_rf_SWA(t_2_SWA)
terrain_mean_shapley_rf_SWA=statistics.mean(abs(shap_values_rf_SWA.values[:,0]))
fire_mean_shapley_rf_SWA=statistics.mean(abs(shap_values_rf_SWA.values[:,1]))


model = xgb.XGBClassifier(booster='gbtree', 
    objective='binary:logistic',
              eval_metric='error',
              gamma=0,
              learning_rate=0.05,
              max_depth=6,
              reg_lambda=10,
              scale_pos_weight=np.sqrt(975/122),
              subsample=.9,
              #seed=0,
              colsample_bytree=.9, monotone_constraints = (1,1,1))

s_xgb_SWA=model.fit(t_2_SWA, c_train_SWA)



explainer_xgb_SWA =shap.Explainer(s_xgb_SWA.predict, t_2_SWA)
shap_values_xgb_SWA = explainer_xgb_SWA(t_2_SWA)
terrain_mean_shapley_xgb_SWA=statistics.mean(abs(shap_values_xgb_SWA.values[:,0]))
fire_mean_shapley_xgb_SWA=statistics.mean(abs(shap_values_xgb_SWA.values[:,1]))


# SHAP  BAR PLOTS  I15

#LOGISTIC

DATA_TERR_15={'LR':terrain_mean_shapley_lm_SWA,'LDA':terrain_mean_shapley_lda_SWA,'RF':terrain_mean_shapley_rf_SWA,'XGB':terrain_mean_shapley_xgb_SWA}
DATA_FIRE_15={'LR':fire_mean_shapley_lm_SWA,'LDA':fire_mean_shapley_lda_SWA,'RF': fire_mean_shapley_rf_SWA, 'XGB':fire_mean_shapley_xgb_SWA}
var_TERR_15 = list(DATA_TERR_15.keys())
values_TERR_15 = list(DATA_TERR_15.values())
var_FIRE_15 = list(DATA_FIRE_15.keys())
values_FIRE_15 = list(DATA_FIRE_15.values())

DATA_15={'$R_{15}*S_{m}$         ':DATA_TERR_15, '   $R_{15}*L2MH$      ':DATA_FIRE_15}
var_DATA_15 = list(DATA_15.keys())
values_DATA_15 = list(DATA_15.values())
fig_LM = plt.figure(figsize=(10, 5))

# plot grouped bar chart BY FEATRURE


df_15_F = pd.DataFrame(DATA_15).T

my_colors = list(islice(cycle(['#DDAA33', '#BB5566', '#004488', '#000000']), None, len((df_15_F.T))))
sh=df_15_F.plot(kind="barh", fontsize=16, color=my_colors)
#plt.title("Average Impact on Model Output Magnitude ", fontsize=14)
sh.set_xlabel("mean|Shapley value|",fontsize=18)
#plt.suptitle('SWA-$R_{15}$',fontsize=16)
plt.legend(bbox_to_anchor=(1., 1), loc='upper left', fontsize=14)
plt.show()



# #INDIVIDULA

data_LM_15 = {'Terrain':terrain_mean_shapley_lm_SWA, 'Fire':fire_mean_shapley_lm_SWA}


# PDP GRAPHS I15 TERRAIN



 
pd_results_lm_SWA_1 = partial_dependence(s_lm_SWA, t_2_SWA, features=0)
df1_a = np.concatenate([pd_results_lm_SWA_1.average,pd_results_lm_SWA_1.grid_values])
pd_results_lda_SWA_1 = partial_dependence(s_lda_SWA, t_2_SWA, features=0)
df1_b = np.concatenate([pd_results_lda_SWA_1.average,pd_results_lda_SWA_1.grid_values])
pd_results_rf_SWA_1 = partial_dependence(s_rf_SWA, t_2_SWA, features=0)
df1_c = np.concatenate([pd_results_rf_SWA_1.average,pd_results_rf_SWA_1.grid_values])
pd_results_xgb_SWA_1 = partial_dependence(s_xgb_SWA, t_2_SWA, features=0)
df1_d = np.concatenate([pd_results_xgb_SWA_1.average,pd_results_xgb_SWA_1.grid_values])


plt.plot(df1_a[1,:], df1_a[0,:], color='#DDAA33', linewidth=3.0)
plt.plot(df1_b[1,:], df1_b[0,:], color ='#BB5566',linewidth=3.0)
plt.plot(df1_c[1,:], df1_c[0,:], color= '#004488',linewidth=3.0)
plt.plot(df1_d[1,:], df1_d[0,:], color='#000000',linewidth=3.0)
plt.xlabel("$R_{15}*S_{m}$",fontsize=18)
plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.legend(["LR", "LDA", "RF",'XGB'],fontsize=14)
plt.show()




#PDP GRAPHS I15 FIRE

#LR
pd_results_lm_SWA_2 = partial_dependence(s_lm_SWA, t_2_SWA, features=1)
df2_a = np.concatenate([pd_results_lm_SWA_2.average,pd_results_lm_SWA_2.grid_values])


pd_results_lda_SWA_2 = partial_dependence(s_lda_SWA, t_2_SWA, features=1)
df2_b = np.concatenate([pd_results_lda_SWA_2.average,pd_results_lda_SWA_2.grid_values])


#RF

pd_results_rf_SWA_2 = partial_dependence(s_rf_SWA, t_2_SWA, features=1)
df2_c = np.concatenate([pd_results_rf_SWA_2.average,pd_results_rf_SWA_2.grid_values])


#XGB

pd_results_xgb_SWA_2 = partial_dependence(s_xgb_SWA, t_2_SWA, features=1)
df2_d = np.concatenate([pd_results_xgb_SWA_2.average,pd_results_xgb_SWA_2.grid_values])





#ALL

plt.plot(df2_a[1,:], df2_a[0,:], color='#DDAA33', linewidth=3.0)
plt.plot(df2_b[1,:], df2_b[0,:], color='#BB5566',  linewidth=3.0)
plt.plot(df2_c[1,:], df2_c[0,:], color='#004488',  linewidth=3.0)
plt.plot(df2_d[1,:], df2_d[0,:], color='#000000', linewidth=3.0)
plt.xlabel("$R_{15}*L2MH$",fontsize=18)
#plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#plt.legend(["LR", "LDA", "RF",'XGB'],fontsize=14)
plt.show()


"""
SWB 
"""
i15=np.load("i15_2.npy")



data_SWB=i15[0:m,:,5]





c_train_SWB=data_SWB[:,0]
t_2_SWB=data_SWB[:,1:3]




model = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150, class_weight={0:1, 1:np.sqrt(975/122)})



s_lm_6_i15=model.fit(t_2_SWB, c_train_SWB)
explainer_lm_6_i15 =shap.Explainer(s_lm_6_i15.predict, t_2_SWB)
shap_values_lm_6_i15 = explainer_lm_6_i15(t_2_SWB)
terrain_mean_shapley_lm_6_i15=statistics.mean(abs(shap_values_lm_6_i15.values[:,0]))
fire_mean_shapley_lm_6_i15=statistics.mean(abs(shap_values_lm_6_i15.values[:,1]))


model = LinearDiscriminantAnalysis()




s_lda_6_i15 = model.fit(t_2_SWB, c_train_SWB)
explainer_lda_6_i15 =shap.Explainer(s_lda_6_i15.predict, t_2_SWB)
shap_values_lda_6_i15 = explainer_lda_6_i15(t_2_SWB)
terrain_mean_shapley_lda_6_i15=statistics.mean(abs(shap_values_lda_6_i15.values[:,0]))
fire_mean_shapley_lda_6_i15=statistics.mean(abs(shap_values_lda_6_i15.values[:,1]))


model= RandomForestClassifier( n_estimators=100, random_state=169,  max_depth=2, max_features=2, class_weight={0:1, 1:np.sqrt(975/122)})
#random_state=0
s_rf_6_i15=model.fit(t_2_SWB, c_train_SWB)
explainer_rf_6_i15 =shap.Explainer(s_rf_6_i15.predict, t_2_SWB)
shap_values_rf_6_i15 = explainer_rf_6_i15(t_2_SWB)
terrain_mean_shapley_rf_6_i15=statistics.mean(abs(shap_values_rf_6_i15.values[:,0]))
fire_mean_shapley_rf_6_i15=statistics.mean(abs(shap_values_rf_6_i15.values[:,1]))



model = xgb.XGBClassifier(booster='gbtree', 
    objective='binary:logistic',
              eval_metric='error',
              gamma=0,
              learning_rate=0.05,
              max_depth=6,
              reg_lambda=10,
              scale_pos_weight=np.sqrt(975/122),
              subsample=.9,
              
              colsample_bytree=.9, monotone_constraints = (1,1,1))

s_xgb_6_i15=model.fit(t_2_SWB, c_train_SWB)



explainer_xgb_6_i15 =shap.Explainer(s_xgb_6_i15.predict, t_2_SWB)
shap_values_xgb_6_i15 = explainer_xgb_6_i15(t_2_SWB)
terrain_mean_shapley_xgb_6_i15=statistics.mean(abs(shap_values_xgb_6_i15.values[:,0]))
fire_mean_shapley_xgb_6_i15=statistics.mean(abs(shap_values_xgb_6_i15.values[:,1]))


# SHAP  BAR PLOTS  I15



DATA_TERR_15={'LR':terrain_mean_shapley_lm_6_i15,'LDA':terrain_mean_shapley_lda_6_i15,'RF':terrain_mean_shapley_rf_6_i15,'XGB':terrain_mean_shapley_xgb_6_i15}
DATA_FIRE_15={'LR':fire_mean_shapley_lm_6_i15,'LDA':fire_mean_shapley_lda_6_i15,'RF': fire_mean_shapley_rf_6_i15, 'XGB':fire_mean_shapley_xgb_6_i15}
var_TERR_15 = list(DATA_TERR_15.keys())
values_TERR_15 = list(DATA_TERR_15.values())
var_FIRE_15 = list(DATA_FIRE_15.keys())
values_FIRE_15 = list(DATA_FIRE_15.values())

DATA_15={'$R_{15}*S_{m}$       ':DATA_TERR_15, '$R_{15}*dNBR/1000$':DATA_FIRE_15}
var_DATA_15 = list(DATA_15.keys())
values_DATA_15 = list(DATA_15.values())
fig_LM = plt.figure(figsize=(10, 5))

# plot grouped bar chart BY FEATRURE

import matplotlib.pyplot as plt
from itertools import cycle, islice

df_15_F = pd.DataFrame(DATA_15).T
my_colors = list(islice(cycle(['#DDAA33', '#BB5566', '#004488', '#000000']), None, len((df_15_F.T))))
sh=df_15_F.plot(kind="barh", fontsize=16, color=my_colors)
#plt.title("Average Impact on Model Output Magnitude ", fontsize=14)
sh.set_xlabel("mean|Shapley value|",fontsize=18)
#plt.suptitle('SWB-R_{15}',fontsize=16)
plt.legend(bbox_to_anchor=(1., 1), loc='upper left', fontsize=14)
plt.show()






 
pd_results_lm_6_i15_1 = partial_dependence(s_lm_6_i15, t_2_SWB, features=0)
df1_a = np.concatenate([pd_results_lm_6_i15_1.average,pd_results_lm_6_i15_1.grid_values])
pd_results_lda_6_i15_1 = partial_dependence(s_lda_6_i15, t_2_SWB, features=0)
df1_b = np.concatenate([pd_results_lda_6_i15_1.average,pd_results_lda_6_i15_1.grid_values])
pd_results_rf_6_i15_1 = partial_dependence(s_rf_6_i15, t_2_SWB, features=0)
df1_c = np.concatenate([pd_results_rf_6_i15_1.average,pd_results_rf_6_i15_1.grid_values])
pd_results_xgb_6_i15_1 = partial_dependence(s_xgb_6_i15, t_2_SWB, features=0)
df1_d = np.concatenate([pd_results_xgb_6_i15_1.average,pd_results_xgb_6_i15_1.grid_values])

plt.plot(df1_a[1,:], df1_a[0,:], color='#DDAA33', linewidth=3.0)
plt.plot(df1_b[1,:], df1_b[0,:], color ='#BB5566',linewidth=3.0)
plt.plot(df1_c[1,:], df1_c[0,:], color= '#004488',linewidth=3.0)
plt.plot(df1_d[1,:], df1_d[0,:], color='#000000',linewidth=3.0)
plt.xlabel("$R_{15}*S_{m}$",fontsize=18)
plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)
#plt.title("Partial Dependecy Plot: Terrain ",fontsize=14)
#plt.suptitle('SWB-R_{15}',fontsize=16)
#plt.legend(["LR", "LDA", "RF",'XGB'],fontsize=14)
plt.show()



# #LOGISTIC

#PDP GRAPHS I15 FIRE

#LOGISTIC
pd_results_lm_6_i15_2 = partial_dependence(s_lm_6_i15, t_2_SWB, features=1)
df2_a = np.concatenate([pd_results_lm_6_i15_2.average,pd_results_lm_6_i15_2.grid_values])

#LDA

pd_results_lda_6_i15_2 = partial_dependence(s_lda_6_i15, t_2_SWB, features=1)
df2_b = np.concatenate([pd_results_lda_6_i15_2.average,pd_results_lda_6_i15_2.grid_values])


#RF

pd_results_rf_6_i15_2 = partial_dependence(s_rf_6_i15, t_2_SWB, features=1)
df2_c = np.concatenate([pd_results_rf_6_i15_2.average,pd_results_rf_6_i15_2.grid_values])

#XGB

pd_results_xgb_6_i15_2 = partial_dependence(s_xgb_6_i15, t_2_SWB, features=1)
df2_d = np.concatenate([pd_results_xgb_6_i15_2.average,pd_results_xgb_6_i15_2.grid_values])





#ALL

plt.plot(df2_a[1,:], df2_a[0,:], color='#DDAA33', linewidth=3.0)
plt.plot(df2_b[1,:], df2_b[0,:], color='#BB5566',  linewidth=3.0)
plt.plot(df2_c[1,:], df2_c[0,:], color='#004488',  linewidth=3.0)
plt.plot(df2_d[1,:], df2_d[0,:], color='#000000', linewidth=3.0)
plt.xlabel("$R_{15}*dNBR/1000$",fontsize=18)
#plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)
#plt.title("Partial Dependecy Plot: Fire ",fontsize=14)
#plt.suptitle('SWB-R_{15}',fontsize=16)
#plt.legend(["LR", "LDA", "RF",'XGB'],fontsize=12)
plt.show()


