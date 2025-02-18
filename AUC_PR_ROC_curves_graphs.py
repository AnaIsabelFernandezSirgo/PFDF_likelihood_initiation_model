# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:53:21 2025

@author: 15206
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

#This info comes from metric_accuracy_tables.py
lda_15=np.load("results_lda_15.npy")
lda_30=np.load("results_lda_30.npy")
lm_15=np.load("results_lm_15.npy")
lm_30=np.load("results_lm_30.npy")
rf_15=np.load("results_rf_15.npy")
rf_30=np.load("results_rf_30.npy")
xgb_15=np.load("results_xgb_15.npy")
xgb_30=np.load("results_xgb_30.npy")


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
df_swa = pd.DataFrame(df_swa).T

all_30_8=np.array((lm_30[1,:],lda_30[1,:],rf_30[1,:],xgb_30[1,:]))

df1_30=pd.DataFrame(all_30_8)
df_swa_30=df1[df1_30.columns[[0,6,1,2,3,4,5,7,8]]]


df_swa_30.index=['LR', 'LDA', 'RF', 'XGB']
df_swa_30.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'AUC Score']  


df_swa_30=df_swa_30.drop('s',axis =1)
df_swa_30=df_swa_30.drop('NPV',axis =1)
df_swa_30.insert(6, "PR AUC Score",pr[4:8],True)
df_swa_30 = pd.DataFrame(df_swa_30).T



all_15_6=np.array((lm_15[0,:],lda_15[0,:],rf_15[0,:],xgb_15[0,:]))
pr_6=(all_6[:,6]+all_6[:,4])/2
df2=pd.DataFrame(all_15_6)
df_swb=df2[df2.columns[[0,6,1,2,3,4,5,7,8]]]

df_swb.index=['LR', 'LDA', 'RF', 'XGB']
df_swb.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'AUC Score']  


df_swb=df_swb.drop('s',axis =1)
df_swb=df_swb.drop('NPV',axis =1)
df_swb.insert(6, "PR AUC Score",pr_6[0:4],True)

df_swb = pd.DataFrame(df_swb).T

all_30_6=np.array((lm_30[0,:],lda_30[0,:],rf_30[0,:],xgb_30[0,:]))

df2_30=pd.DataFrame(all_30_6)
df_swb_30=df1[df2_30.columns[[0,6,1,2,3,4,5,7,8]]]


df_swb_30.index=['LR', 'LDA', 'RF', 'XGB']
df_swb_30.columns =['TS', 'TPR' ,  's','FPR', 'Specificity', "Precision", 'NPV', "F1 Score", 'AUC Score']  


df_swb_30=df_swb_30.drop('s',axis =1)
df_swb_30=df_swb_30.drop('NPV',axis =1)
df_swb_30.insert(6, "PR AUC Score",pr_6[4:8],True)
df_swb_30 = pd.DataFrame(df_swb_30).T



false_positive_rate_15_8=np.array((lm_15[1,2], lda_15[1,2], rf_15[1,2], xgb_15[1,2]))     
true_positive_rate_15_8= np.array((lm_15[1,6], lda_15[1,6], rf_15[1,6], xgb_15[1,6]))    



false_positive_rate_15_6=np.array((lm_15[0,2], lda_15[0,2], rf_15[0,2], xgb_15[0,2]))     
true_positive_rate_15_6= np.array((lm_15[0,6], lda_15[0,6], rf_15[0,6], xgb_15[0,6]))    
     




#my_colors = list(islice(cycle(['#DDAA33', '#BB5566', '#004488', '#000000']), None, len((df_15_F.T))))


fig, ax = plt.subplots(1, figsize=(7,7))
#plt.title('Receiver Operating Characteristic', fontsize=14)
#plt.suptitle('SWA-R15', fontsize=16)
plt.plot(false_positive_rate_15_8[0], true_positive_rate_15_8[0],'o', color='#DDAA33',  fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot(false_positive_rate_15_8[1], true_positive_rate_15_8[1],'o', color='#BB5566',fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot(false_positive_rate_15_8[2], true_positive_rate_15_8[2],'o',color='#004488', fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot(false_positive_rate_15_8[3], true_positive_rate_15_8[3],'o',color= '#000000',fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot(false_positive_rate_15_6[0], true_positive_rate_15_6[0],'x',color='#DDAA33',fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot(false_positive_rate_15_6[1], true_positive_rate_15_6[1],'x', color='#BB5566',fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot(false_positive_rate_15_6[2], true_positive_rate_15_6[2],'x',color='#004488',fillstyle='none',markeredgewidth=2, markersize=10 )
plt.plot(false_positive_rate_15_6[3], true_positive_rate_15_6[3],'x',color= '#000000', fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot([0,.1 ,.2,.6,1], [0,.2,.4,.8,1],ls="--", c='grey', label='ROC AUC=0.65')
plt.plot([0,.1 ,.2,.4,1], [0,.4,.6,.8,1],ls="--", c='lightgrey',label='ROC AUC=0.75')
# plt.text(0.08, 0.2, 'AUC=0.75')'_Hidden label'ls='--',
# plt.text(0.9, 0.2, 'AUC=0.65')
plt.plot([0, 1],  c='k')
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7",label='Perfect Classifier')
plt.ylabel('True Positive Rate', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=16)
legend_elements = [
                   Line2D([0], [0], marker='o',   color='w', label="$S_{m}$ , L2MH", markeredgecolor='#666666',markeredgewidth=2, markerfacecolor='w', markersize=10),
                   Line2D([0], [0], marker='x', color='w', label="$S_{m}$, dNBR/1000" , markeredgecolor='#666666',markeredgewidth=2, markerfacecolor='#666666', markersize=10),
                   Patch(facecolor='#DDAA33', edgecolor='k', label='LR'),
                   Patch(facecolor='#BB5566', edgecolor='k', label='LDA'),
                   Patch(facecolor='#004488', edgecolor='k', label='RF'),
                   Patch(facecolor='k', edgecolor='k', label='XGB'),
                   Line2D([0], [0], linestyle='--', color='grey', label='ROC AUC 0.65'),
                   Line2D([0], [0], linestyle='dashed', color='lightgrey', label='ROC AUC 0.75'),
                   Line2D([0], [0], linestyle='solid', color='.7', label='Perfect Classifier')
                   ]

ax.legend(handles=legend_elements, loc='best')
# plt.plot('xtick', labelsize=14)    # fontsize of the tick labels
# plt.plot('ytick', labelsize=14)    # fontsize of the tick labels
#plt.legend(("s", ["b", "o", "g", "r"], ["*","o"], "k"),  "LR", "LDA", "RF", "XGB", "$S_{m}$ , L2MH", "$S_{m}$, dNBR/1000",  loc='lower right', fontsize=14)
# handels=[("s", ["b", "o", "g", "r"]), (["*","o"], "k")]
# labels=[ "LR", "LDA", "RF", "XGB", "$S_{m}$ , L2MH", "$S_{m}$, dNBR/1000"]
# plt.legend(handels, labels,   loc='lower right', fontsize=14)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
plt.show()

 



precision_15_8=np.array((lm_15[1,4], lda_15[1,4], rf_15[1,4], xgb_15[1,4]))     
precision_15_6=np.array((lm_15[0,4], lda_15[0,4], rf_15[0,4], xgb_15[0,4])) 
bb=122/(122+975)
fig, ax2= plt.subplots(1, figsize=(7,7))
#plt.title('Presicion recall curve', fontsize=14)
#plt.suptitle('SWA-R15', fontsize=16)
plt.plot( true_positive_rate_15_8[0], precision_15_8[0],'o', color='#DDAA33', fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_8[1],precision_15_8[1],'o', color='#BB5566', fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_8[2],precision_15_8[2],'o',color='#004488', fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_8[3],precision_15_8[3],'o', color= '#000000',fillstyle='none',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_6[0], precision_15_6[0],'x',color='#DDAA33',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_6[1],precision_15_6[1],'x', color='#BB5566',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_6[2],precision_15_6[2],'x',color='#004488',markeredgewidth=2, markersize=10)
plt.plot( true_positive_rate_15_6[3],precision_15_6[3],'x',color= '#000000',markeredgewidth=2, markersize=10)
plt.plot([0, 1], [bb,bb],  c='k', label='Baseline Clasifier')
plt.plot([1, 1], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7", label="Perfect Classifier")
plt.ylabel('Positive Predictive Value', fontsize=16)
plt.xlabel('True Positive Rate', fontsize=16)
plt.legend(loc='lower right', fontsize=14)
legend_elements = [
                   Line2D([0], [0], marker='o',   color='w', label="$S_{m}$ , L2MH", markeredgecolor='#666666',markeredgewidth=2, markerfacecolor='w', markersize=10),
                   Line2D([0], [0], marker='x', color='w', label="$S_{m}$, dNBR/1000" , markeredgecolor='#666666',markeredgewidth=2, markerfacecolor='#666666', markersize=10),
                   Patch(facecolor='#DDAA33', edgecolor='k', label='LR'),
                   Patch(facecolor='#BB5566', edgecolor='k', label='LDA'),
                   Patch(facecolor='#004488', edgecolor='k', label='RF'),
                   Patch(facecolor='k', edgecolor='k', label='XGB'),
                   Line2D([0], [0], linestyle='solid', color='k', label='Baseline Classifier'),
                   #Line2D([0], [0], linestyle='dashed', color='lightgrey', label='ROC AUC 0.75'),
                   Line2D([0], [0], linestyle='solid', color='.7', label='Perfect Classifier')
                   ]
# plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
ax2.legend(handles=legend_elements, loc='best')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

plt.show()










false_positive_rate_30_8=np.array((lm_30[1,2], lda_30[1,2], rf_30[1,2], xgb_30[1,2]))     
true_positive_rate_30_8= np.array((lm_30[1,6], lda_30[1,6], rf_30[1,6], xgb_30[1,6]))


false_positive_rate_30_6=np.array((lm_30[0,2], lda_30[0,2], rf_30[0,2], xgb_30[0,2]))     
true_positive_rate_30_6= np.array((lm_30[0,6], lda_30[0,6], rf_30[0,6], xgb_30[0,6]))    
    

 
    
     

fig, ax= plt.subplots(1, figsize=(7,7))
#plt.title('Receiver Operating Characteristic', fontsize=14)
#plt.suptitle('SWA-R30', fontsize=16)
plt.plot(false_positive_rate_30_8[0], true_positive_rate_30_8[0],'bo', fillstyle='none',markeredgewidth=2)
plt.plot(false_positive_rate_30_8[1], true_positive_rate_30_8[1],'o', color='orange', fillstyle='none',markeredgewidth=2)
plt.plot(false_positive_rate_30_8[2], true_positive_rate_30_8[2],'go', fillstyle='none',markeredgewidth=2)
plt.plot(false_positive_rate_30_8[3], true_positive_rate_30_8[3],'ro', fillstyle='none',markeredgewidth=2)
plt.plot(false_positive_rate_30_8[0], true_positive_rate_30_6[0],'bx',markeredgewidth=2)
plt.plot(false_positive_rate_30_8[1], true_positive_rate_30_6[1],'x', color='orange',markeredgewidth=2)
plt.plot(false_positive_rate_30_8[2], true_positive_rate_30_6[2],'gx',markeredgewidth=2)
plt.plot(false_positive_rate_30_6[3], true_positive_rate_30_6[3],'rx',markeredgewidth=2)
plt.plot([0,.1 ,.2,.6,1], [0,.2,.4,.8,1],ls="--", c='grey', label='ROC AUC=0.65')
plt.plot([0,.1 ,.2,.4,1], [0,.4,.6,.8,1],ls="--", c='lightgrey',label='ROC AUC=0.75')
plt.plot([0, 1],  c='k')
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
legend_elements = [
                   Line2D([0], [0], marker='o',   color='w', label="$S_{m}$ , L2MH", markeredgecolor='k',markeredgewidth=2, markerfacecolor='w', markersize=10),
                   Line2D([0], [0], marker='X', color='w', label="$S_{m}$, dNBR/1000" , markeredgecolor='k',markeredgewidth=.5, markerfacecolor='k', markersize=10),
                   Patch(facecolor='blue', edgecolor='k', label='LR'),
                   Patch(facecolor='orange', edgecolor='k', label='LDA'),
                   Patch(facecolor='green', edgecolor='k', label='RF'),
                   Patch(facecolor='red', edgecolor='k', label='XGB'),
                   Line2D([0], [0], linestyle='--', color='grey', label='ROC AUC 0.65'),
                   Line2D([0], [0], linestyle='dashed', color='lightgrey', label='ROC AUC 0.75'),
                   Line2D([0], [0], linestyle='solid', color='.7', label='Perfect Classifier')
                   ]
ax.legend(handles=legend_elements, loc='best')
#plt.legend(loc='lower right', fontsize=14)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

plt.show()




precision_30_8=np.array((lm_30[1,4], lda_30[1,4], rf_30[1,4], xgb_30[1,4]))     
precision_30_6=np.array((lm_30[0,4], lda_30[0,4], rf_30[0,4], xgb_30[0,4])) 
bb=122/(122+975)
fig, ax= plt.subplots(1, figsize=(7,7))
#plt.title('Presicion recall curve', fontsize=14)
#plt.suptitle('SWA-R30', fontsize=16)
plt.plot( true_positive_rate_30_8[0], precision_30_8[0],'bo', fillstyle='none',markeredgewidth=2)
plt.plot( true_positive_rate_30_8[1],precision_30_8[1],'o', fillstyle='none', color='orange',markeredgewidth=2)
plt.plot( true_positive_rate_30_8[2],precision_30_8[2],'go', fillstyle='none',markeredgewidth=2)
plt.plot( true_positive_rate_30_8[3],precision_30_8[3],'ro', fillstyle='none',markeredgewidth=2)
plt.plot( true_positive_rate_30_6[0], precision_30_6[0],'bx',markeredgewidth=2)
plt.plot( true_positive_rate_30_6[1],precision_30_6[1],'x', color='orange',markeredgewidth=2)
plt.plot( true_positive_rate_30_6[2],precision_30_6[2],'gx',markeredgewidth=2)
plt.plot( true_positive_rate_30_6[3],precision_30_6[3],'rx',markeredgewidth=2)
plt.plot([0, 1], [bb,bb],  c='k', label='Baseline Clasifier',markeredgewidth=2)
plt.plot([1, 1], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7", label="Perfect Classifier")
plt.ylabel('Positive Predictive Value', fontsize=14)
plt.xlabel('True Positive Rate', fontsize=14)
legend_elements = [
                   Line2D([0], [0], marker='o',   color='w', label="$S_{m}$ , L2MH", markeredgecolor='k',markeredgewidth=2, markerfacecolor='w', markersize=10),
                   Line2D([0], [0], marker='X', color='w', label="$S_{m}$, dNBR/1000" , markeredgecolor='k',markeredgewidth=.5, markerfacecolor='k', markersize=10),
                   Patch(facecolor='blue', edgecolor='k', label='LR'),
                   Patch(facecolor='orange', edgecolor='k', label='LDA'),
                   Patch(facecolor='green', edgecolor='k', label='RF'),
                   Patch(facecolor='red', edgecolor='k', label='XGB'),
                   Line2D([0], [0], linestyle='solid', color='k', label='Baseline Classifier'),
                   #Line2D([0], [0], linestyle='dashed', color='lightgrey', label='ROC AUC 0.75'),
                   Line2D([0], [0], linestyle='solid', color='.7', label='Perfect Classifier')
                   ]
ax.legend(handles=legend_elements, loc='best')
#plt.legend(loc='lower right', fontsize=14)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

plt.show()


 





