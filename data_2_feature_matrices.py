# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:24:34 2025

@author: 15206

"""

import pandas as pd
import numpy as np


#creates the data you will need for 2 feature models 
#record results
#check names


# clean ordered (0-1) reduced and the data is multiplied by rain 
data_rained=pd.read_excel(r"D:\Fire_Mar24\data_rained.xlsx", sheet_name=['Sheet1', 'Sheet2','Sheet3','Sheet4'])
i10=data_rained['Sheet1']
i15=data_rained['Sheet2']
i30=data_rained['Sheet3']
i60=data_rained['Sheet4']




#all combo

response=i10[['Response']].copy()

re=response.to_numpy()

topography_i10= i10[['Ruggedness_S', 'Slope_mean', 's23', 's23MH', 's23H','a2000s15', 
                     'a2000s15MH', 'a2000s15H', 'a2000s23', 'a2000s23MH','a2000s23H','MeanSlopeLMH', 'MeanSlopeMH',
                     'S23LMH', 'S30LMH' ,'S30MH']].copy()
topography_i15= i15[['Ruggedness_S', 'Slope_mean', 's23', 's23MH', 's23H','a2000s15', 
                     'a2000s15MH', 'a2000s15H', 'a2000s23', 'a2000s23MH','a2000s23H','MeanSlopeLMH', 'MeanSlopeMH',
                     'S23LMH', 'S30LMH', 'S30MH']].copy()
topography_i30=i30[['Ruggedness_S', 'Slope_mean', 's23', 's23MH', 's23H','a2000s15', 
                    'a2000s15MH', 'a2000s15H', 'a2000s23', 'a2000s23MH','a2000s23H','MeanSlopeLMH', 'MeanSlopeMH',
                    'S23LMH', 'S30LMH', 'S30MH']].copy()
topography_i60= i60[['Ruggedness_S', 'Slope_mean', 's23', 's23MH', 's23H','a2000s15', 
                     'a2000s15MH', 'a2000s15H', 'a2000s23', 'a2000s23MH','a2000s23H','MeanSlopeLMH' ,'MeanSlopeMH' ,
                     'S23LMH', 'S30LMH' , 'S30MH']].copy()

t_i10=topography_i10.to_numpy()
t_i15=topography_i15.to_numpy()
t_i30=topography_i30.to_numpy()
t_i60=topography_i60.to_numpy()



fire_i10= i10[['mdNBR', 'MH', 'L_2_MH', 'L_3_MH', 'L_4_MH']].copy()
fire_i15= i15[['mdNBR', 'MH', 'L_2_MH', 'L_3_MH', 'L_4_MH']].copy()
fire_i30= i30[['mdNBR', 'MH', 'L_2_MH', 'L_3_MH', 'L_4_MH']].copy()
fire_i60= i60[['mdNBR', 'MH', 'L_2_MH', 'L_3_MH', 'L_4_MH']].copy()


f_i10=fire_i10.to_numpy()
f_i15=fire_i15.to_numpy()
f_i30=fire_i30.to_numpy()
f_i60=fire_i60.to_numpy()

soil_i10= i10[['Clay', 'OM','KF', 'Prem', 'Thick']].copy()
soil_i15= i15[['Clay', 'OM','KF', 'Prem', 'Thick']].copy()
soil_i30= i30[['Clay', 'OM','KF', 'Prem', 'Thick']].copy()
soil_i60= i60[['Clay', 'OM','KF', 'Prem', 'Thick']].copy()



s_i10=soil_i10.to_numpy()
s_i15=soil_i15.to_numpy()
s_i30=soil_i30.to_numpy()
s_i60=soil_i60.to_numpy()





tz1,tz2=topography_i10.shape
fz1,fz2=fire_i10.shape
sz1,sz2=soil_i10.shape





combo=tz2*fz2
i10_r=np.zeros((tz1*4, 3,combo))
i15_r=np.zeros((tz1*4, 3,combo))
i30_r=np.zeros((tz1*4, 3,combo))
i60_r=np.zeros((tz1*4, 3,combo))
s=0;
for j in range (tz2):
    for k in range (fz2):
        
            ver_re=np.concatenate((re[:,0], re[:,0], re[:,0],  re[:,0]))
            ver_t=np.concatenate(( t_i10[:,j], t_i10[:,j], t_i10[:,j], t_i10[:,j]))
            ver_f=np.concatenate(( f_i10[:,k], f_i10[:,k], f_i10[:,k], f_i10[:,k]))
        
            ver=np.column_stack((ver_re, ver_t, ver_f))
            
            i10_r[:,:,s]=ver
                
            ve_re=np.concatenate((re[:,0], re[:,0], re[:,0],  re[:,0]))
            ver_t=np.concatenate(( t_i15[:,j], t_i15[:,j], t_i15[:,j], t_i15[:,j]))
            ver_f=np.concatenate(( f_i15[:,k], f_i15[:,k], f_i15[:,k], f_i15[:,k]))
            
            ver=np.column_stack((ver_re, ver_t, ver_f))
            
            i15_r[:,:,s]=ver
                
            ver_re=np.concatenate((re[:,0], re[:,0], re[:,0],  re[:,0]))
            ver_t=np.concatenate(( t_i30[:,j], t_i30[:,j], t_i30[:,j], t_i30[:,j]))
            ver_f=np.concatenate(( f_i30[:,k], f_i30[:,k], f_i30[:,k], f_i30[:,k]))
            
            ver=np.column_stack((ver_re, ver_t, ver_f))
            
            i30_r[:,:,s]=ver
                
            ver_re=np.concatenate((re[:,0], re[:,0], re[:,0],  re[:,0]))
            ver_t=np.concatenate(( t_i60[:,j], t_i60[:,j], t_i60[:,j], t_i60[:,j]))
            ver_f=np.concatenate(( f_i60[:,k], f_i60[:,k], f_i60[:,k], f_i60[:,k]))
            
            ver=np.column_stack((ver_re, ver_t, ver_f))
            
            i60_r[:,:,s]=ver
            s=s+1
            
            
            