# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:26:16 2025

@author: 15206
"""



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

#dowload data_clean.xlsx and use it here
data_clean=pd.read_excel(r"HERE")
pinal=data_clean.iloc[880]
tadpole=data_clean.iloc[1660]


#2 variables 
#NOTE:_8==SWA features  _6== SWB features

pinal_6=pinal[['Slope_mean', "mdNBR" ]].copy()
tadpole_6=tadpole[['Slope_mean', "mdNBR" ]].copy()



pinal_8=pinal[['Slope_mean', "L_2_MH" ]].copy()
tadpole_8=tadpole[['Slope_mean', "L_2_MH" ]].copy()

contreras=pd.read_excel(r"D:\Fire_Mar24\con01.xlsx")


contreras_6=contreras[['Slope_mean', "mdNBR" ]].copy()

contreras_8=contreras[['Slope_mean', "L_2_MH" ]].copy()


i15= np.load("i15_2.npy")
i30=np.load("i30_2.npy")

response=i15[0:1463,0,5]
model_6_i15= i15[0:1463,1:3,5]
model_8_i15= i15[0:1463,1:3,7]





m1=300
m2=15
m3=m1-m2
rain=np.array(range(m2,m1))
rain_i15b=rain/4

rain_i15=np.reshape(rain_i15b,[m3,1])




hr_6=pinal_6.to_numpy()
zz_6=tadpole_6.to_numpy()



hr_6=np.reshape(hr_6,[1,2])
zz_6=np.reshape(zz_6,[1,2])



pinal_6_i15=hr_6*rain_i15
tadpole_6_i15=zz_6*rain_i15


hr_8=pinal_8.to_numpy()
zz_8=tadpole_8.to_numpy()


hr_8=np.reshape(hr_8,[1,2])
zz_8=np.reshape(zz_8,[1,2])



pinal_8_i15=hr_8*rain_i15
tadpole_8_i15=zz_8*rain_i15

con_6=contreras_6.to_numpy()
con_6=np.reshape(con_6,[1,2])
contreras_6_i15=con_6*rain_i15



con_8=contreras_8.to_numpy()
con_8=np.reshape(con_8,[1,2])
contreras_8_i15=con_8*rain_i15





np.random.seed(13)
m_6_i15=model_6_i15[np.random.permutation(len(model_6_i15)),:]
np.random.seed(13)
response=response[np.random.permutation(len(model_6_i15))]
# data_6_15=i_15_6[:,1:3]


np.random.seed(13)
m_8_i15=model_8_i15[np.random.permutation(len(model_8_i15)),:]
# m_6_i30=model_6_i30
# m_8_i30=model_8_i30

response=response


model_lm = LogisticRegression(penalty=None, solver = 'newton-cg', max_iter= 150,class_weight={0:1, 1:np.sqrt(1300/163)})
                              



s_6_i15=model_lm.fit(m_6_i15, response)
beta_lm_6_15=print(s_6_i15.intercept_,s_6_i15.coef_)
lm_pinal_6_i15 = s_6_i15.predict(pinal_6_i15 )
lm_tadpole_6_i15 = s_6_i15.predict(tadpole_6_i15)
lm_contreras_6_i15 = s_6_i15.predict(contreras_6_i15 )







s_8_i15=model_lm.fit(m_8_i15, response)
beta_lm_8_15=print(s_6_i15.intercept_,s_8_i15.coef_)
lm_pinal_8_i15 = s_8_i15.predict(pinal_8_i15 )
lm_tadpole_8_i15 = s_8_i15.predict(tadpole_8_i15)
lm_contreras_8_i15 = s_8_i15.predict(contreras_8_i15 )



pinal_8_i15=hr_8*rain_i15
tadpole_8_i15=zz_8*rain_i15






pinal_8_6_i15=np.vstack([rain, rain_i15b, lm_pinal_8_i15 ,lm_pinal_6_i15 ])
pinal_8_6_i15=np.transpose(pinal_8_6_i15)
pinal_8_6_i15=pd.DataFrame(pinal_8_6_i15)




tadpole_8_6_i15=np.vstack([rain, rain_i15b, lm_tadpole_8_i15 ,lm_tadpole_6_i15 ])
tadpole_8_6_i15=np.transpose(tadpole_8_6_i15)
tadpole_8_6_i15=pd.DataFrame(tadpole_8_6_i15)

contreras_8_6_i15=np.vstack([rain, rain_i15b, lm_contreras_8_i15 ,lm_contreras_6_i15 ])
contreras_8_6_i15=np.transpose(contreras_8_6_i15)
contreras_8_6_i15=pd.DataFrame(contreras_8_6_i15)






pinal_8_6_i15.columns =['rain intensity ','rain/4', 'SWA', 'SWB']
tadpole_8_6_i15.columns =['rain intensity ','rain/4', 'SWA', 'SWB']
contreras_8_6_i15.columns =['rain intensity ','rain/4', 'SWA', 'SWB']


    
    

# with pd.ExcelWriter("D:/Fire_jul24/watershead.xlsx") as writer:  
#     tadpole_8_6_i15.to_excel(writer,sheet_name='Tadpole',startrow=1, startcol=1)
#     pinal_8_6_i15.to_excel(writer,sheet_name='Pinal',startrow=1, startcol=1)
#     contreras_8_6_i15.to_excel(writer,sheet_name='Contreras',startrow=1, startcol=1)




