# -*- coding: utf-8 -*-
"""



"""

# =============================================================================
# 
#  Demand Estimation - Logit Model
# 
# =============================================================================

#Libraries
##############################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

#directory -please change
#%%

loc= "/PartD.csv"


#upload data;select variables of interest

data_raw = pd.read_csv(loc,sep=',',encoding='cp1252')
dt = data_raw.loc[:,('regionid','year', 'enrol_tot_num','enrol_lis_num','pdp', 'Part_D_Total_Premium','basic','Part_D_Drug_Deductible','unrestricted_drug','in_area_flag','Parent_Organization','Part_D_Basic_Premium')]

#have a look at the structure of the following variables:
#dt.describe()     
dt[['regionid','year']].describe()

#or use groupby
dt.groupby(['regionid','year']).enrol_tot_num.sum()
dt.groupby(['regionid','year','pdp']).enrol_tot_num.sum()


#note here there is a problem with missing variables
dt['enrol_lis_num'].isna().sum()
dt=dt.loc[dt['enrol_lis_num'].isna()==0,:] #excluding all observations with missing values


#create dummies for Parent organization and region (need to delete one)
dummies_organization=pd.get_dummies(dt.loc[:,'Parent_Organization'] ).drop('XLHealth Corporation',axis=1) 
dummies_region=pd.get_dummies(dt.loc[:,'regionid'] ).drop(34,axis=1) 
dummies_year=pd.get_dummies(dt.loc[:,'year']).drop(2013,axis=1)   

dt= dt.join([dummies_organization,dummies_region,dummies_year])

#market shares

dt['mkt_id']= dt['regionid'].apply(str)+dt['year'].apply(str) #define market id region-year

dt['demand'] = dt.enrol_tot_num - dt.enrol_lis_num #calculate s_numerator (market share numerator for all j (j=0 included) in all R markets and add to data table

s0 = dt.loc[dt['pdp']==0].groupby('mkt_id').sum().demand #select "outside good" plans - demand for outside good plans
s = dt.groupby('mkt_id').sum().demand  #total demand in each market

mj= dt.loc[dt['pdp']==1].join(s, on='mkt_id',rsuffix='_total').join(s0,on='mkt_id',rsuffix='_0') #select plans pdp==1 (inside goods) add column with corresponding market total and outside good total
m= mj.assign(mkt_sh_j= lambda x: x['demand']/x['demand_total']).assign(mkt_sh_0= lambda x: x['demand_0']/x['demand_total'])



# =============================================================================
# #OLS regression
# =============================================================================
#defining y, x
y0 = np.log(m['mkt_sh_j'])-np.log(m['mkt_sh_0'] ) # to avoid divide by zero warning 
y = y0.mask(np.isinf(y0)| np.isnan(y0)) #ignore inf or nan
y.name='y'
x = m.loc[:,('Part_D_Total_Premium','basic','Part_D_Drug_Deductible','unrestricted_drug','in_area_flag')]
dummies = m.iloc[:,12:362]
#x=x.join([dummies_region,dummies_year])
x=sm.add_constant(x)

model = sm.OLS(y,x, missing='drop').fit() #model
print(model.summary()) #results 

# =============================================================================
#2SLS -  Hausman instrument
# =============================================================================

#Hausman IV
IV=dt.loc[dt['pdp']==1,('Part_D_Total_Premium','Parent_Organization','basic','regionid','year')]
#elements to exclude from the sum - own market values
temp1=IV.groupby(['Parent_Organization','basic','regionid','year']).sum()
temp2=IV.groupby(['Parent_Organization','basic','regionid','year']).count()
IV=IV.join(temp1,on=['Parent_Organization','basic','regionid','year'],rsuffix='_mktsum')
IV=IV.join(temp2,on=['Parent_Organization','basic','regionid','year'],rsuffix='_mktcount')

#totals
temp3=IV.groupby(['Parent_Organization','basic','year']).sum().Part_D_Total_Premium
temp4=IV.groupby(['Parent_Organization','basic','year']).count().Part_D_Total_Premium
IV=IV.join(temp3,on=['Parent_Organization','basic','year'],rsuffix='_allsum')
IV=IV.join(temp4,on=['Parent_Organization','basic','year'],rsuffix='_allcount')


ivhaus=(IV.Part_D_Total_Premium_allsum-IV.Part_D_Total_Premium_mktsum).div(IV.Part_D_Total_Premium_allcount-IV.Part_D_Total_Premium_mktcount) #some parent organization only have one market
ivhaus.name='ivhaus'
#IV=IV.assign(ivhaus= lambda x: (x['Part_D_Total_Premium_allsum']-x['Part_D_Total_Premium_mktsum'])/(x['Part_D_Total_Premium_allcount']-x['Part_D_Total_Premium_mktcount']))


#This gives an idea of what 2sls does, but standard errors are not properly dealt with
#First Stage
m2=x.join(ivhaus).join(y)
m2=m2.dropna() #!! Dropped missing values!!


#This gives an idea of what 2sls does, but standard errors are not properly dealt with
#First Stage
model1s = sm.OLS(m2['Part_D_Total_Premium'],m2[['Part_D_Drug_Deductible','unrestricted_drug','ivhaus']],missing='drop').fit()
print(model1s.summary())
#Second Stage
d=model1s.predict()
m2['predicted_Part_D_Total_Premium']= d
x2=m2.drop(['ivhaus','Part_D_Total_Premium','y','basic'],axis=1)
model21 = sm.OLS(m2.y,x2, missing='drop').fit() #drop ivhaus and premium 
print(model21.summary())


#Statsmodel package function:
exo=m2.drop(['ivhaus','y','basic','predicted_Part_D_Total_Premium'],axis=1)
end=m2[['const','ivhaus','Part_D_Drug_Deductible', 'unrestricted_drug', 'in_area_flag']]
model22=IV2SLS(m2.y,exo,end).fit()
print(model22.summary())


# =============================================================================
# Nested Logit
# =============================================================================
#two nests based on basic
total_nest=m.groupby(['regionid','year','basic']).sum().mkt_sh_j 
m3= m.join(total_nest, on=['regionid','year','basic'], how='left',rsuffix='_nest')
m3=m3.assign(s_nest= lambda x: np.log(x['mkt_sh_j']/x['mkt_sh_j_nest']))
m3=x.join(ivhaus).join(y).join(m3['s_nest'])

m3=m3.dropna()
#simple regression
x3=m3[['const', 'Part_D_Total_Premium', 'basic', 'Part_D_Drug_Deductible','unrestricted_drug', 'in_area_flag', 's_nest']]
model2 = sm.OLS(m3.y,x3, missing='drop').fit() #model
print(model2.summary()) #results 

#IV - 2sls

#create BLP instrument based on Part_D_Drug_Deductible, no particular reason why this one
ivblp=dt.loc[dt['pdp']==1,('Part_D_Drug_Deductible','basic','regionid','year')]

#elements to exclude from the sum - own market values
temp1=ivblp.groupby(['basic','regionid','year']).sum()
temp2=ivblp.groupby(['basic','regionid','year']).count()
ivblp=ivblp.join(temp1,on=['basic','regionid','year'],rsuffix='_mktsum')
ivblp=ivblp.join(temp2,on=['basic','regionid','year'],rsuffix='_mktcount')
#totals
temp3=ivblp.groupby(['basic','year']).sum().Part_D_Drug_Deductible
temp4=ivblp.groupby(['basic','year']).count().Part_D_Drug_Deductible
ivblp=ivblp.join(temp3,on=['basic','year'],rsuffix='_allsum')
ivblp=ivblp.join(temp4,on=['basic','year'],rsuffix='_allcount')

ivblp=(ivblp.Part_D_Drug_Deductible_allsum-ivblp.Part_D_Drug_Deductible_mktsum).div(ivblp.Part_D_Drug_Deductible_allcount-ivblp.Part_D_Drug_Deductible_mktcount) #some parent organization only have one market
ivblp.name='ivblp'

m3=m3.join(ivblp)
m3=m3.dropna()

exo=m3.drop(['ivhaus','ivblp','y','basic','Part_D_Drug_Deductible'],axis=1)
end=m3[['const','ivhaus', 'unrestricted_drug', 'in_area_flag','ivblp']]
model3=IV2SLS(m3.y,exo,end).fit()
print(model3.summary())
