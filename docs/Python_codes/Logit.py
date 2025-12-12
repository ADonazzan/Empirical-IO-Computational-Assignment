#!/usr/bin/env python3
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

#market shares
dt['mkt_id']= dt['regionid'].apply(str)+dt['year'].apply(str)
#dt['mkt_id2']= (dt['regionid']*10000)+dt['year'] #define market id region-year

dt['demand'] = dt.enrol_tot_num - dt.enrol_lis_num #calculate s_numerator (market share numerator for all j (j=0 included) in all R markets and add to data table

s0 = dt.loc[dt['pdp']==0].groupby('mkt_id').sum().demand #select "outside good" plans - demand for outside good plans
s = dt.groupby('mkt_id').sum().demand  #total demand in each market

mj= dt.loc[dt['pdp']==1].join(s, on='mkt_id',rsuffix='_total').join(s0,on='mkt_id',rsuffix='_0') #select plans pdp==1 (inside goods) add column with corresponding market total and outside good total
m= mj.assign(mkt_sh_j= lambda x: x['demand']/x['demand_total']).assign(mkt_sh_0= lambda x: x['demand_0']/x['demand_total'])
