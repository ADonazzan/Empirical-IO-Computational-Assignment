#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


python 3
"""
# =============================================================================
#
# This code implements demand estimation using a BLP algorithm.  
#
#lines 47-258 cleans and structures the Medicare data 
#lines 260-580 BLP algorithm
#
# =============================================================================
# =============================================================================
#    data : object containing variables for estimation and a data summary function. It  includes:
#         
#         demogr:   demographic variables with (T by ns*D) dimension
#         X1 :     product characteristics
#         X2 :     subset of X1 that gets random coefficients by interacting with 
#         the demographic and noise distributions 
#
#        X1_names, X2_names, D_names: names of variables from X1, X2
#                  and demographic variables respectively
#        
#         IV  :     Instruments with (TJ by K) dimension 
#         s_jt:    Observed market shares of brand j in market t ; (TJ by 1) dimension 
#         cdindex: Index of the last observation for a market (T by 1)
#         cdid:    Vector of market indexes (TJ by 1)
#         cdid_demogr: vector assigning market to each demographic observation
#         ns: number of simulated "indviduals" per market 
#         nmkt: number of markets
#         K1: number of product characteristics/elements in x1 
#         k2: number of elements in x2
#         TJ is number of observations (TJ = T*J if all products are observed in all markets)
#          v  :     Random draws given for the estimation with (T by ns*(k1+k2)) dimension
# Random draws. For each market ns*K2 iid normal draws are provided.
#        They correspond to ns "individuals", where for each individual
#        there is a different draw for each column of x2. 
# =============================================================================

#Please change here the directory where you stored the data  
dir= "docs/Python_codes/"

import pandas as pd
import numpy as np  

class data():        

        
#data location
    loc1= dir + "data.csv"
    loc2= dir+ "demogr.csv"
    loc3= dir + "demogr_means.csv"
    loc4= dir + "demogr_year_region_id.csv"
    loc5= dir + "demogr_iqr.csv"
    loc6= dir + "demogr_std.csv"

#load data
    data = pd.read_csv(loc1, sep=',', names= list(range(1,189)), encoding='cp1252')
    demogr =  pd.read_csv(loc2, sep=',', names= list(range(1,1001)), encoding='cp1252')
    demogr_means = pd.read_csv(loc3, sep=',', names= list(range(1,1001)), encoding='cp1252')
    demogr_year_region_id = pd.read_csv(loc4, sep=',', names= list(range(1,3)), encoding='cp1252')
    demogr_iqr= pd.read_csv(loc5, sep=',', names= list(range(1,1001)), encoding='cp1252')
    demogr_std= pd.read_csv(loc6, sep=',', names= list(range(1,1001)), encoding='cp1252')


#column names
    names =  ['id_colid_market_col','ms_col','dummy_firm_start','dummy_firm_end','dummy_year_start','dummy_year_end','dummy_region_start','dummy_region_end','dummy_firmname_start','dummy_firmname_end','dummy_macroregion_start','dummy_macroregion_end','price_col','deductible_col','n_medications_col','drug_frf_col','d_cov_gap_col','d_enhance_col','top_drug_col','in_area_flag_col','vintage_col','tier_12_col','d_below_lagged_col','ms_lis_firm_col','ms_lis_firm_lagged_col','id_col','id_market_col','ms_col','dummy_firm_start','dummy_firm_end','dummy_year_start','dummy_year_end','dummy_region_start','dummy_firmname_start','dummy_firmname_end','dummy_macroregion_start','dummy_macroregion_end','deductible_col','n_medications_col','drug_frf_col','d_cov_gap_col','d_enhance_col','op_drug_col','in_area_flag_col','vintage_col','tier_12_col','d_below_lagged_col','ms_lis_firm_col','ms_lis_firm_lagged_col','iv_start_col','iv_end_col','bid_col','gamma_col','true_weight_col','nat_weight_col','groupA_col','group_region_col','group_bid_test_col','ms_lischoosers_col','tot_lis_col','tot_lis_choosers_col','market_sizes_col','ms_lis_col','indicator_bench_col','contract_id_col','benefit_type_col','plan_name_col','lis_col'] 
    x1_names = ['id_colid_market_col','ms_col','dummy_firm_start','dummy_firm_end','dummy_year_start','dummy_year_end','dummy_region_start','dummy_region_end','dummy_firmname_start','dummy_firmname_end','dummy_macroregion_start','dummy_macroregion_end','deductible_col','n_medications_col','drug_frf_col','d_cov_gap_col','d_enhance_col','top_drug_col','in_area_flag_col','vintage_col','tier_12_col','d_below_lagged_col','ms_lis_firm_col','ms_lis_firm_lagged_col','id_col','id_market_col','ms_col','dummy_firm_start','dummy_firm_end','dummy_year_start','dummy_year_end','dummy_region_start','dummy_firmname_start','dummy_firmname_end','dummy_macroregion_start','dummy_macroregion_end','deductible_col','n_medications_col','drug_frf_col','d_cov_gap_col','d_enhance_col','op_drug_col','in_area_flag_col','vintage_col','tier_12_col','d_below_lagged_col','ms_lis_firm_col','ms_lis_firm_lagged_col'] 
    x2_names = ['price_col']
    D_names = list(range(1,1001))
    IV_names = ['iv_start_col','iv_end_col','bid_col','gamma_col','true_weight_col','nat_weight_col','groupA_col','group_region_col','group_bid_test_col','ms_lischoosers_col','tot_lis_col','tot_lis_choosers_col','market_sizes_col','ms_lis_col','indicator_bench_col','contract_id_col','benefit_type_col','plan_name_col','lis_col']


    id_col=1 #id market plan
    id_market_col=2 #id market
    ms_col=3 #market shares
    dummy_firm_start=4 #first column of firm id's dummies
    dummy_firm_end=79 #last column of firm id's dummies
    dummy_year_start=80 #first column of year's dummies
    dummy_year_end=85 #last column of year's dummies
    dummy_region_start=86 #first column of region's dummies
    dummy_region_end=119 #last column of region's dummies
    dummy_firmname_start=120 #first column of firm name 
    dummy_firmname_end=140 #last column of firm name
    dummy_macroregion_start=141 #first column of firm name 
    dummy_macroregion_end=143 #last column of firm name
    price_col=144
    deductible_col=price_col+1
    n_medications_col=price_col+2 #col 146 - try adding this to x1
    drug_frf_col=price_col+3
    d_cov_gap_col=price_col+4
    d_enhance_col=price_col+5
    top_drug_col=price_col+6
    in_area_flag_col=price_col+7
    vintage_col=price_col+8
    tier_12_col=price_col+9
    d_below_lagged_col=price_col+10
    ms_lis_firm_col=price_col+11
    ms_lis_firm_lagged_col=price_col+12


#LIST OF IV'S
    iv_start_col=price_col+13
    iv_end_col=iv_start_col+12
    bid_col=iv_start_col+15
    gamma_col=iv_start_col+16
    true_weight_col=iv_start_col+17
    nat_weight_col=iv_start_col+18
    groupA_col=iv_start_col+19
    group_region_col=iv_start_col+20
    group_bid_test_col=iv_start_col+21
    ms_lischoosers_col=iv_start_col+22
    tot_lis_col=iv_start_col+233
    tot_lis_choosers_col=iv_start_col+24
    market_sizes_col=iv_start_col+25
    ms_lis_col=iv_start_col+26
    indicator_bench_col=iv_start_col+27 #=1 if below benchmark plan
    contract_id_col=iv_start_col+28
    benefit_type_col=iv_start_col+29
    plan_name_col=iv_start_col+30
    lis_col=iv_start_col+31




# #cleaning
# #drop missing iv, dates, no-pharmacy plans


#missing IV
    data.loc[:,iv_start_col+3] = data.loc[:,iv_start_col+3].fillna(value=0)
    data.loc[:,iv_start_col+4]=data.loc[:,iv_start_col+4].fillna(value=0)
    data.loc[:,iv_start_col+5]=data.loc[:,iv_start_col+5].fillna(value=0)
    data.loc[:,iv_start_col+6]=data.loc[:,iv_start_col+6].fillna(value=0)
    data.loc[:,iv_start_col+11]=data.loc[:,iv_start_col+11].fillna(value=0)
#drop dates
    index = (data.loc[:,1] < 11000000) & (data.loc[:,1]>7000000)
    data = data.loc[index,:]
#drop plans with no pharmacy? check column 151 is this one
    index = (data.loc[:,in_area_flag_col]!= 0) & (data.loc[:,in_area_flag_col]<10000)
    data = data.loc[index,:]

#re-scaling
    scale=100
    ns=500
    data.loc[:,[price_col, deductible_col, n_medications_col, drug_frf_col, d_cov_gap_col, d_enhance_col, top_drug_col, vintage_col, tier_12_col,dummy_macroregion_start,142,dummy_macroregion_end]]=data.loc[:,[price_col, deductible_col, n_medications_col, drug_frf_col, d_cov_gap_col, d_enhance_col, top_drug_col, vintage_col, tier_12_col,dummy_macroregion_start,142,dummy_macroregion_end]].div(scale)
    data.loc[:,dummy_year_start:dummy_year_end]=data.loc[:,dummy_year_start:dummy_year_end].div(scale)
    data.loc[:,dummy_region_start:dummy_region_end]=data.loc[:,dummy_region_start:dummy_region_end].div(scale)
    data.loc[:,dummy_firmname_start:dummy_firmname_end]=data.loc[:,dummy_firmname_start:dummy_firmname_end].div(scale)
    data.loc[:,in_area_flag_col]=data.loc[:,in_area_flag_col].div(1000)

    demogr.loc[:,1:500]=demogr.loc[:,1:500].div(1000)
    demogr_means.loc[:,1:ns] = demogr_means.loc[:,1:ns].div(1000)

#IV - matrix    
    
    IV= data[list(range(iv_start_col,iv_end_col+1)) + [deductible_col, d_cov_gap_col, d_enhance_col, tier_12_col, top_drug_col, in_area_flag_col ,vintage_col]+list(range(dummy_macroregion_start,dummy_macroregion_end))+list(range(dummy_year_start+1,dummy_year_end-2))+[dummy_firmname_start, dummy_firmname_start+10, dummy_firmname_start+13, dummy_firmname_start+16, dummy_firmname_end]]
    #these columns give non singuality: 6-10
    IV = IV.iloc[:,0:6].join(IV.iloc[:,11:29])
    

#Cleaning missing IV
    index = (np.sum(IV.T.isna()) == 0)
    data = data.loc[index,:]   
    IV = IV.loc[index,:] 
    
 #reindexing IV 
    IV.index = pd.RangeIndex(len(IV.index))
    IV.columns= pd.RangeIndex(len(IV.columns))
        
         
# x1, x2
    constant=pd.Series(np.ones(np.shape(data[1])),index=data.index,name='constant') #add constant
    x11=data[[price_col, deductible_col, drug_frf_col, d_cov_gap_col, d_enhance_col, tier_12_col, top_drug_col, in_area_flag_col, vintage_col]+list(range(dummy_macroregion_start,dummy_macroregion_end))+list(range(dummy_year_start+1,dummy_year_end-2))+[dummy_firmname_start,dummy_firmname_start+10,dummy_firmname_start+13,dummy_firmname_start+15,dummy_firmname_start+16,dummy_firmname_end]]
    #try to add n_medications_col
    #x11=data[[price_col, deductible_col, drug_frf_col, d_cov_gap_col, d_enhance_col, tier_12_col, top_drug_col, in_area_flag_col, vintage_col,n_medications_col]+list(range(dummy_macroregion_start,dummy_macroregion_end))+list(range(dummy_year_start+1,dummy_year_end-2))+[dummy_firmname_start,dummy_firmname_start+10,dummy_firmname_start+13,dummy_firmname_start+15,dummy_firmname_start+16,dummy_firmname_end]]
 
    x1 = x11.join(constant)
    col=x1.columns.tolist()
    col= col[-1:] + col[:-1]
    x1=x1[col]
    
    
    x2 = data.loc[:,price_col].values.reshape(len(x1),1) 
    #x2 = data.loc[:,[price_col, deductible_col]].values.reshape(len(x1),2) #try to add another variable to x2

#Cleaning x1:eliminate dummies that are always 0 (companies that never offer a plan in 2006) 
    index= np.sum(x1)
    x1 = x1.loc[:,index!=0]

#reindexing
    x1.index = pd.RangeIndex(len(x1.index))#reindex x1
    x1.columns= pd.RangeIndex(len(x1.columns))

#indices
    id = data.loc[:,id_col] #id of each observation
    id_demo = data.loc[:,id_market_col].unique() #  number of markets 


# The  vector  below relates each observation to the market it is in / both for data and demographics
    
    cdid=list()
    cdindex=[-1]
    cdid_demogr=list()    

    for i in range(len(id_demo)):
        nbrand_market = np.sum(data.loc[:,id_market_col] == id_demo[i])
        cdid.extend(i*np.ones(nbrand_market)) 
        cdindex.append(cdindex[-1]+nbrand_market) 
        #create the cdid for the demographics
        index_market = data.loc[data[id_market_col]==id_demo[i]].index
        i_index_region= data.loc[index_market[0],dummy_region_start:dummy_region_end]==(1/scale) 
        index_region = np.where(i_index_region)[0] +1 # +1 since id_region [1,34]
        i_index_year= data.loc[index_market[0],dummy_year_start:dummy_year_end] == (1/scale)
        index_year = np.where(i_index_year)[0] + 2005 
        index_demogr=(demogr_year_region_id.loc[ demogr_year_region_id[1]==(index_year)[0]].index) & (demogr_year_region_id.loc[demogr_year_region_id[2] == index_region[0] ].index)
        if np.isnan(index_demogr.values)== 1:
            print(['Demographics not found for region ' + np.array2string(index_region) + ' and year ' + np.array2string(index_year)])
        else:
            cdid_demogr.extend(index_demogr.values*np.ones(nbrand_market,int))## [0,135]
        
    cdindex=cdindex[1:137]
    cdindex=np.asarray(cdindex,dtype=int)

#create instruments based on demographics
    mean_income= demogr_means.loc[cdid_demogr,1]
    mean_diffall= demogr_means.loc[cdid_demogr,ns+1] 
    std_income = demogr_std.loc[cdid_demogr,1]
    iqr_income=demogr_iqr.loc[cdid_demogr,1] 
    iqr_diffall=demogr_iqr.loc[cdid_demogr,ns+1]#

    IV1 = pd.DataFrame(np.concatenate([np.array([IV.loc[:,13]])*np.array([mean_income]), np.array([IV.loc[:,13]])*np.array([mean_diffall]), np.array([IV.loc[:,13]])*np.array([iqr_income]), np.array([IV.loc[:,13]])*np.array([iqr_diffall])],axis=0).T, index=IV.index, columns=[24,25,26,27])
    IV=IV.merge(IV1, right_index=True, left_index=True) #Try to run without these IV


#other variables
    K1 = x1.shape[1] #number of product characteristics
    K2 = x2.shape[1] #
    s_jt=data.loc[:,ms_col]
    s_jt= np.reshape(s_jt.values, (np.shape(s_jt)[0],1))#observed market share for product j at time t
    nmkt = len(id_demo)
    r='/Users/sofiateles/Dropbox/PythonDemand/BLP algo/V.csv'
    v=pd.read_csv(r,sep=',',names= list(range(1000)))

    #v = pd.DataFrame(np.random.randn(nmkt,demogr.shape[1]))


   
    def summary(self):
        print('the number of characteristics in X1 is: '+str(data.x1.shape[1]))
        print('the number of characteristics in X2 is: '+str(data.x2.shape[1]))
        print('Object cdid dimensions: ' +str(np.shape(data.cdid)) )
        print('The dimensions of object cdindex are      ' + str(np.shape(data.cdindex)))
        print('The number of instruments for the price is ' +  str(np.shape(data.IV)[1]))
        print('The dimensions of object IV are ' + str(np.shape(data.IV)))
        print('size id_demo: '+ str(np.shape(data.id_demo)) )
        print("size of cdid is " + str(np.shape((data.cdid))))
        print("size of cdindex is "+ str(np.shape((data.cdindex))))
        print("size of cdid_demogr is "+ str(np.shape((data.cdid_demogr))))



data.summary(data)

# =============================================================================
# BLP Algorithm
#
#It is based on Nevo(2001) matlab code, Prof. Decarolis matlab version, and Daria Pus code.  
#
#note that it uses a Nelder-Mead algorithm to minimize objective function,as in the original; it is a simplex algo, i.e. with no derivatives
#
# OUTPUTS:
# 
# csv file with:
# theta1: vector of estimated parameters for the indiriect utility function, length: # of plans's characteristics 
# theta2: matrix of estimated interaction with demographicsparameters for
#         the indiriect utility function, # rows: of plans's characteristics
#        interacted, # columns: # of demographics
# se: matrix of standard errors
#
# histogram:
#   alfa_i_reshaped: estimated distribution of the price coefficient
#                 (interacted with the demographics)
# =============================================================================

#from scipy.optimize import fmin
from scipy.optimize import minimize
import matplotlib.pyplot as plt    
import time
#import optimize2 as opt # import modified optimize.minimize_nelder-mead ; alternatively change line 569 from 'and' to 'or' on original optimize.py

    
class BLP:
    
    def __init__ (self,data,theta2w,mtol,niter): #define data instances
         
        self.niter = niter 
        self.mtol=mtol
        self.theta2w=theta2w 
        self.x1 = data.x1
        self.x2 = data.x2 
        self.s_jt = data.s_jt
        self.v = data.v
        self.cdindex = data.cdindex #consider asarray
        self.cdid = data.cdid
        self.cdid_demogr=data.cdid_demogr
        self.vfull = data.v.loc[self.cdid_demogr,:] 
        dfull = data.demogr.loc[self.cdid_demogr,:]
        dfull.columns= pd.RangeIndex(len(dfull.columns))
        dfull.index = pd.RangeIndex(len(dfull.index))
        self.dfull = dfull
        self.demogr= data.demogr
        self.IV = data.IV
        self.K1 = self.x1.shape[1]
        self.K2 = self.x2.shape[1]
        self.ns = data.ns          # number of simulated "indviduals" per market 
        self.D = int(self.dfull.shape[1]/self.ns)       # number of demographic variables
        self.T = np.asarray(self.cdindex).shape[0]      # number of markets = (# of cities)*(# of quarters)  
        self.TJ = self.x1.shape[0]      #number of observations
        self.J = int(self.TJ/self.T) #average number of observation per market
        
      
        
        self.invA = np.linalg.inv(self.IV.T@self.IV)
        
        
        #mid2 = np.mat(self.x1.T)*np.mat(self.IV)*self.invA*np.mat(self.IV.T)*np.mat(self.x1)
        #invmid2=np.linalg.inv(mid2)
        
        # Calculating s0 -outside good shares
        temp= self.s_jt.cumsum()
        sum1 = temp[self.cdindex] 
        sum1[1:np.shape(sum1)[0]] = np.diff(sum1) 
        outshr = (1 - sum1[np.asarray(self.cdid,dtype=int)]) 
        outshr= np.abs(outshr)
        outshr=np.reshape(outshr,(np.shape(outshr)[0],1))       

        #find initial guess of coefficients by running simple logit regression
        y=np.log(self.s_jt)-np.log(outshr)
        mid = self.x1.T@self.IV@self.invA@self.IV.T
        t = np.linalg.inv(mid@self.x1)@mid@y 
        
        #mean utility - initial
        self.mvalold = self.x1@(t) 
        self.mvalold = np.exp(self.mvalold).values.reshape(np.shape(self.mvalold)[0],1)
        self.oldt2 = np.zeros(np.shape(self.theta2w)) #initial old2
        self.gmmvalold = 0
        self.gmmdiff = 1
        self.gmmresid = np.ones((6024,1)) #@@@@@@ initialization
   
    #transforms inputed guess of theta2w into an ndarray type
    def init_theta(self,theta2w):
        theta2w=theta2w.reshape(self.K2,1+self.D) # #rows: # x2 variables
        self.theti, self.thetj = list(np.where(theta2w != 0))#columns: #demographic variables + 1 (constant)
        self.theta2 = theta2w[np.where(theta2w != 0)]
        return self.theta2
     
        
    #mean utility function
    def mufunc(self,theta2w): 
        mu = np.zeros((self.TJ, self.ns))
        for i in range(self.ns):
            v_i = np.array(self.vfull.loc[:, np.arange(i, self.K2*self.ns, self.ns)])
            d_i = np.array(self.dfull.loc[:, np.arange(i, self.D*self.ns, self.ns)]) 
            temp = d_i @ theta2w[:, 1:(self.D+1)].T
            mu[:, i]=(np.multiply(self.x2, v_i) @ theta2w[:, 0]) + np.multiply(self.x2, temp) @ np.ones((self.K2))

        return mu
        
    def ind_sh(self,expmu): 
        eg = np.multiply(expmu, np.kron(np.ones((1, self.ns)), self.mvalold)) 
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[np.asarray(self.cdid,dtype=int), :]
        
        return np.multiply(eg, denom)

    def mktsh(self,expmu):
        # compute the market share for each product
        temp = self.ind_sh(expmu).T
        f = (sum(temp) / float(self.ns)).T
        f = f.reshape(self.x1.shape[0],1)
        return f
        
        
    def meanval(self,theta2):
        #phased tolerance to speed up computation
#        if self.gmmdiff < 1e-15:
#            self.mtol = 1e-15
#        elif self.gmmdiff < 1e-3:
#            self.mtol = 1e-4
#        else:
#            self.mtol = 1e-2
        
        if np.ndarray.max(np.absolute(theta2-self.oldt2)) < 0.01: 
            tol = self.mtol
            flag=0
        else:
            tol = self.mtol #when theta2 (output of minimize) is still larger than 0.01, flag=1 => pass it to theta old2, that will be used in next iteration
            flag = 1
        norm = 1
        avgnorm = 1
        i = 0
        theta2w = np.zeros((self.K2,self.D+1)) 
        for ind in range(len(self.theti)): 
            theta2w[self.theti[ind], self.thetj[ind]] = theta2[ind] 
        expmu=np.exp(self.mufunc(theta2w))
       
        while (norm > self.mtol) & (i<self.niter): 
           
            pred_s_jt = self.mktsh(expmu) 
            self.mval = np.multiply(self.mvalold,self.s_jt) / pred_s_jt 
            t = np.abs(self.mval - self.mvalold)
            norm = np.max(t)
            avgnorm = np.mean(t)
            self.mvalold = self.mval
            i += 1
           
            if (norm > self.mtol) & (i > self.niter-1):
                print('Max number of ' + str(niter) + 'iterations reached')
    
        print(['# of iterations for delta convergence: ' , i])
	
        if (flag == 1) & (sum(np.isnan(self.mval)))==0: 
            self.mvalold = self.mval
            self.oldt2 = theta2
        return np.log(self.mval)
 
 
    # calculates the jacobian of the 
    def jacob(self,theta2): 
        cdindex=np.asarray(self.cdindex,dtype=int)
        theta2w = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = self.theta2[ind] #build theta2w (from array to ndarray) to plug in ind_sh
        expmu=np.exp(self.mufunc(theta2w)) 
        shares = self.ind_sh(expmu)
        f1 = np.zeros((np.asarray(self.cdid).shape[0] ,self.K2 * (self.D + 1)))
        # calculate derivative of shares with respect to the first column of theta2w (variable that are not interacted with demogr var, sigmas)
        for i in range(self.K2):
            xv = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1,self.ns)),self.v.loc[self.cdid, self.ns*i:self.ns * (i+1)-1])
            temp = np.cumsum(np.multiply(xv, shares), 0).values
            sum1 = temp[cdindex, :]
            sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
            f1[:,i] = np.mean((np.multiply(shares, xv - sum1[np.asarray(self.cdid,dtype=int),:])),1) #mean over columns
           
        for j in range(self.D):
            d = self.demogr.loc[self.cdid,self.ns*(j)+1:self.ns*(j+1)] 
            temp1 = np.zeros((np.asarray(self.cdid).shape[0],self.K2))
            for i in range(self.K2):
                xd = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1,self.ns)), d)
                temp = np.cumsum(np.multiply(xd, shares), 0).values
                sum1 = temp[cdindex, :]
                sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
                temp1[:, i] = np.mean((np.multiply(shares, xd-sum1[np.asarray(self.cdid,dtype=int), :])), 1)
            f1[:,self.K2 * (j + 1):self.K2 * (j + 2)] = temp1 

        self.rel = self.theti + self.thetj * (max(self.theti)+1) 
        f = np.zeros((np.shape(self.cdid)[0],self.rel.shape[0]))
        n = 0
        
        for i in range(np.shape(self.cdindex)[0]):
            temp = shares[n:(self.cdindex[i] + 1), :]
            H1 = temp @ temp.T
            H = (np.diag(np.array(sum(temp.T)).flatten())-H1) / self.ns
            f[n:(cdindex[i]+1),:] = np.linalg.inv(H) @ f1[n:(cdindex[i] + 1),self.rel]
            n = cdindex[i] + 1
        return f
       
    # compute GMM objective function       
    def gmmobj(self,theta2):
        print(theta2)
        delta = self.meanval(theta2)
        self.theta2=theta2
        # the following deals with cases where the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(delta)) == 1: 
            f = np.ndarray((1,1),buffer=np.array([1e+10,1e+10])) 
        else:
            temp1 = self.x1.T @ self.IV
            temp2 = delta.T@self.IV
            self.theta1 = np.linalg.inv(temp1@self.invA@temp1.T)@temp1@self.invA@temp2.T
            self.gmmresid = delta - self.x1@self.theta1
            temp1 = self.gmmresid.T@(self.IV) 
            f = temp1@self.invA@temp1.T
            f=f.values
            if np.shape(f) > (1,1):
                temp = self.jacob(theta2).T 
                df = 2*temp@self.IV@self.invA@self.IV.T@self.gmmresid 
        print('fval:', f[0,0]) #value of the gmm objective function
        
        #to be used in meanval to phase tolerance based on gmm value
        self.gmmvalnew = f[0,0]
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        
        return (f[0,0])
       
               

    #calculates the gradient of the objective function based on jacobian
    def gradobj(self,theta2):
        temp = self.jacob(theta2).T 
        df = 2*temp@self.IV@self.invA@self.IV.T@self.gmmresid 
        df=df.values ##@@
        print('this is the gradient '+str(df))
    
        return df
    
  
    #WARNING!!!!!!!! In some versions of scipy there can be a problem with xatol in optimize.minimize_neldermead. IF so, change optimize.py codeline 569: replace 'and' to 'or'OR use modified script optimize2.py 
    
     
    # maximizes the objective function
    def iterate_optimization(self,opt_func, param_vec,jac,options ): 
        res = minimize(opt_func, param_vec,method='nelder-mead',options=options) 
        ##res = opt._minimize_neldermead(opt_func, param_vec,disp=True,maxiter=100,fatol=0.0001,xatol=0.0001)
        #res = minimize(opt_func, param_vec, method='BFGS', bounds=None, callback=None, options=options)
    
        return res
            
    
    #calculates the variance-covariance matrix of the estimates
    
    def varcov(self,theta2): 
        Z = self.IV.shape[1]
        temp = self.jacob(theta2)                                              
        a = np.concatenate((self.x1.values, temp), 1).T @ self.IV.values
        IVres = np.multiply(self.IV.values,self.gmmresid.values @ np.ones((1, Z)))
        b = IVres.T@(IVres)
        if np.linalg.det(a @ np.asarray(self.invA) @ a.T)!= 0:
            inv_aAa = np.linalg.inv(a @ np.asarray(self.invA) @ a.T)
        else:
            print('Error: singular matrix in covariate function ; forced result') #if jacobian has row of zeros
            inv_aAa = np.linalg.lstsq(a @ np.asarray(self.invA) @ a.T)
        f = inv_aAa @ a @ np.asarray(self.invA) @ b @ np.asarray(self.invA) @ a.T @ inv_aAa   
        return f    


    # computes estimates of the coefficients and its standard errors; outputs a histogram and a csv file with estimates
    def results(self,theta2):
        self.theta1=self.theta1.values
        var = self.varcov(self.theta2)
        se = np.sqrt(var.diagonal())
        t = se.shape[0] - self.theta2.shape[0]
        
        print('Object vcov dimensions: ' + str(np.shape(var)) )
        print('Object se dimensions: ' + str(np.shape(se)))
        
        print('Object theta2w dimensions:     ' + str(np.shape(self.theta2)))
        print('Object t dimensions:     ' + str(np.shape(t)))
        

#histogram of the coefficients
        alfa_i=[]
        alfa_i2=[]
        for i in range(0,self.T): 
            data_market=np.reshape(self.demogr.loc[i,0:self.ns*self.D].values,(self.ns,self.D))
            v_market=np.reshape(self.v.loc[i,0:self.ns-1].values,(self.ns,1)) # caution: v is created to have 1000 columns
            alfa_i2.extend(np.add(data_market@(self.theta2[1:3]).T, self.theta2[0]*v_market[:,0])) 
            alfa_i.extend(data_market@(self.theta2[1:3].T))
        #alfa_i=(self.theta1[1]+alfa_i2)/100 
        alfa_i=(self.theta1[1]+alfa_i2)/100
        h=plt.figure()
        plt.hist(alfa_i,bins=25,range=(np.min(alfa_i),np.max(alfa_i)))
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of the price coefficient -2007')
        h.savefig('elasticities.png')
        #compute the elasticities
        #reshape the alfas
        alfa_i_reshaped=np.reshape(alfa_i,(self.ns,len(self.cdindex))).T #changed from cdindex (136) to cdid (6024) to match matlab size
        alfa_i_reshaped=alfa_i_reshaped[np.asarray(self.cdid,dtype=int),:]
        
        #convert theta2 into ndarray to plug in ind_sh
        theta = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
            theta[self.theti[ind], self.thetj[ind]] = self.theta2[ind] 
       #######
        expmu=np.exp(self.mufunc(theta))
        f = self.ind_sh(expmu)
        mval=(self.x1@self.theta1)
     
        #from ind_sh function: here mvalold = gmmresid +mval
        eg = np.multiply(np.exp(self.mufunc(theta)), np.kron(np.ones((1, self.ns)), (mval+self.gmmresid))) 
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[np.asarray(self.cdid,dtype=int), :]
        f22= np.multiply(eg, denom)
        ######
        f2 = np.sum(f22.T,0)/self.ns
        f2 = f2.T
        error=self.s_jt-f2     

        #table with parameters + export to csv file
        
        self.theta1_results= pd.DataFrame({'Theta1':self.theta1.reshape(self.K1,),'Std.Error_theta1': se[0:-self.theta2.shape[0]]},columns=(['Theta1','Std.Error_theta1'])) 
        self.theta2_results= pd.DataFrame({'Theta2':self.theta2.reshape(self.theta2.shape[0],), 'Std.Error_theta2':se[-self.theta2.shape[0]:]},columns=(['Theta2','Std.Error_theta2']))
        self.theta1_results.to_csv(dir + "theta1.csv")
        self.theta2_results.to_csv(dir + "theta2.csv")
        self.gmmvalold
        
 
# =============================================================================
# Define parameters to input in the algorithm    
# =============================================================================

if __name__ == '__main__':
    starttime = time.time()

    #maximum number of iterations for convergence of mval-mvalold contraction
    niter = 2500

    #initial guess for coefficients
    theta2w = np.array([0.50,-0.5,0.5])
    #theta2w = np.array([0.20,-0.01,0.01,-0.5,0.05,0.05])

    #maximum tolerance
    mtol= 1e-5

    #set optimization options
    options={'disp': None,'maxiter': 100,'xatol':0.0001,'fatol':0.0001}

    #get output
    blp = BLP(data,theta2w,mtol,niter)
    init_theta = blp.init_theta(theta2w)
    res = blp.iterate_optimization(opt_func=blp.gmmobj,param_vec=init_theta,jac=blp.gradobj,options=options)
    blp.results(res)

    endtime=time.time()
    run = endtime-starttime
    print('running time: ' + str(endtime-starttime))
    a=time.time()
