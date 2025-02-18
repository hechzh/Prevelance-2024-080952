import glob
import pandas as pd
from basefunc import getpopulation

# SDI1, SDI2, SDI3 represent the growth in the number of people, the growth in prevalence, and the growth in the number of people affected in SDI, respectively.
SDI1,SDI2,SDI3=pd.read_excel('..\\prevelance data\\SDI1.xlsx',index_col=[0]),pd.read_excel('..\\prevelance data\\SDI2.xlsx',index_col=[0]),pd.read_excel('..\\prevelance data\\SDI3.xlsx',index_col=[0])

# decomposition for all sex
for i,j in [('Both','both'),('Male','male'),('Female','female')]:
    # read the basic datas
    popt=getpopulation(j)
    prevelance_df=pd.read_excel(r'{}_prevelance_16_age_groups.xlsx'.format(i),index_col=[0,1])
    pop=pd.read_excel(r'{}_numbers_16_age_groups.xlsx'.format(i),index_col=[0,1])/prevelance_df
    
    # decom
    for before,after in [(2021,2050)]:
        Origin=(pop[before]*prevelance_df[before]).groupby(level=0).sum()
        factor1=Origin / (popt[2021].groupby(level=0).sum()) *(popt[2050].groupby(level=0).sum())
        factor2 = (pop[after]*prevelance_df[before]).groupby(level=0).sum()
        After = (pop[after]*prevelance_df[after]).groupby(level=0).sum()
        l,m,n=(factor1-Origin),(factor2-factor1),(After-factor2)
        l/=Origin
        m/=Origin
        n/=Origin
        SDIarea=['High SDI','High-middle SDI','Middle SDI','Low-middle SDI','Low SDI']
        R=pd.DataFrame([l,m,n],index=['population growth','population aging','changes in prevelance'])
        l,m,n=SDI1.loc[i],SDI2.loc[i],SDI3.loc[i]
        R.loc['population growth',SDIarea]=l.values-1
        R.loc['population aging',SDIarea]=m.values-l.values+1
        R.loc['changes in prevelance',SDIarea]=n.values-m.values
        R.to_excel('{}_Decomposition analysis_{}-{}.xlsx'.format(i,before,after))