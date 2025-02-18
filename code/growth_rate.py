import pandas as pd
from basefunc import bootstrap_prediction2
# calculate grwoth rate of crude_prevelance and number of cases for both sex
for sex in ["Both"]:
    for content in ["numbers","crude_prevalence"]:
        df_up=pd.read_excel(r'{}_{}_upper.xlsx'.format(sex,content),index_col='location_name')[[2021,2050]]
        df_lo=pd.read_excel(r'{}_{}_lower.xlsx'.format(sex,content),index_col='location_name')[[2021,2050]]
        lb,ub,mb,lr,ur=bootstrap_prediction2(df_lo,df_up)
        df_rate=pd.DataFrame(index=df_lo.index)
        df_rate['lower95']=lr-1
        df_rate['upper95']=ur-1
        df_rate.to_excel(sex+"sex_percentage change of"+content+'2021-205095UI.xlsx')