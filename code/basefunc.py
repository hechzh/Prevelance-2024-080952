import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# Basic settings for age list and file path
part_data = glob.glob(r'..\population data\IHME_GBD_2019_POP_1990_2019\*') + \
    [r'..\population data\IHME_POP_2017_2100_POP_REFERENCE\IHME_POP_2017_2100_POP_REFERENCE_Y2020M05D01.CSV']
PD_GBD = pd.read_csv('..\\prevelance data\\CountryPrevelanceData2.csv')
locations=np.unique(PD_GBD['location_name'])
age_list=['1 to 4']+[str(5*i)+' to '+str(5*i+4) for i in range(1,19)]+['95 plus']

# read population data for determined sex
def getpopulation(sex,value='val'):
    population_df=[]
    for csv_path in part_data[:-1]:
        df=pd.read_csv(csv_path)
        df=df[df['age_group_name'].isin(age_list)]#pd.MultiIndex.from_product([])
        df['location_name'][df['location_name']=='Turkey']='Türkiye'
        df=df[df['location_name'].isin(locations)]
        df=df[df['sex_name']==sex]
        pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values=value)
        population_df.append(pivot_df)
    for csv_path in part_data[-1:]:
        df=pd.read_csv(csv_path)
        df=df[df['age_group_name'].isin(age_list)]#pd.MultiIndex.from_product([])
        df['location_name'][df['location_name']=='Turkey']='Türkiye'
        df=df[df['location_name'].isin(locations)]
        df=df[df['year_id']>=2020]
        if sex == 'both' and sex not in df['sex_name']:
            pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values='val',aggfunc=sum)
        else:
            df=df[df['sex_name']==sex]
            pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values='val')
        population_df.append(pivot_df)
    population_df=pd.concat(population_df,axis=1)
    return population_df

# predict future values
def doprediction(dfl, years, future_years):
    df = dfl[years].copy()
    K = df.columns[-1]
    model = LinearRegression()
    log_values = np.log(df.values)
    for index, row in df.iterrows():
        y = np.log(row.values)
        yx = years
        yxf = future_years
        if np.sum(np.isnan(yx)) == 0 and np.sum(np.isnan(y)) == 0:
            yx = [[i] for i in yx]
            yxf = [[i] for i in yxf]
            model.fit(yx, y)
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]/np.exp(future_log_pred[0])
            future_pred = np.exp(future_log_pred)*multiplier
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df

# bootstrap to get the prediction result
# UIs may vary slightly across multiple runs.
def bootstrap_prediction2(df_lower, df_upper, n_iterations=1000):
    predictions = []
    rate = []
    for _ in range(n_iterations):
        df = df_lower + (df_upper - df_lower) * np.random.rand(*df_lower.shape)
        if 2050 not in df.columns:
            years = np.arange(1990, 2022)
            future_years = np.arange(2021, 2051)
            df = doprediction(df, years, future_years)
        predictions.append(df.values)
        rate.append((df[2050]/df[2021]).values)

    predictions = np.array(predictions)

    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    mid_bound = np.percentile(predictions, 50.0, axis=0)
    lr = np.percentile(rate, 2.5, axis=0)
    ur = np.percentile(rate, 97.5, axis=0)
    return lower_bound, upper_bound, mid_bound, lr, ur