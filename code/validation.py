from statsmodels.tsa.arima.model import ARIMA
from basefunc import getpopulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

#Basic Function Module
#This module provides several commonly used statistical functions for calculating the deviation, root mean square error (RMSE), and residual sum of squares (RSS)  between two arrays.


def bias(array1, array2):
    #calculate bias of two arrays
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")
    return np.nanmean(array1 - array2)


def rmse(array1, array2):
    #calculate rmse of two arrays
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")
    return np.sqrt(np.nanmean((array1 - array2) ** 2))


def rss(array1, array2):
    #calculate rss of two arrays
    return np.nansum((array1 - array2) ** 2)

# Data Loading and Preprocessing Module
# This module is used to load the population data file and preprocess the data, including age grouping, standardizing country/region names, etc.

# Get the data file path
part_data = glob.glob(r'..\population data\IHME_GBD_2019_POP_1990_2019\*') + \
    [r'..\population data\IHME_POP_2017_2100_POP_REFERENCE\IHME_POP_2017_2100_POP_REFERENCE_Y2020M05D01.CSV']
# Define age groups
age_list = ['1 to 4']+[str(5*i)+' to '+str(5*i+4)
                       for i in range(1, 19)]+['95 plus']
age_list_small = age_list[4:]
# +[str(i)+'-'+str(i+4) for i in range(80,91,5)]+['95+ years']
age_list2 = [str(i)+'-'+str(i+4) +
             ' years' for i in range(20, 91, 5)]+['95+ years']

# Load prevalence data of countries
PD_GBD = pd.read_csv('..\\prevelance data\\CountryPrevelanceData2.csv')

# Correct the names of specific countries/regions
locations = np.unique(PD_GBD['location_name'])
locations[np.where("C么te d'Ivoire" == locations)] = "Côte d'Ivoire"
locations_old = locations.copy()
replacement_mapping = {"Micronesia (Fed. States of)": "Micronesia (Federated States of)",
                       'State of Palestine': 'Palestine',
                       "Dem. People's Republic of Korea": "Democratic People's Republic of Korea",
                       "C么te d'Ivoire": "Côte d'Ivoire",
                       'China, Taiwan Province of China': 'Taiwan (Province of China)',
                       'T眉rkiye': 'Turkey',
                       'Türkiye': 'Turkey',
                       'Central African Rep': 'Central African Republic',
                       'Dem Rep Congo': 'Democratic Republic of the Congo',
                       'Iran': 'Iran (Islamic Republic of)',
                       'Laos': 'Lao People\'s Democratic Republic',
                       'Northern Mariana Is': 'Northern Mariana Islands',
                       'Saint Kitts and Nev': 'Saint Kitts and Nevis',
                       'Saint Vincent and t': 'Saint Vincent and the Grenadines',
                       'Sao Tome and Princi': 'Sao Tome and Principe',
                       'South Korea': 'Republic of Korea',
                       'Syrian Arab Republi': 'Syrian Arab Republic',
                       'United Arab Emirate': 'United Arab Emirates',
                       'United States Virgi': 'United States Virgin Islands',
                       'Venezuela': 'Venezuela (Bolivarian Republic of)',
                       'Brunei': 'Brunei Darussalam',
                       'Cape Verde': 'Cabo Verde',
                       'Democratic Republic of Congo': 'Democratic Republic of the Congo',
                       'Moldova': 'Republic of Moldova',
                       'World': 'Global'
                       }
location_all = np.unique(pd.read_csv(part_data[0])['location_name'])
location_region = ['Global', 'High SDI', 'High-middle SDI', 'Low SDI', 'Low-middle SDI', 'Middle SDI',
                   "African Region", "Eastern Mediterranean Region", "European Region", "Region of the Americas", "South-East Asia Region", "Western Pacific Region",
                   "Andean Latin America",
                   "Australasia",
                   "Caribbean",
                   "Central Asia",
                   "Central Europe",
                   "Central Latin America",
                   "Central Sub-Saharan Africa",
                   "East Asia",
                   "Eastern Europe",
                   "Eastern Sub-Saharan Africa",
                   "High SDI",
                   "High-income Asia Pacific",
                   "High-income North America",
                   "Low SDI",
                   "North Africa and Middle East",
                   "Oceania",
                   "South Asia",
                   "Southeast Asia",
                   "Southern Latin America",
                   "Southern Sub-Saharan Africa",
                   "Tropical Latin America",
                   "Western Europe",
                   "Western Sub-Saharan Africa"]

# Age Standardization
mapping = {age_list_small[i]: age_list2[i] for i in range(len(age_list2))}
for i in range(1, 4):
    mapping[str(5*i)+' to '+str(5*i+4)] = str(5*i)+' to '+str(5*i+4)
mapping['1 to 4'] = '1 to 4'

# Read the three components of the SDI value and predict the future values of each component separately
A = [pd.read_csv(r'..\prevelance data\1.csv'),
     pd.read_csv(r'..\prevelance data\2.csv'),
     pd.read_csv(r'..\prevelance data\3.csv')]
P = 0


def dopredictionSDI(df):
    #predict SDI values and adjust with exp multiplier
    K = df.columns[-1]
    years = [[i] for i in np.arange(1980, 2020)]
    future_years = np.arange(2019, 2052).reshape(-1, 1)
    for index, row in df.iterrows():
        y = np.log(row.values)
        model = LinearRegression()
        model.fit(years, y)
        future_log_pred = model.predict(future_years)
        multiplier = row[K] / np.exp(future_log_pred[0])
        future_pred = np.exp(future_log_pred) * multiplier
        for year, value in zip(future_years, future_pred):
            df.loc[index, year[0]] = value
    return


def dopredictionSDIl(df):
    #predict SDI values and adjust with multiplier
    K = df.columns[-1]
    years = [[i] for i in np.arange(1980, 2020)]
    future_years = np.arange(2019, 2052).reshape(-1, 1)
    for index, row in df.iterrows():
        y = row.values
        model = LinearRegression()
        years = [[i] for i in np.arange(1980, 2020)]
        model.fit(years, y)
        future_log_pred = model.predict(future_years)
        multiplier = row[K] / future_log_pred[0]
        future_pred = future_log_pred * multiplier
        for year, value in zip(future_years, future_pred):
            df.loc[index, year[0]] = value
    return

# Read the first component of the SDI data
C = A[0]
D = C.pivot_table(index='location_name', columns='year_id', values='val')
D = D[[i for i in range(1980, 2020)]]
# Predict the first component of the SDI
dopredictionSDI(D)
# Normalize the data
maxn, minn = 3, 0
D[D > maxn] = maxn
D[D < minn] = minn
D_norm = D.apply(lambda x: 1-(x-minn)/(maxn-minn))
P = D_norm

# Read the second component of the SDI data
C = A[1]
D = C.pivot_table(index='location_name', columns='year_id', values='val')
D = D[[i for i in range(1980, 2020)]]
# Predict the second component of the SDI
dopredictionSDIl(D)
# Normalize the data
maxn, minn = 60000, 250
D[D > maxn] = maxn
D[D < minn] = minn
maxn, minn = 11, 5.5
D_norm = D.apply(lambda x: (x-minn)/(maxn-minn))
P *= D_norm

# Read the third component of the SDI data
C = A[2]
D = C.pivot_table(index='location_name', columns='year_id', values='val')
D = D[[i for i in range(1980, 2020)]]
# Predict the third component of the SDI
dopredictionSDIl(D)
# Normalize the data
maxn, minn = 17, 0
D[D > maxn] = maxn
D[D < minn] = minn
D_norm = D.apply(lambda x: (x-minn)/(maxn-minn))
P *= D_norm
SDI_value = pow(P, 0.333333)

# Classify SDI levels for different countries
SDIarea = ['High SDI', 'High-middle SDI',
           'Middle SDI', 'Low-middle SDI', 'Low SDI']
for p, area in enumerate(SDIarea):
    SDI_value.loc[area] = [np.nanpercentile(
        SDI_value[c], 90-20*p) for c in SDI_value.columns]

# Sort out the GBD regions and SDI levels for different countries
GBDR = [
    "Andean Latin America",
    "Australasia",
    "Caribbean",
    "Central Asia",
    "Central Europe",
    "Central Latin America",
    "Central Sub-Saharan Africa",
    "East Asia",
    "Eastern Europe",
    "Eastern Sub-Saharan Africa",
    "High-income Asia Pacific",
    "High-income North America",
    "North Africa and Middle East",
    "Oceania",
    "South Asia",
    "Southeast Asia",
    "Southern Latin America",
    "Southern Sub-Saharan Africa",
    "Tropical Latin America",
    "Western Europe",
    "Western Sub-Saharan Africa"]
years = np.arange(1990, 2022).reshape(-1, 1)
future_year = np.arange(2021, 2052)
future_years = future_year.reshape(-1, 1)
Dictgen = pd.read_excel('..\\prevelance data\\BB.xlsx')[
    ['Country', 'SDI_Cate', 'WHO region', 'Location']]
DSDI = {Dictgen['Country'][i]: Dictgen['SDI_Cate'][i] for i in range(204)}
DWHO = {Dictgen['Country'][i]: Dictgen['WHO region'][i] for i in range(204)}
Dictgen1 = pd.read_excel('..\\prevelance data\\GBD Regions.xlsx')
Dictgen1['Location-GBD'] = [replacement_mapping.get(i, i)
                            for i in Dictgen1['Location-GBD']]
DGBD = {Dictgen1['Location-GBD'][i]: Dictgen1['GBD Region'][i]
        for i in range(204)}
def dict2func(x): return lambda y: x.get(y, 'nan')


# if need to use logit, uncomment the following codes which transfer log to logit.
# from scipy.special import logit,expit
# np.log = lambda x:logit(x/100000)
# np.exp=lambda x:100000*expit(x)

def getpopulation(sex, value='val'):
    #Read population of countries from file and adjust name of Turkey to Türkiye
    population_df = []
    for csv_path in part_data[:-1]:
        df = pd.read_csv(csv_path)
        df = df[df['age_group_name'].isin(age_list)]
        df['location_name'][df['location_name'] == 'Turkey'] = 'Türkiye'
        df = df[df['location_name'].isin(locations)]
        df = df[df['sex_name'] == sex]
        pivot_df = df.pivot_table(
            index=['location_name', 'age_group_name'], columns='year_id', values=value)
        population_df.append(pivot_df)
    for csv_path in part_data[-1:]:
        df = pd.read_csv(csv_path)
        df = df[df['age_group_name'].isin(age_list)]
        df['location_name'][df['location_name'] == 'Turkey'] = 'Türkiye'
        df = df[df['location_name'].isin(locations)]
        df = df[df['year_id'] >= 2020]
        df = df[df['year_id'] <= 2050]
        if sex == 'both' and sex not in df['sex_name']:
            pivot_df = df.pivot_table(
                index=['location_name', 'age_group_name'], columns='year_id', values='val', aggfunc=sum)
        else:
            df = df[df['sex_name'] == sex]
            pivot_df = df.pivot_table(
                index=['location_name', 'age_group_name'], columns='year_id', values='val')
        population_df.append(pivot_df)
    population_df = pd.concat(population_df, axis=1)
    return population_df


def transage(i):
    # Change 1-5 years old to sequence 0, 5-10 years old to sequence 1, etc.
    m = int(i)//5
    return age_list[m]


def getprevelance(sex, value='val'):
    #Read prevelance of countries from file 
    csv_path = '..\\prevelance data\\CountryPrevelanceData2.csv'
    df = pd.read_csv(csv_path)
    df = df[df['age_name'].isin(age_list2)]
    df = df[df['location_name'].isin(locations)]
    df = df[df['sex_name'] == sex]
    df = df[df['metric_id'] == 3]
    df = df[df['year'] < 2022]
    if 'age_group_name' not in df.columns:
        df['age_group_name'] = df['age_name']
        df['year_id'] = df['year']
    pivot_df = df.pivot_table(
        index=['location_name', 'age_group_name'], columns='year_id', values=value)
    return pivot_df


def myfloat(s):
    # duel with the fault of uncorrect .
    s1 = s.replace('·', '.')
    return float(s1)


def predict_future(row, order=(0, 1, 0), steps=len(future_year)):
    #use ARIMA to predict future values
    row2 = row
    model = ARIMA(row2, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def dopredictionSDI_countryage(dfl, years, future_years):
    #model 1 in the appendix
    df = dfl[years].copy()
    K = df.columns[-1]
    for index, row in df.iterrows():
        y = np.log(row.values)
        model = LinearRegression()
        if 'Türkiye' == index[0]:
            yx = SDI_value.loc['Turkey', years].values
            yxf = SDI_value.loc['Turkey', future_years].values
        else:
            yx = SDI_value.loc[index[0], years].values
            yxf = SDI_value.loc[index[0], future_years].values
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


def dopredictionSDI_GLM(dfl, years, future_years):
    #model 2 in the appendix
    df = dfl[years].copy()
    K = df.columns[-1]
    D = {i: 0 for i in age_list2}
    counter = 0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yx = SDI_value.loc['Turkey', years].values
        else:
            yx = SDI_value.loc[index[0], years].values
        if np.sum(np.isnan(yx)) == 0 and np.sum(np.isnan(y)) == 0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[1]] += model.coef_
            counter += 1
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yxf = SDI_value.loc['Turkey', future_years].values
        else:
            yxf = SDI_value.loc[index[0], future_years].values
        if np.sum(np.isnan(yxf)) == 0 and np.sum(np.isnan(y)) == 0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[1]]/counter
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]/np.exp(future_log_pred[0])
            future_pred = multiplier*np.exp(future_log_pred)
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df


def dopredictionSDI_GLMARIMA(dfl, years, future_years):
    #model 3 in the appendix
    df = dfl[years].copy()
    K = df.columns[-1]
    D = {i: 0 for i in age_list2}
    counter = 0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yx = SDI_value.loc['Turkey', years].values
        else:
            yx = SDI_value.loc[index[0], years].values
        if np.sum(np.isnan(yx)) == 0 and np.sum(np.isnan(y)) == 0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[1]] += model.coef_
            counter += 1

    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yxf = SDI_value.loc['Turkey', future_years].values
        else:
            yxf = SDI_value.loc[index[0], future_years].values
        if np.sum(np.isnan(yxf)) == 0 and np.sum(np.isnan(y)) == 0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[1]]/counter
            future_log_pred = model.predict(yxf)
            residual = row.values[-1]-np.exp(future_log_pred[0])
            future_pred = np.exp(future_log_pred)+residual
            future_pred += row.values[-1] - future_pred[0]
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df


def dopredictionSDI_countryageARIMA(dfl, years, future_years):
    #model 4 in the appendix
    df = dfl[years].copy()
    K = df.columns[-1]
    for index, row in df.iterrows():
        y = np.log(row.values)
        model = LinearRegression()
        if 'Türkiye' == index[0]:
            yx = SDI_value.loc['Turkey', years].values
            yxf = SDI_value.loc['Turkey', future_years].values
        else:
            yx = SDI_value.loc[index[0], years].values
            yxf = SDI_value.loc[index[0], future_years].values
        if np.sum(np.isnan(yx)) == 0 and np.sum(np.isnan(y)) == 0:
            yx = [[i] for i in yx]
            yxf = [[i] for i in yxf]
            model.fit(yx, y)
            past_log_pred = model.predict(yx)
            residual = row.values-np.exp(past_log_pred)
            future_residual = predict_future(residual, steps=len(future_years))
            future_log_pred = model.predict(yxf)
            future_pred = np.exp(future_log_pred)+future_residual
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df


def dopredictionSDI_GLM2(dfl, years, future_years):
    #Model 5 in the appendix
    df = dfl[years].copy()
    K = df.columns[-1]
    D = {i: 0 for i in locations}
    counter = 0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yx = SDI_value.loc['Turkey', years].values
        else:
            yx = SDI_value.loc[index[0], years].values
        if np.sum(np.isnan(yx)) == 0 and np.sum(np.isnan(y)) == 0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[0]] += model.coef_
            counter += 1
    counter /= len(locations)
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yxf = SDI_value.loc['Turkey', future_years].values
        else:
            yxf = SDI_value.loc[index[0], future_years].values
        if np.sum(np.isnan(yxf)) == 0 and np.sum(np.isnan(y)) == 0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[0]]/counter
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]/np.exp(future_log_pred[0])
            future_pred = multiplier*np.exp(future_log_pred)
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df


def dopredictionSDI_GLM2ARIMA(dfl, years, future_years):
    #Model 6 in the appendix
    df = dfl[years].copy()
    K = df.columns[-1]
    D = {i: 0 for i in locations}
    counter = 0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yx = SDI_value.loc['Turkey', years].values
        else:
            yx = SDI_value.loc[index[0], years].values
        if np.sum(np.isnan(yx)) == 0 and np.sum(np.isnan(y)) == 0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[0]] += model.coef_
            counter += 1
    counter /= len(locations)
    counter *= 1.05
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye' == index[0]:
            yxf = SDI_value.loc['Turkey', future_years].values
        else:
            yxf = SDI_value.loc[index[0], future_years].values
        if np.sum(np.isnan(yxf)) == 0 and np.sum(np.isnan(y)) == 0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[0]]/counter
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]-np.exp(future_log_pred[0])
            future_pred = multiplier+np.exp(future_log_pred)
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df


def map_age1(age):
    # Group every 5-year age range into youth, middle-aged, and elderly categories
    age_start = int(age[:2])
    if age_start < 40:
        return '0-39'
    elif age_start < 60:
        return '40-59'
    elif age_start < 80:
        return '60-79'
    else:
        return '80+'


def map_age2(age): 
    # Group every 5-year age range into youth and elderly categories
    age_start = int(age[:2])
    if age_start < 60:
        return '0-60'
    else:
        return '60+'


import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python a.py Sex sex.")
        sys.exit(1)

    # Load determined prevelance and population
    x1 = sys.argv[1]  
    x2 = sys.argv[2]  
    prevelance_df=getprevelance(x1)
    population_df=getpopulation(x2)


    # prediction for 2011-2020 for 6 models
    df=population_df.copy()
    df.index=df.index.set_levels(df.index.levels[1].map(mapping.get), level=1)
    values_to_remove=age_list[:4]
    df = df.loc[~df.index.get_level_values(1).isin(values_to_remove)]
    
    # calculate result of 6 models
    years=np.arange(1990,2012)
    future_years=np.arange(2011,2022)
    prevelance_df1=dopredictionSDI_countryage(prevelance_df,years,future_years)
    prevelance_df2=dopredictionSDI_countryageARIMA(prevelance_df,years,future_years)
    prevelance_df3=dopredictionSDI_GLM(prevelance_df,years,future_years)
    prevelance_df4=dopredictionSDI_GLMARIMA(prevelance_df,years,future_years)
    prevelance_df5=dopredictionSDI_GLM2(prevelance_df,years,future_years)
    prevelance_df6=dopredictionSDI_GLM2ARIMA(prevelance_df,years,future_years)
    
    df=population_df.copy()
    df.index=df.index.set_levels(df.index.levels[1].map(mapping.get), level=1)

    values_to_remove=age_list[:4]
    df = df.loc[~df.index.get_level_values(1).isin(values_to_remove)]
    
    # Concat the predicted results
    dfs = [prevelance_df1, prevelance_df2, prevelance_df3, prevelance_df4, prevelance_df5, prevelance_df6,prevelance_df]

    # Find data with nan
    nan_indices = set()
    for df1 in dfs:
        nan_indices.update(df1[df1.isna().any(axis=1)].index)

    # drop all the nan data
    dfs_clean = [df1.drop(index=nan_indices) for df1 in dfs]

    # refresh the dataframes
    prevelance_df1, prevelance_df2, prevelance_df3, prevelance_df4, prevelance_df5, prevelance_df6,prevelance_df = dfs_clean
    # calculate bma coefficients
    Coeffs = []
    coeff_num= [14,14,19,19,20,20]
    for i in locations:
        coeff=[0 for j in range(6)]
        gt= prevelance_df.loc[(i,slice(None))]
        gt=gt[[j for j in range(2011,2022)]].values
        gt=gt.ravel()
        pre = []
        for j,df_pre in enumerate(dfs[:-1]):
            pred = df_pre.loc[(i,slice(None))]
            pred=pred[[j for j in range(2011,2022)]].values
            pred=pred.ravel()
            current_res = np.abs(rss(gt,pred))
            BIC = coeff_num[j]+15*np.log(current_res/15)
            coeff[j] = np.exp(-0.5*BIC)
        coeff=np.array(coeff)
        coeff/=np.sum(coeff)
        Coeffs.append(list(coeff))
    c=np.array(Coeffs)
    c=np.repeat(c,16,axis=0)
    prevelance_df7 = 0*prevelance_df1
    for i,df in enumerate(dfs[:-1]):
        prevelance_df7+=df*c[:,i].reshape(-1,1)
    
    # save the constant result
    # np.save(c,'c.npy')
    
    # list for store rmse and bias
    A=[]
    rmse_data=[]
    bias_data=[]
    # calculate rmse and bias
    for y in range(2011,2022):
        # Am=[]
        bm=prevelance_df[y].ravel()
        # rmse for every model
        # for i in dfs[:-1]:
        #     Am.append(rmse(i[y].ravel(),bm))
        # A.append(Am)
        rmse_data.append(rmse(prevelance_df7[y].ravel(),bm)/100000)
        bias_data.append(bias(prevelance_df7[y].ravel(),bm)/100000)
        print(f'{y} years bias value between bma model and real value is {bias_data[-1]}, rmse value between bma model and real value is {rmse_data[-1]}')
        
    # save the validation result
    df=pd.DataFrame([rmse_data,bias_data])
    df.index=['rmse','bias']
    df.columns=[i for i in range(2011,2022)]
    df.to_csv('validation_result.csv')