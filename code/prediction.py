import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from basefunc import getpopulation

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

# Get the data file 
part_data = glob.glob(r'..\population data\IHME_GBD_2019_POP_1990_2019\*') + \
    [r'..\population data\IHME_POP_2017_2100_POP_REFERENCE\IHME_POP_2017_2100_POP_REFERENCE_Y2020M05D01.CSV']
age_list=['1 to 4']+[str(5*i)+' to '+str(5*i+4) for i in range(1,19)]+['95 plus']
age_list_small = age_list[4:]
age_list2=[str(i)+'-'+str(i+4)+' years' for i in range(20,91,5)]+['95+ years']

# Load prevalence data of countries
PD_GBD = pd.read_csv('..\\prevelance data\\CountryPrevelanceData2.csv')

# Correct the names of specific countries/regions
locations=np.unique(PD_GBD['location_name'])
locations[np.where("C么te d'Ivoire"==locations )]="Côte d'Ivoire"
locations_old=locations.copy()
mapping={age_list_small[i]:age_list2[i] for i in range(len(age_list2))}
for i in range(1,4):
    mapping[str(5*i)+' to '+str(5*i+4)]=str(5*i)+' to '+str(5*i+4)
mapping['1 to 4']='1 to 4'
replacement_mapping = {"Micronesia (Fed. States of)":"Micronesia (Federated States of)",
        'State of Palestine':'Palestine', 
        "Dem. People's Republic of Korea":"Democratic People's Republic of Korea",
        "C么te d'Ivoire": "Côte d'Ivoire",
        'China, Taiwan Province of China': 'Taiwan (Province of China)',
        'T眉rkiye':'Turkey',
        'Türkiye':'Turkey',
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
        'World':'Global'
}
# for i in replacement_mapping.values():
#     if i not in locations:
#         print(i)
# for i in replacement_mapping.keys():
#     if i not in locations2 and replacement_mapping[i] not in locations2:
#         print(i)
location_all=np.unique(pd.read_csv(part_data[0])['location_name'])
location_region=['Global','High SDI','High-middle SDI','Low SDI','Low-middle SDI','Middle SDI',
    "African Region","Eastern Mediterranean Region","European Region","Region of the Americas","South-East Asia Region","Western Pacific Region",
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


#predict SDI values
A = [pd.read_csv(r'..\prevelance data\1.csv'),
     pd.read_csv(r'..\prevelance data\2.csv'),
     pd.read_csv(r'..\prevelance data\3.csv')]
P=0
def dopredictionSDI(df):
    K=df.columns[-1]
    years = [[i] for i in np.arange(1980,2020)]
    future_years = np.arange(2019, 2051).reshape(-1, 1)
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
    K=df.columns[-1]
    years = [[i] for i in np.arange(1980,2020)]
    future_years = np.arange(2019, 2051).reshape(-1, 1)
    for index, row in df.iterrows():
        y = row.values
        model = LinearRegression()
        years = [[i] for i in np.arange(1980,2020)]
        model.fit(years, y)
        future_log_pred = model.predict(future_years)
        multiplier = row[K] / future_log_pred[0]
        future_pred = future_log_pred * multiplier
        for year, value in zip(future_years, future_pred):
            df.loc[index, year[0]] = value
    return

# Read and predict SDI data
C=A[0]
D=C.pivot_table(index='location_name',columns='year_id',values='val')
D=D[[i for i in range(1980,2020)]]
dopredictionSDI(D)
maxn,minn=3,0
D[D>maxn]=maxn
D[D<minn]=minn
D_norm = D.apply(lambda x:1-(x-minn)/(maxn-minn))
P=D_norm
C=A[1]
D=C.pivot_table(index='location_name',columns='year_id',values='val')
D=D[[i for i in range(1980,2020)]]
dopredictionSDIl(D)
maxn,minn=60000,250
D[D>maxn]=maxn
D[D<minn]=minn
D=np.log(D)
maxn,minn=11,5.5
D_norm = D.apply(lambda x:(x-minn)/(maxn-minn))
P*=D_norm
C=A[2]
D=C.pivot_table(index='location_name',columns='year_id',values='val')
D=D[[i for i in range(1980,2020)]]
dopredictionSDIl(D)
maxn,minn=17,0
D[D>maxn]=maxn
D[D<minn]=minn
D_norm = D.apply(lambda x:(x-minn)/(maxn-minn))
P*=D_norm
SDI_value=pow(P,0.333333)
SDIarea=['High SDI','High-middle SDI','Middle SDI','Low-middle SDI','Low SDI']
# Classify SDI levels for different countries
for p,area in enumerate(SDIarea):
    SDI_value.loc[area]=[np.nanpercentile(SDI_value[c],90-20*p) for c in SDI_value.columns]
    
# Sort out the GBD regions and SDI levels for different countries
GBDR=[
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
future_year = np.arange(2021, 2051)
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
dict2func = lambda x:lambda y:x.get(y,'nan')




def getpopulation(sex,value='val'):
    #Read population of countries from file and adjust name of Turkey to Türkiye
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
        df=df[df['year_id']<=2050]
        if sex == 'both' and sex not in df['sex_name']:
            pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values='val',aggfunc=sum)
        else:
            df=df[df['sex_name']==sex]
            pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values='val')
        population_df.append(pivot_df)
    population_df=pd.concat(population_df,axis=1)
    return population_df

def transage(i):
    # Change 1-5 years old to sequence 0, 5-10 years old to sequence 1, etc.
    m=int(i)//5
    return age_list[m]
    
def getprevelance(sex,value='val'):
    #Read prevelance of countries from file 
    csv_path='..\\prevelance data\\CountryPrevelanceData2.csv'
    df=pd.read_csv(csv_path)
    df=df[df['age_name'].isin(age_list2)]
    df=df[df['location_name'].isin(locations)]
    df=df[df['sex_name']==sex]
    df=df[df['metric_id']==3]
    df=df[df['year']<2022]
    if 'age_group_name' not in df.columns:
        df['age_group_name']=df['age_name']
        df['year_id']=df['year']
    pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values=value)
    return pivot_df

def myfloat(s):
    # duel with the fault of uncorrect .
    s1=s.replace('·','.')
    return float(s1)

from statsmodels.tsa.arima.model import ARIMA
def predict_future(row,order=(0,1,0),steps=len(future_year)):
    #use ARIMA to predict future values
    row2 = row
    model = ARIMA(row2,order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def dopredictionSDI_countryage(dfl,years,future_years):
    #model 1 in the appendix
    df=dfl[years].copy()
    K = df.columns[-1]
    for index, row in df.iterrows():
        y = np.log(row.values)
        model = LinearRegression()
        if 'Türkiye'==index[0]:
            yx = SDI_value.loc['Turkey',years].values
            yxf = SDI_value.loc['Turkey',future_years].values
        else:
            yx = SDI_value.loc[index[0],years].values
            yxf = SDI_value.loc[index[0],future_years].values
        if np.sum(np.isnan(yx))==0 and np.sum(np.isnan(y))==0:
            yx = [[i] for i in yx]
            yxf = [[i] for i in yxf]
            model.fit(yx, y)
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]/np.exp(future_log_pred[0])
            future_pred = np.exp(future_log_pred)*multiplier
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df

def dopredictionSDI_countryageARIMA(dfl,years,future_years):
    #model 4 in the appendix
    df=dfl[years].copy()
    K = df.columns[-1]
    for index, row in df.iterrows():
        y = np.log(row.values)
        model = LinearRegression()
        if 'Türkiye'==index[0]:
            yx = SDI_value.loc['Turkey',years].values
            yxf = SDI_value.loc['Turkey',future_years].values
        else:
            yx = SDI_value.loc[index[0],years].values
            yxf = SDI_value.loc[index[0],future_years].values
        if np.sum(np.isnan(yx))==0 and np.sum(np.isnan(y))==0:
            yx = [[i] for i in yx]
            yxf = [[i] for i in yxf]
            model.fit(yx, y)
            past_log_pred = model.predict(yx)
            residual = row.values-np.exp(past_log_pred)
            future_residual = predict_future(residual,steps=len(future_years))
            future_log_pred = model.predict(yxf)
            future_pred = np.exp(future_log_pred)+future_residual
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df

def dopredictionSDI_GLM(dfl,years,future_years):
    #model 2 in the appendix
    df=dfl[years].copy()
    K=df.columns[-1]
    D = {i:0 for i in age_list2}
    counter=0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yx = SDI_value.loc['Turkey',years].values
        else:
            yx = SDI_value.loc[index[0],years].values
        if np.sum(np.isnan(yx))==0 and np.sum(np.isnan(y))==0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[1]]+=model.coef_
            counter+=1
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yxf = SDI_value.loc['Turkey',future_years].values
        else:
            yxf = SDI_value.loc[index[0],future_years].values
        if np.sum(np.isnan(yxf))==0 and np.sum(np.isnan(y))==0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[1]]/counter
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]/np.exp(future_log_pred[0])
            future_pred = multiplier*np.exp(future_log_pred)
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df

def dopredictionSDI_GLMARIMA(dfl,years,future_years):
    #model 3 in the appendix
    df=dfl[years].copy()
    K=df.columns[-1]
    D = {i:0 for i in age_list2}
    counter=0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yx = SDI_value.loc['Turkey',years].values
        else:
            yx = SDI_value.loc[index[0],years].values
        if np.sum(np.isnan(yx))==0 and np.sum(np.isnan(y))==0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[1]]+=model.coef_
            counter+=1
    
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yxf = SDI_value.loc['Turkey',future_years].values
        else:
            yxf = SDI_value.loc[index[0],future_years].values
        if np.sum(np.isnan(yxf))==0 and np.sum(np.isnan(y))==0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[1]]/counter
            future_log_pred = model.predict(yxf)
            residual = row.values[-1]-np.exp(future_log_pred[0])
            future_pred = np.exp(future_log_pred)+residual
            future_pred +=row.values[-1] -future_pred[0]
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df

def dopredictionSDI_GLM2(dfl,years,future_years):
    #model 5 in the appendix
    df=dfl[years].copy()
    K=df.columns[-1]
    D = {i:0 for i in locations}
    counter=0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yx = SDI_value.loc['Turkey',years].values
        else:
            yx = SDI_value.loc[index[0],years].values
        if np.sum(np.isnan(yx))==0 and np.sum(np.isnan(y))==0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[0]]+=model.coef_
            counter+=1
    counter/=len(locations)
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yxf = SDI_value.loc['Turkey',future_years].values
        else:
            yxf = SDI_value.loc[index[0],future_years].values
        if np.sum(np.isnan(yxf))==0 and np.sum(np.isnan(y))==0:
            yxf = [[i] for i in yxf]
            model.coef_ = D[index[0]]/counter
            future_log_pred = model.predict(yxf)
            multiplier = row.values[-1]/np.exp(future_log_pred[0])
            future_pred = multiplier*np.exp(future_log_pred)
            for year, value in zip(future_years, future_pred):
                df.loc[index, year] = value
    return df

def dopredictionSDI_GLM2ARIMA(dfl,years,future_years):
    #model 6 in the appendix
    df=dfl[years].copy()
    K=df.columns[-1]
    D = {i:0 for i in locations}
    counter=0
    model = LinearRegression()
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yx = SDI_value.loc['Turkey',years].values
        else:
            yx = SDI_value.loc[index[0],years].values
        if np.sum(np.isnan(yx))==0 and np.sum(np.isnan(y))==0:
            yx = [[i] for i in yx]
            model.fit(yx, y)
            D[index[0]]+=model.coef_
            counter+=1
    counter/=len(locations)
    counter*=1.05
    for index, row in df.iterrows():
        y = np.log(row.values)
        if 'Türkiye'==index[0]:
            yxf = SDI_value.loc['Turkey',future_years].values
        else:
            yxf = SDI_value.loc[index[0],future_years].values
        if np.sum(np.isnan(yxf))==0 and np.sum(np.isnan(y))==0:
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
    
c = np.load('..\\prevelance data\\c.npy')
def doprediction(model,prevelance_df,years,future_years):
    # the bma model for prediction
    prevelance_df1=dopredictionSDI_countryage(prevelance_df,years,future_years)
    prevelance_df2=dopredictionSDI_countryageARIMA(prevelance_df,years,future_years)
    prevelance_df3=dopredictionSDI_GLM(prevelance_df,years,future_years)
    prevelance_df4=dopredictionSDI_GLMARIMA(prevelance_df,years,future_years)
    prevelance_df5=dopredictionSDI_GLM2(prevelance_df,years,future_years)
    prevelance_df6=dopredictionSDI_GLM2ARIMA(prevelance_df,years,future_years)
    dfs = [prevelance_df1,prevelance_df2,prevelance_df3,prevelance_df4,prevelance_df5,prevelance_df6]
    prevelance_df7 = 0*prevelance_df1
    for i,df in enumerate(dfs[:-1]):
        df[df<0]=0
        prevelance_df7+=df*c[:,i].reshape(-1,1)
    prevelance_df7[years]=prevelance_df[years]
    return prevelance_df7

def bootstrap_prediction(model,years,future_years,df_lower, df_upper, n_iterations=1000):
    # Sample to calculate the 95% uncertainty interval
    # UIs may vary slightly across multiple runs.
    # This iteration cost a lot of time if n_iterations is large.
    predictions= []
    growth = []
    for _ in range(n_iterations):
        df = df_lower + (df_upper - df_lower) * np.random.rand(*df_lower.shape)
        df = doprediction(model,df,years,future_years)
        predictions.append(df.values)
        growth_percent = ((df.loc[:, 2050].values / df.loc[:, 2021].values) - 1) * 100
        growth.append(growth_percent)
    predictions = np.array(predictions)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    return lower_bound, upper_bound

model = 0

for (sex1,sex2) in [('both','Both'),('male','Male'),('female','Female')]:
    
    # Initial setup for prediction (forecast to 2050 using data between 1990 and 2021)
    years=np.arange(1990,2022)
    future_years=np.arange(2021,2051)
    sex=sex1
    
    # read the data of population
    population_df=getpopulation(sex)

    # Since there is no SDI population in the GBD prediction, manually set the SDI population based on the population of countries belonging to different SDI levels
    df_reset =population_df.reset_index()
    df_reset=df_reset[df_reset['location_name'].isin(Dictgen['Country'])]
    df_reset['location_name'] = df_reset['location_name'].apply(dict2func(DSDI))
    SDI_population = df_reset.groupby(['location_name', 'age_group_name']).sum()

    sub_df=SDI_population
    population_df.loc[sub_df.index, sub_df.columns] = sub_df

    df_reset =population_df.reset_index()
    df_reset['age_group'] = df_reset['age_group_name'].apply(map_age1)
    total_population = df_reset.groupby(['location_name', 'age_group']).sum()

    age_list2=[str(i)+'-'+str(i+4)+' years' for i in range(20,91,5)]+['95+ years']
    sex=sex2
    prevelance_df = getprevelance(sex)
    years=np.arange(1990,2022)
    future_years=np.arange(2021,2051)

    prevelance_dfp = doprediction(model,prevelance_df,years,future_years)

    df=population_df.copy()
    df.index=df.index.set_levels(df.index.levels[1].map(mapping.get), level=1)

    values_to_remove=age_list[:4]
    df = df.loc[~df.index.get_level_values(1).isin(values_to_remove)]

    pre_numbers=df*prevelance_dfp/100000
    df_reset=pre_numbers.reset_index()
    dict2func = lambda x:lambda y:x.get(y,'nan')
    SDIarea=['High SDI','High-middle SDI','Low SDI','Low-middle SDI','Middle SDI']
    df_reset['location_name'] = df_reset['location_name'].apply(dict2func(DSDI))
    df_reset=df_reset[df_reset['location_name'].isin(SDIarea)]
    SDI_pre_numbers = df_reset.groupby(['location_name', 'age_group_name']).sum()
    df_reset2=df.reset_index()
    df_reset2=df_reset2[df_reset2['location_name'].isin(SDIarea)]
    SDI_population=df_reset2.groupby(['location_name', 'age_group_name']).sum()
    sub_df=SDI_pre_numbers/SDI_population*100000
    prevelance_dfp.loc[sub_df.index, sub_df.columns] = sub_df
    
    numbers_sum_1 = pre_numbers.groupby(level=0).sum()
    crude_prevelance = numbers_sum_1/population_df.groupby(level=0).sum()*100000
    
    df_reset =pre_numbers.reset_index()
    df_reset['age_group'] = df_reset['age_group_name'].apply(map_age1)
    
    pre_numbers.to_excel(sex+'_numbers_16_age_groups.xlsx')
    prevelance_dfp.to_excel(sex+'_prevelance_16_age_groups.xlsx')
    numbers_sum_1.to_excel(sex+'_numbers.xlsx')
    crude_prevelance.to_excel(sex+'_crude_prevelance.xlsx')
    
    numbers_sum_by_reset_groups = df_reset.groupby(['location_name', 'age_group']).sum()
    crude_prevelance_groups=numbers_sum_by_reset_groups/total_population*100000
    numbers_sum_by_reset_groups.to_excel(sex+'_numbers_4_age_groups.xlsx')
    crude_prevelance_groups.to_excel(sex+'_prevelance_4_age_groups.xlsx')
    prevelance_df=prevelance_dfp
    df_reset =pre_numbers.reset_index()
    df_reset['age_group'] = df_reset['age_group_name'].apply(map_age2)
    numbers_sum_by_reset_groups = df_reset.groupby(['location_name', 'age_group']).sum()
    df_reset =population_df.reset_index()
    df_reset['age_group'] = df_reset['age_group_name'].apply(map_age2)
    population_by_group2 = df_reset.groupby(['location_name', 'age_group']).sum()
    crude_prevelance_groups=numbers_sum_by_reset_groups/population_by_group2*100000
    numbers_sum_by_reset_groups.to_excel(sex+'_numbers_2_age_groups.xlsx')
    crude_prevelance_groups.to_excel(sex+'_prevelance_2_age_groups.xlsx')
    
    
    prevelance_df_up = getprevelance(sex,'upper')
    prevelance_df_lo = getprevelance(sex,'lower')
    years=np.arange(1990,2022)
    future_years=np.arange(2021,2051)
    prevelance_df_lo,prevelance_df_up=bootstrap_prediction(model,years,future_years, prevelance_df_lo, prevelance_df_up)
    prevelance_df_up=prevelance_df_up[:,:len(prevelance_df.columns)]
    prevelance_df_lo=prevelance_df_lo[:,:len(prevelance_df.columns)]
    mask=prevelance_df_lo>prevelance_df.values
    prevelance_df_lo[mask]=0.8*prevelance_df.values[mask]
    mask=prevelance_df_up<prevelance_df.values
    prevelance_df_up[mask]=1.2*prevelance_df.values[mask]

    prevelance_df = pd.DataFrame(prevelance_df_up, index=prevelance_df.index, columns=prevelance_df.columns)

    pre_numbers = df*prevelance_df/100000
    
    numbers_sum_1 = pre_numbers.groupby(level=0).sum()
    crude_prevelance = numbers_sum_1/population_df.groupby(level=0).sum()*100000

    df_reset =pre_numbers.reset_index()
    df_reset['age_group'] = df_reset['age_group_name'].apply(map_age1)

    numbers_sum_by_reset_groups = df_reset.groupby(['location_name', 'age_group']).sum()
    crude_prevelance_groups=numbers_sum_by_reset_groups/total_population*100000
    pre_numbers.to_excel(sex+'_numbers_16_age_groups_upper.xlsx')
    prevelance_df.to_excel(sex+'_prevalence_16_age_groups_upper.xlsx')
    numbers_sum_1.to_excel(sex+'_numbers_upper.xlsx')
    crude_prevelance.to_excel(sex+'_crude_prevalence_upper.xlsx')
    numbers_sum_by_reset_groups.to_excel(sex+'_numbers_4_age_groups_upper.xlsx')
    crude_prevelance_groups.to_excel(sex+'_prevalence_4_age_groups_upper.xlsx')
    if sex == 'Both':
        df_reset =pre_numbers.reset_index()
        df_reset['age_group'] = df_reset['age_group_name'].apply(map_age2)
        numbers_sum_by_reset_groups = df_reset.groupby(['location_name', 'age_group']).sum()
        df_reset =population_df.reset_index()
        df_reset['age_group'] = df_reset['age_group_name'].apply(map_age2)
        population_by_group2 = df_reset.groupby(['location_name', 'age_group']).sum()
        crude_prevelance_groups=numbers_sum_by_reset_groups/population_by_group2*100000
        numbers_sum_by_reset_groups.to_excel(sex+'_numbers_2_age_groups_upper.xlsx')
        crude_prevelance_groups.to_excel(sex+'_prevalence_2_age_groups_upper.xlsx')
    prevelance_df = pd.DataFrame(prevelance_df_lo, index=prevelance_df.index, columns=prevelance_df.columns)

    pre_numbers=df*prevelance_df/100000
    
    numbers_sum_1 = pre_numbers.groupby(level=0).sum()
    crude_prevelance = numbers_sum_1/population_df.groupby(level=0).sum()*100000

    df_reset =pre_numbers.reset_index()
    df_reset['age_group'] = df_reset['age_group_name'].apply(map_age1)

    numbers_sum_by_reset_groups = df_reset.groupby(['location_name', 'age_group']).sum()
    crude_prevelance_groups=numbers_sum_by_reset_groups/total_population*100000

    pre_numbers.to_excel(sex+'_numbers_16_age_groups_lower.xlsx')
    prevelance_df.to_excel(sex+'_prevalence_16_age_groups_lower.xlsx')
    numbers_sum_1.to_excel(sex+'_numbers_lower.xlsx')
    crude_prevelance.to_excel(sex+'_crude_prevalence_lower.xlsx')
    numbers_sum_by_reset_groups.to_excel(sex+'_numbers_4_age_groups_lower.xlsx')
    crude_prevelance_groups.to_excel(sex+'_prevalence_4_age_groups_lower.xlsx')

    if sex == 'Both':
        df_reset =pre_numbers.reset_index()
        df_reset['age_group'] = df_reset['age_group_name'].apply(map_age2)
        numbers_sum_by_reset_groups = df_reset.groupby(['location_name', 'age_group']).sum()
        df_reset =population_df.reset_index()
        df_reset['age_group'] = df_reset['age_group_name'].apply(map_age2)
        population_by_group2 = df_reset.groupby(['location_name', 'age_group']).sum()
        crude_prevelance_groups=numbers_sum_by_reset_groups/population_by_group2*100000
        numbers_sum_by_reset_groups.to_excel(sex+'_numbers_2_age_groups_lower.xlsx')
        crude_prevelance_groups.to_excel(sex+'_prevalence_2_age_groups_lower.xlsx')
    print(f"{sex} data processing completed, results saved.")
    



def getprevelance(sex,value='val'):
    # Get the standardized prevalence rate
    csv_path=r'..\\prevelance data\\standardrate.csv'
    df=pd.read_csv(csv_path)
    df=df[df['age_name'].isin(['Age-standardized'])]
    df=df[df['sex_name']==sex]
    df=df[df['metric_id']==3]
    df=df[df['year']<2022]
    df["location_name"]=[{"T眉rkiye":"Turkey"}.get(i,i) for i in df["location_name"].values]
    if 'age_group_name' not in df.columns:
        df['age_group_name']=df['age_name']
        df['year_id']=df['year']
    pivot_df=df.pivot_table(index=['location_name','age_group_name'], columns='year_id', values=value)
    return pivot_df

from basefunc import bootstrap_prediction2

def process_sex_data(sex, getprevelance, bootstrap_prediction2,a=[i for i in range(1990,2051)]):
    """
    Prcoess data of age-standard rate for determined sex
    """
    # get bounds of prevelance data
    prevelance_df = getprevelance(sex)
    df_upper = getprevelance(sex, 'upper')
    df_lower = getprevelance(sex, 'lower')

    # do bootstrap prediction
    df_lower, df_upper, df_mid, lr, ur = bootstrap_prediction2(df_lower, df_upper)

    # create dataframe to solve the results of 95UI
    predr = pd.DataFrame(index=prevelance_df.index)
    predr['lower95'] = lr -1
    predr['upper95'] = ur -1
    predr.to_csv(f'{sex} sex_percentage change of age-standardized rate during 2021-2050_95UI.csv')

    # write data to dataframe
    df_lower = pd.DataFrame(df_lower, index=prevelance_df.index, columns=a)
    df_upper = pd.DataFrame(df_upper, index=prevelance_df.index, columns=a)
    df_mid = pd.DataFrame(df_mid, index=prevelance_df.index, columns=a)

    # save all the results
    df_mid.to_csv(f'{sex} sex age-standardized rate.csv')
    df_upper.to_csv(f'{sex} sex age-standardized rate_upper.csv')
    df_lower.to_csv(f'{sex} sex age-standardized rate_lower.csv')

    print(f"{sex} sex age-standardized rate ended")
    
sex_list = ['Both','Female', 'Male']

# prediction for all sex in sex_list
for sex in sex_list:
    process_sex_data(sex, getprevelance, bootstrap_prediction2)