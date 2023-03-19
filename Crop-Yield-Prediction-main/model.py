import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")


data=pd.read_csv("C:\\Users\\prakash\\data.csv")
data
#data=data.fillna(data.mean())
data.drop(['Unnamed: 0'],axis=1)
''' data.drop(data[(data['soil_type'] == 'peaty')].index, inplace=True)
data.drop(data[(data['soil_type']=='loamy')].index,inplace=True)
data.drop(data[(data['crop_names']=='Bajra')].index,inplace=True)
data.drop(data[(data['crop_names']=='Jowar')].index,inplace=True)
data.drop(data[(data['crop_names']=='Sunflower')].index,inplace=True)
data.drop(data[(data['crop_names']=='Wheat')].index,inplace=True)
data.drop(data[(data['district_names']=='AMRAVATI')].index,inplace=True)
def district_names(x):
    if x=='AHMEDNAGAR':
        return 1
    if x=='AURANGABAD':
        return 2
    if x=='NAGPUR':
        return 3
    if x=='NANDED':
        return 4
    if x=='PUNE':
        return 5
def season_names(x):
    if x=='Kharif     ':
        return 1
    if x=='Rabi       ':
        return 2
def crop_names(x):
    if x=='Maize':
        return 1
    if x=='Rice':
        return 2
    if x=='Cotton(lint)':
        return 3
    if x=='Soyabean':
        return 4
    if x=='Groundnut':
        return 5
def soil_type(x):
    if x=='clay':
        return 1
    if x=='sandy':
        return 2
    if x=='chalky':
        return 3
    if x=='silt':
        return 4
    if x=='silty':
        return 5
data['district_names']=data['district_names'].apply(district_names)
data['season_names']=data['season_names'].apply(season_names)
data['crop_names']=data['crop_names'].apply(crop_names)
data['soil_type']=data['soil_type'].apply(soil_type)
data.drop(data[(data['Yield.'])>3.0].index,inplace=True)'''
data["temperature"].astype("int64",errors='ignore')
data["humidity"].astype("int64",errors='ignore')
x=data[['district_names','season_names','crop_names','area','temperature','soil_type']]
y=data[['Yield.']].astype("int64",errors='ignore')
model= LinearRegression()
model.fit(x,y)
#inputt=[int(x) for x in x.split(' ')]
#final=[np.array(inputt)]
model.predict(x)
pickle.dump(model,open('crop.pkl','wb'))
model=pickle.load(open('crop.pkl','rb'))


