# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:01:55 2020

@author: Pankaj Kumar Sah
@LinkedIn: https://www.linkedin.com/in/pankaj-sah-b7aa39186/
@Github: https://github.com/52punk

"""
import pandas as pd
import numpy as np

df1=pd.read_csv("Bengaluru_House_Data.csv")
df2=df1.dropna()
df3=df2.drop(["area_type","society","availability","balcony"],axis="columns")
df3["bhk"]=df3["size"].apply(lambda x: int(x.split(' ')[0]))

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4=df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4["price_per_sqft"]=df4["price"]*100000/df4["total_sqft"]
df5=df4.copy()
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats_less_than_10=location_stats[location_stats<10]
df5['location']=df5['location'].apply(lambda x: 'others' if x in location_stats_less_than_10 else x)
df6=df5[~(df5.total_sqft/df5.bhk<300)]
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Hebbal")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
    
df8 = remove_bhk_outliers(df7)
df8.shape

plot_scatter_chart(df8,"Hebbal")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,10)
plt.hist(df8.price_per_sqft,rwidth = 0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.hist(df8.bath,rwidth = 0.8)
plt.xlabel("Number of bathroom")
plt.ylabel("Count")
df9 = df8[~(df8.bath>df8.bhk+2)]
df9.shape
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(10)
dummies = pd.get_dummies(df10.location)
dummies.head(3)
df11 = pd.concat([df10,dummies.drop('others',axis='columns')],axis='columns')
df11.head(3)
df12 = df11.drop('location', axis='columns')
df12.head(2)
X = df12.drop('price', axis='columns')
X.head()
Y = df12.price
Y.head()



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
regressor.score(X_test,Y_test)

def predict_values(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return regressor.predict([x])[0]

predict_values('1st Phase JP Nagar', 1000, 2, 2)
import pickle
with open('Bengaluru_house_model.picle', 'wb') as f:
    pickle.dump(regressor,f)
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


X_save=X.to_csv("X_save.csv")
X.shape