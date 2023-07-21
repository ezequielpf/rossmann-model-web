import requests
import pandas as pd
import json

df_store_raw = pd.read_csv('../datasets/store.csv', low_memory=False)

# loading test dataset
df = pd.read_csv('/home/ezequiel/Documentos/Prejetos_Data_Science/DS_em_producao/datasets/test.csv')

# merge test dataset + store
df_test = pd.merge(df, df_store_raw, how='left', on='Store')

# choosse store for prediction
df_test = df_test[df_test['Store'] == 30]

# remove closed days
df_test = df_test[df_test['Open'] != 0]
df_test = df_test[~df_test['Open'].isnull()]
df_test = df_test.drop('Id', axis=1)

data = df_test.to_json(orient='records', date_format='iso')

# API call
url = 'http://localhost:5000/rossmann/predict'
#header = {'Content-type': 'application/jason'}
data = data

#r = requests.post(url, data=data, headers=header)
r = requests.post(url=url, json=data)
print(f'Status Code {r.status_code}')

d1 = pd.json_normalize(r.json()).head()

d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

for i in range(len(d2)):
    print(f'Store number {d2.loc[i,"store"]} will sell {d2.loc[i,"prediction"]} in the next 6 weeks')