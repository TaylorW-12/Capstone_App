import pandas as pd 

data1=pd.read_csv('air_yards_data.csv')
data2=pd.read_csv('lead_changes.csv')
data3=pd.read_csv('receiving_data.csv')


print(data1.columns)
print(data2.columns)
print(data3.columns)