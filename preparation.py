import pandas as pd
import numpy as np
import os

# Loading dataset
df = pd.read_csv('data_raw.csv') # data from CRSP
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
# Change date format to int
df['yyyymm'] = df['year'].astype(int) * 100 + df['month'].astype(int)

# Leave only shrcd in (10,11)
df = df[(df['SHRCD']==10) | (df['SHRCD']==11)]

# permno
df.rename(columns={'PERMNO': 'permno'}, inplace=True)
df = df.drop_duplicates(subset=['yyyymm', 'permno']) # drop duplicates

factor_list = []
n = 0

# List all files in the folder
file_list = os.listdir('Predictors')

# Iterate through each file and read it into a DataFrame
for file in file_list:
    # Check if the file is a CSV file (you can modify this condition based on your file types)
    if file.endswith('.csv'):
        # Remove the file extension to use as the variable name
        variable_name = file.split('.')[0]
        print(variable_name)
        factor_list.append(variable_name)

        # Read the CSV file and store it in the dictionary
        factor = pd.read_csv(os.path.join(
            'Predictors', file))

        df = pd.merge(df, factor, on=['yyyymm', 'permno'], how='inner')

        print("Number of observations:", len(df))
        n += 1

breakpoints = pd.read_csv('ME_Breakpoints.csv')
breakpoints = breakpoints[['yyyymm', '20']]

df = pd.merge(df, breakpoints, on=['yyyymm'], how='inner')

# Market Capitalization
df['mktcap'] = abs(df['PRC']) * df['SHROUT'] /1000

# Drop small stocks
df = df[df['mktcap'] >= df['20']]

# Drop stocks with price less than $1
df = df[np.abs(df['PRC']) >= 1]

# Lagged returns
df['RET'] = pd.to_numeric(df['RET'], errors='coerce') # Converting RET
df['RET_lag'] = df.groupby('permno')['RET'].shift(-1)
df = df.dropna(subset=['RET_lag'])

# Norm Return
df['RET_lag_mean'] = df.groupby(['yyyymm'])['RET_lag'].transform('mean')
df['RET_lag_std'] = df.groupby(['yyyymm'])['RET_lag'].transform('std')
df['RET_lag'] = (df['RET_lag'] - df['RET_lag_mean']) / df['RET_lag_std']
df = df.drop('RET_lag_mean', axis=1)
df = df.drop('RET_lag_std', axis=1)

# Norm Factors
for i in factor_list:
    df[f'{i}_mean'] = df.groupby(['yyyymm'])[i].transform('mean')
    df[f'{i}_std'] = df.groupby(['yyyymm'])[i].transform('std')
    df[f'{i}_norm'] = (df[i] - df[f'{i}_mean']) / df[f'{i}_std']
    df = df.drop(f'{i}', axis=1)
    df = df.drop(f'{i}_mean', axis=1)
    df = df.drop(f'{i}_std', axis=1)

df.to_csv('data_norm.csv', index=True)
