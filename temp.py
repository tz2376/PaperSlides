# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np

# Example panel data with an additional universe DataFrame
data = {
    'stock': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
    'date': ['2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
             '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05'],
    'return': [0.01, 0.02, 0.015, -0.005, 0.01, 0.03, 0.025, -0.01, 0.005, 0.02],
    'weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Example universe data
universe_data = {
    'date': ['2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05'],
    'A': [1, 1, 1, 1, 1],
    'B': [1, 1, 1, 1, 1]
}
universe = pd.DataFrame(universe_data)
universe['date'] = pd.to_datetime(universe['date'])
universe.set_index('date', inplace=True)

# Merge df with universe to filter the stocks included at each time t
df = df.set_index(['date', 'stock']).join(universe.stack().reset_index().rename(columns={0: 'included'}).set_index(['date', 'level_1']), how='inner').reset_index()
df = df[df['included'] == 1]

def weighted_mean(df):
    weighted_sum = np.sum(df['return'] * df['weight'])
    total_weight = np.sum(df['weight'])
    return weighted_sum / total_weight

def weighted_covariance(df, lag=1):
    mean = weighted_mean(df)
    df['return_shifted'] = df.groupby('stock')['return'].shift(-lag)
    df['weight_shifted'] = df.groupby('stock')['weight'].shift(-lag)
    
    valid = df.dropna(subset=['return_shifted'])
    
    cov = np.sum(valid['weight'] * (valid['return'] - mean) * (valid['return_shifted'] - mean))
    weight_sum = np.sum(valid['weight'])
    
    return cov / weight_sum

def weighted_covariance_version2(df, lag=1):
    mean = weighted_mean(df)
    df['return_shifted'] = df.groupby('stock')['return'].shift(-lag)
    valid = df.dropna(subset=['return_shifted'])

    cov = np.sum(valid['weight'] * (valid['return'] - mean) * (valid['return_shifted'] - mean))
    weight_sum = np.sum(valid['weight'])
    
    return cov / weight_sum

def weighted_variance(df):
    mean = weighted_mean(df)
    var = np.sum(df['weight'] * (df['return'] - mean) ** 2)
    total_weight = np.sum(df['weight'])
    return var / total_weight

def weighted_acf(df, lag=1):
    weighted_cov = weighted_covariance(df, lag)
    weighted_var = weighted_variance(df)
    return weighted_cov / weighted_var

# Compute weighted autocorrelation for lag 1
weighted_acf_value = weighted_acf(df, lag=1)
print(f"Weighted Autocorrelation for lag 1: {weighted_acf_value}")



