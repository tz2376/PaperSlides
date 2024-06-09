#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:49:12 2024

@author: tonyzhang
"""



######   unconditional universe  #####

import numpy as np
import pandas as pd

def acf_unconditional(df_x, df_w, df_u, lags):
    x = df_x.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()
    
    mask = ~np.isnan(x)
    w_masked = np.where(mask, w * u, np.nan)
    total_weight = np.nansum(w_masked)
    mean_x = np.nansum(w_masked * x) / total_weight
    
    dev = x - mean_x
    var = np.nansum(w_masked * (dev ** 2)) / total_weight
    
    acf_values = []

    for lag in lags:
        if lag > 0:
            x_shifted = np.roll(x, lag)
            x_shifted[:lag] = np.nan
        else:
            x_shifted = x

        valid_weights = w_masked * ~np.isnan(x_shifted)
        autcov = np.nansum(dev * (x_shifted - mean_x) * valid_weights)
        normalization = np.nansum(valid_weights)
        
        if normalization == 0:
            acf_values.append(float('nan'))
        else:
            autcov /= normalization
            acf_values.append(autcov / var)
    
    return acf_values

# Example usage
df_x = pd.DataFrame(np.random.rand(100))
df_w = pd.DataFrame(np.random.rand(100))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(100,), p=[0.5, 0.5]))
lags = [1, 2, 3]
print(acf_unconditional(df_x, df_w, df_u, lags))



# unconditional multiprocessing

import numpy as np
import pandas as pd
from multiprocessing import Pool

def compute_acf_unconditional(lag, x, w_masked, mean_x, dev, var):
    if lag > 0:
        x_shifted = np.roll(x, lag)
        x_shifted[:lag] = np.nan
    else:
        x_shifted = x

    valid_weights = w_masked * ~np.isnan(x_shifted)
    autcov = np.nansum(dev * (x_shifted - mean_x) * valid_weights)
    normalization = np.nansum(valid_weights)
    
    if normalization == 0:
        return float('nan')
    else:
        autcov /= normalization
        return autcov / var

def acf_unconditional_parallel(df_x, df_w, df_u, lags, num_workers):
    x = df_x.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()
    
    mask = ~np.isnan(x)
    w_masked = np.where(mask, w * u, np.nan)
    total_weight = np.nansum(w_masked)
    mean_x = np.nansum(w_masked * x) / total_weight
    
    dev = x - mean_x
    var = np.nansum(w_masked * (dev ** 2)) / total_weight
    
    with Pool(num_workers) as pool:
        acf_values = pool.starmap(compute_acf_unconditional, [(lag, x, w_masked, mean_x, dev, var) for lag in lags])

    return acf_values

# Example usage
df_x = pd.DataFrame(np.random.rand(100))
df_w = pd.DataFrame(np.random.rand(100))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(100,), p=[0.5, 0.5]))
lags = [1, 2, 3]
num_workers = 4
print(acf_unconditional_parallel(df_x, df_w, df_u, lags, num_workers))






#


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def compute_acf_unconditional(lag, x, w_masked, mean_x, dev, var):
    if lag > 0:
        x_shifted = np.roll(x, lag)
        x_shifted[:lag] = np.nan
    else:
        x_shifted = x

    valid_weights = w_masked * ~np.isnan(x_shifted)
    autcov = np.nansum(dev * (x_shifted - mean_x) * valid_weights)
    normalization = np.nansum(valid_weights)
    
    if normalization == 0:
        return float('nan')
    else:
        autcov /= normalization
        return autcov / var

def acf_unconditional_parallel_joblib(df_x, df_w, df_u, lags, num_workers):
    x = df_x.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()
    
    mask = ~np.isnan(x)
    w_masked = np.where(mask, w * u, np.nan)
    total_weight = np.nansum(w_masked)
    mean_x = np.nansum(w_masked * x) / total_weight
    
    dev = x - mean_x
    var = np.nansum(w_masked * (dev ** 2)) / total_weight
    
    acf_values = Parallel(n_jobs=num_workers)(
        delayed(compute_acf_unconditional)(lag, x, w_masked, mean_x, dev, var) for lag in lags
    )

    return acf_values

# Example usage
df_x = pd.DataFrame(np.random.rand(100))
df_w = pd.DataFrame(np.random.rand(100))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(100,), p=[0.5, 0.5]))
lags = [1, 2, 3]
num_workers = 4
print(acf_unconditional_parallel_joblib(df_x, df_w, df_u, lags, num_workers))






######### conditional without multi; std_err_bar

import numpy as np
import pandas as pd

def compute_acf_for_lag(lag, x, w_masked, mean_x, dev, var):
    if lag > 0:
        x_shifted = np.roll(x, lag)
        x_shifted[:lag] = np.nan
    else:
        x_shifted = x

    valid_weights = w_masked * ~np.isnan(x_shifted)
    autcov = np.nansum(dev * (x_shifted - mean_x) * valid_weights)
    normalization = np.nansum(valid_weights)

    if normalization == 0:
        return float('nan'), float('nan')
    else:
        autcov /= normalization
        acf_value = autcov / var
        effective_sample_size = np.nansum(valid_weights) ** 2 / np.nansum(valid_weights ** 2)
        std_error = 1 / np.sqrt(effective_sample_size)
        return acf_value, std_error

def weighted_acf_conditional(df_x, df_w, df_u, df_cap_daily, lags, num_categories, split_data=False):
    x = df_x.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()
    cap_daily = df_cap_daily.to_numpy()

    df_u_daily = df_u.at_time('10:00:00').astype(float)
    valid_daily_mktcap = df_u_daily.to_numpy() * cap_daily

    ranked_mktcap = np.nanpercentile(valid_daily_mktcap, np.arange(0, 101, 100 / num_categories), axis=1)
    groups = np.digitize(valid_daily_mktcap, ranked_mktcap) - 1

    if split_data:
        halves = {
            'first half': df_x.loc[:'2015-12-31'].to_numpy(),
            'second half': df_x.loc['2016-01-01':].to_numpy()
        }
    else:
        halves = {'whole data': x}

    results = {}

    def process_half(category, half_name, x_half, hourly_mask):
        w_masked = np.where(np.isnan(x_half), np.nan, w * u * hourly_mask)
        total_weight = np.nansum(w_masked)

        if total_weight == 0:
            return []

        mean_x = np.nansum(w_masked * x_half) / total_weight
        dev = x_half - mean_x
        var = np.nansum(w_masked * (dev ** 2)) / total_weight

        acfs = [compute_acf_for_lag(lag, x_half, w_masked, mean_x, dev, var) for lag in lags]
        return acfs

    for category in range(num_categories):
        results[category] = {}
        mask = np.where(groups == category, 1, np.nan)
        hourly_mask = pd.DataFrame(mask, index=df_u_daily.index).reindex(index=df_u.index).groupby(df_u.index.date).ffill().to_numpy()
        
        for half_name, x_half in halves.items():
            results[category][half_name] = process_half(category, half_name, x_half, hourly_mask)

    return results

# Example usage
df_x = pd.DataFrame(np.random.rand(8000, 24000))
df_w = pd.DataFrame(np.random.rand(8000, 24000))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(8000, 24000), p=[0.5, 0.5]))
df_cap_daily = pd.DataFrame(np.random.rand(8000, 24000))

lags = [1, 2, 3]
num_categories = 5
split_data = False

results = weighted_acf_conditional(df_x, df_w, df_u, df_cap_daily, lags, num_categories, split_data)
print(results)



###### conditional with multi; std_err_bar

import numpy as np
import pandas as pd
from multiprocessing import Pool

def compute_acf_for_lag(args):
    lag, x, w_masked, mean_x, dev, var = args
    if lag > 0:
        x_shifted = np.roll(x, lag)
        x_shifted[:lag] = np.nan
    else:
        x_shifted = x

    valid_weights = w_masked * ~np.isnan(x_shifted)
    autcov = np.nansum(dev * (x_shifted - mean_x) * valid_weights)
    normalization = np.nansum(valid_weights)

    if normalization == 0:
        return float('nan'), float('nan')
    else:
        autcov /= normalization
        acf_value = autcov / var
        # Calculate effective sample size
        effective_sample_size = np.nansum(valid_weights) ** 2 / np.nansum(valid_weights ** 2)
        # Calculate standard error using effective sample size
        std_error = 1 / np.sqrt(effective_sample_size)
        return acf_value, std_error

def weighted_acf_conditional_parallel(df_x, df_w, df_u, df_cap_daily, lags, num_categories, num_workers, split_data=False):
    x = df_x.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()
    cap_daily = df_cap_daily.to_numpy()

    df_u_daily = df_u.at_time('10:00:00').astype(float)
    valid_daily_mktcap = df_u_daily.to_numpy() * cap_daily

    ranked_mktcap = np.nanpercentile(valid_daily_mktcap, np.arange(0, 101, 100 / num_categories), axis=1)
    groups = np.digitize(valid_daily_mktcap, ranked_mktcap) - 1

    if split_data:
        halves = {
            'first half': df_x.loc[:'2015-12-31'].to_numpy(),
            'second half': df_x.loc['2016-01-01':].to_numpy()
        }
    else:
        halves = {'whole data': x}

    results = {}

    def process_half(category, half_name, x_half, hourly_mask):
        w_masked = np.where(np.isnan(x_half), np.nan, w * u * hourly_mask)
        total_weight = np.nansum(w_masked)

        if total_weight == 0:
            return []

        mean_x = np.nansum(w_masked * x_half) / total_weight
        dev = x_half - mean_x
        var = np.nansum(w_masked * (dev ** 2)) / total_weight

        with Pool(num_workers) as pool:
            acfs = pool.map(compute_acf_for_lag, [(lag, x_half, w_masked, mean_x, dev, var) for lag in lags])
        return acfs

    for category in range(num_categories):
        results[category] = {}
        mask = np.where(groups == category, 1, np.nan)
        hourly_mask = pd.DataFrame(mask, index=df_u_daily.index).reindex(index=df_u.index).groupby(df_u.index.date).ffill().to_numpy()
        
        for half_name, x_half in halves.items():
            results[category][half_name] = process_half(category, half_name, x_half, hourly_mask)

    return results

# Example usage
df_x = pd.DataFrame(np.random.rand(8000, 24000))
df_w = pd.DataFrame(np.random.rand(8000, 24000))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(8000, 24000), p=[0.5, 0.5]))
df_cap_daily = pd.DataFrame(np.random.rand(8000, 24000))

lags = [1, 2, 3]
num_categories = 3
num_workers = 4
split_data = False

results = weighted_acf_conditional_parallel(df_x, df_w, df_u, df_cap_daily, lags, num_categories, num_workers, split_data)
print(results)