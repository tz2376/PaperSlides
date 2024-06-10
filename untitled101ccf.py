#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:11:45 2024

@author: tonyzhang
"""

import numpy as np
import pandas as pd

def compute_ccf_for_lag(lag, x, y, w_masked, mean_x, mean_y, dev_x, dev_y, var_x, var_y):
    if lag > 0:
        y_shifted = np.roll(y, lag)
        y_shifted[:lag] = np.nan
    else:
        y_shifted = y

    valid_weights = w_masked * ~np.isnan(y_shifted)
    crosscov = np.nansum(dev_x * (y_shifted - mean_y) * valid_weights)
    normalization = np.nansum(valid_weights)

    if normalization == 0:
        return float('nan'), float('nan')
    else:
        crosscov /= normalization
        ccf_value = crosscov / np.sqrt(var_x * var_y)
        effective_sample_size = np.nansum(valid_weights) ** 2 / np.nansum(valid_weights ** 2)
        std_error = 1 / np.sqrt(effective_sample_size)
        return ccf_value, std_error

def weighted_ccf_unconditional(df_x, df_y, df_w, df_u, lags):
    x = df_x.to_numpy()
    y = df_y.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()

    mask = ~np.isnan(x) & ~np.isnan(y)
    w_masked = np.where(mask, w * u, np.nan)
    total_weight = np.nansum(w_masked)

    if total_weight == 0:
        return [(float('nan'), float('nan')) for _ in lags]

    mean_x = np.nansum(w_masked * x) / total_weight
    mean_y = np.nansum(w_masked * y) / total_weight
    dev_x = x - mean_x
    dev_y = y - mean_y
    var_x = np.nansum(w_masked * (dev_x ** 2)) / total_weight
    var_y = np.nansum(w_masked * (dev_y ** 2)) / total_weight

    ccf_values = [compute_ccf_for_lag(lag, x, y, w_masked, mean_x, mean_y, dev_x, dev_y, var_x, var_y) for lag in lags]
    return ccf_values

# Example usage
df_x = pd.DataFrame(np.random.rand(100))
df_y = pd.DataFrame(np.random.rand(100))
df_w = pd.DataFrame(np.random.rand(100))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(100,), p=[0.5, 0.5]))
lags = [1, 2, 3]
results = weighted_ccf_unconditional(df_x, df_y, df_w, df_u, lags)
print(results)



import numpy as np
import pandas as pd

def compute_ccf_for_lag(lag, x, y, w_masked, mean_x, mean_y, dev_x, dev_y, var_x, var_y):
    if lag > 0:
        y_shifted = np.roll(y, lag)
        y_shifted[:lag] = np.nan
    else:
        y_shifted = y

    valid_weights = w_masked * ~np.isnan(y_shifted)
    crosscov = np.nansum(dev_x * (y_shifted - mean_y) * valid_weights)
    normalization = np.nansum(valid_weights)

    if normalization == 0:
        return float('nan'), float('nan')
    else:
        crosscov /= normalization
        ccf_value = crosscov / np.sqrt(var_x * var_y)
        effective_sample_size = np.nansum(valid_weights) ** 2 / np.nansum(valid_weights ** 2)
        std_error = 1 / np.sqrt(effective_sample_size)
        return ccf_value, std_error

def weighted_ccf_conditional(df_x, df_y, df_w, df_u, df_cap_daily, lags, num_categories, split_data=False):
    x = df_x.to_numpy()
    y = df_y.to_numpy()
    w = df_w.to_numpy()
    u = df_u.to_numpy()
    cap_daily = df_cap_daily.to_numpy()

    df_u_daily = df_u.at_time('10:00:00').astype(float)
    valid_daily_mktcap = df_u_daily.to_numpy() * cap_daily

    ranked_mktcap = np.nanpercentile(valid_daily_mktcap, np.arange(0, 101, 100 / num_categories), axis=1)
    groups = np.digitize(valid_daily_mktcap, ranked_mktcap) - 1

    if split_data:
        halves = {
            'first half': (df_x.loc[:'2015-12-31'].to_numpy(), df_y.loc[:'2015-12-31'].to_numpy()),
            'second half': (df_x.loc['2016-01-01':].to_numpy(), df_y.loc['2016-01-01':].to_numpy())
        }
    else:
        halves = {'whole data': (x, y)}

    results = {}

    def process_half(category, half_name, x_half, y_half, hourly_mask):
        w_masked = np.where(np.isnan(x_half) | np.isnan(y_half), np.nan, w * u * hourly_mask)
        total_weight = np.nansum(w_masked)

        if total_weight == 0:
            return []

        mean_x = np.nansum(w_masked * x_half) / total_weight
        mean_y = np.nansum(w_masked * y_half) / total_weight
        dev_x = x_half - mean_x
        dev_y = y_half - mean_y
        var_x = np.nansum(w_masked * (dev_x ** 2)) / total_weight
        var_y = np.nansum(w_masked * (dev_y ** 2)) / total_weight

        ccf_values = [compute_ccf_for_lag(lag, x_half, y_half, w_masked, mean_x, mean_y, dev_x, dev_y, var_x, var_y) for lag in lags]
        return ccf_values

    for category in range(num_categories):
        results[category] = {}
        mask = np.where(groups == category, 1, np.nan)
        hourly_mask = pd.DataFrame(mask, index=df_u_daily.index).reindex(index=df_u.index).groupby(df_u.index.date).ffill().to_numpy()
        
        for half_name, (x_half, y_half) in halves.items():
            results[category][half_name] = process_half(category, half_name, x_half, y_half, hourly_mask)

    return results

# Example usage
df_x = pd.DataFrame(np.random.rand(8000, 24000))
df_y = pd.DataFrame(np.random.rand(8000, 24000))
df_w = pd.DataFrame(np.random.rand(8000, 24000))
df_u = pd.DataFrame(np.random.choice([True, np.nan], size=(8000, 24000), p=[0.5, 0.5]))
df_cap_daily = pd.DataFrame(np.random.rand(8000, 24000))

lags = [1, 2, 3]
num_categories = 5
split_data = False

results = weighted_ccf_conditional(df_x, df_y, df_w, df_u, df_cap_daily, lags, num_categories, split_data)
print(results)



