#! /usr/bin/env python
#-*- coding: utf-8 -*-
# For iPython notebook
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('ggplot')

### DATA IMPORT ###
# Loading 2016 past dataset for Training
#gen = pd.read_csv("training/Training_Genetic_Dataset.csv")
pfrm = pd.read_csv("training/Training_Performance_Dataset.csv")
soil = pd.read_csv("training/Training_Soil_Dataset.csv")
wtr = pd.read_csv("training/Training_Weather_Dataset.csv")

# loading 2017 dataset for TEST
#gen_t = pd.read_csv("test/Test_Genetic_Dataset.csv")
pfrm_t = pd.read_csv("test/Test_Performance_Dataset.csv")
soil_t = pd.read_csv("test/Test_Soil_Dataset.csv")
wtr_t = pd.read_csv("test/Test_Weather_Dataset.csv")


### DATAFRAME CREATION ###
# train
pfrm_df = fe4pfrm(pfrm)
wtr_df = fe4wtr(wtr)
soil_df = fe4soil(soil)

# Hybrid variance flag
#pfrm_df_flg = hybrid_var_flagger(pfrm_df)
train_df = mrg(pfrm_df_flg, wtr_df, soil_df)

# test
pfrm_df_t = fe4pfrm_t(pfrm_t)
wtr_df_t = fe4wtr_t(wtr_t)
soil_df_t = fe4soil(soil_t)

train_df_y_t = mrg_t(pfrm_df_t, wtr_df_t, soil_df_t)

# Append 2017 dataset to past dataset
df = train_df_y.append(train_df_y_t).fillna(value=0)

# Feature engineering for Training dataset
def fe4pfrm(pfrm):
    p1 = pfrm.Hybrid.str[0:5]
    p2 = pfrm.Hybrid.str[6:11]
    p1_df = p1.to_frame(name='p1')
    p2_df = p2.to_frame(name='p2')

    # Normalize Lat and Lon
    latlon_norm = pfrm[['Latitude', 'Longitude']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    latlon_norm = latlon_norm.rename(columns={'Latitude': 'Lat_n', 'Longitude': 'Lon_n'})
    #latlon_norm['lat_n_pow'] = latlon_norm['Lat_n'] ** 2
    #latlon_norm['lon_n_pow'] = latlon_norm['Lon_n'] ** 2

    #latlon_norm = pfrm[['Latitude', 'Longitude']]
    #pfrm_drp = pfrm.drop(['Latitude', 'Longitude'], axis=1)

    mrg = pd.concat([pfrm, p1_df, p2_df, latlon_norm], axis=1, join_axes=[pfrm.index])
    mrg_drp = mrg.drop(['Yield', 'Check_Yield'], axis=1)
    # One-hot Encoding
    p1_hot = pd.get_dummies(mrg_drp, prefix ='', prefix_sep='', columns=['p1'])
    p2_hot = pd.get_dummies(mrg_drp, prefix ='', prefix_sep='', columns=['p2'])
    p1_drp = p1_hot.drop(['Hybrid', 'Year', 'Maturity_Group', 'Location_ID', 'Latitude', 'Longitude', 'Lat_n', 'Lon_n', 'Yield_Difference', 'HighV_Flag', 'p2'], axis=1)
    p2_drp = p2_hot.drop(['Hybrid', 'Year', 'Maturity_Group', 'Location_ID', 'Latitude', 'Longitude', 'Lat_n', 'Lon_n', 'Yield_Difference', 'HighV_Flag', 'p1'], axis=1)
    parents_df = p1_drp.add(p2_drp, fill_value=0).astype(int)

    # Merge parents_df
    mrg_drp_seed = mrg_drp.drop(['Hybrid', 'p1', 'p2'], axis=1)
    mrg_drp_seed['M_Group_pow'] = mrg_drp_seed.Maturity_Group ** 2
    out = pd.concat([mrg_drp_seed, parents_df], axis=1)
    return out

# for Weather
def fe4wtr(wtr):
    # Chose May-October
    # Calc Rate of Change between month
    w_vars = []
    for i in range(1, 7):
        for m in range(5, 11): # Corn grow from May to October 5, 10
            w_vars.append('w_' + str(i) + '_' + str(m))
    wtr_sbt = wtr[['Location_ID', 'Year'] + w_vars]
    for i in range(1, 7):
        # initialize sum list
        wtr_i_sum = wtr['w_' + str(i) + '_4']
        for m in range(5, 11): # Corn grow from May to October 5, 10
            wtr_sbt = wtr_sbt.join(pd.DataFrame(wtr['w_' + str(i) + '_' + str(m + 1)] - wtr['w_' + str(i) + '_' + str(m)], columns=['w_' + str(i) + '_' + str(m + 1) + 'minus' + str(m)]))
            wtr_sbt = wtr_sbt.join(pd.DataFrame(wtr['w_' + str(i) + '_' + str(m + 1)] * wtr['w_' + str(i) + '_' + str(m)], columns=['w_' + str(i) + '_' + str(m + 1) + 'multi' + str(m)]))
            wtr_sbt['w_' + str(i) + '_' + str(m) + 'pow'] = wtr['w_' + str(i) + '_' + str(m)] **2
            wtr_i_sum += wtr['w_' + str(i) + '_' + str(m)]
        wtr_i_mean = wtr_i_sum / 12
        wtr_sbt = wtr_sbt.join(wtr_i_sum.rename('w_sum_' + str(i)))
        wtr_sbt = wtr_sbt.join(wtr_i_mean.rename('w_mean_' + str(i)))
    return wtr_sbt

# for Soil
def fe4soil(soil):
    # Drop Latitude and Longitude
    soil_drp = soil.drop(['Latitude', 'Longitude'], axis=1)
    var_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    for v in var_list:
        soil_drp[v + 'pow'] = soil_drp[v] ** 2
    return soil_drp

# Add flag of high variance hybrid pair
def hybrid_var_flagger(df_in):
    # calc variance under same Hybrid pair, same Year, and same Location_ID
    df = df_in[['Hybrid', 'Year', 'Location_ID', 'Yield_Difference']]
    df = df[df.Year < 2016]
    # variance of each combination
    each_var = df.groupby(['Hybrid', 'Year', 'Location_ID'])['Yield_Difference'].var()
    # variance of each
    each_var_drpna = pd.DataFrame(each_var.dropna()).reset_index()
    each_hybrid_var_mean = each_var_drpna.groupby(['Hybrid'])['Yield_Difference'].mean()
    # threshhold of high variance
    th = each_hybrid_var_mean.describe()[6] # 6: 75%tile, 5: 50%tile, 4: 25%tile, 3: ALL

    # Sort
    df_sorted = pd.DataFrame(each_hybrid_var_mean.sort_values(ascending=False)).reset_index()
    df_sorted.to_csv('hybrid_variance.csv')

    # High Variance: 1, Low Variance: 0
    df_sorted['HighV_Flag'] = df_sorted['Yield_Difference'].apply(lambda x: 1 if x > th else 0)
    df_sorted = df_sorted.drop(['Yield_Difference'], axis=1)

    # merge the flag and return
    out = pd.merge(df_in, df_sorted, on=['Hybrid'], how='left')
    # the define of only once appeared seed hybrid
    # 1: classify as high variance, 0: classify as low variance
    out = out.fillna(value = 0)
    #out = out.fillna(value = 1)
    return out

# Merge Performance, Soil, Weather
def mrg(pfrm, wtr, soil):
    mrg_pf_soil = pd.merge(pfrm, soil, on=['Location_ID'])
    mrg_pf_soil_wtr = pd.merge(mrg_pf_soil, wtr, on=['Location_ID', 'Year'])
    mrg_pf_soil_wtr_drp = mrg_pf_soil_wtr.drop(['Location_ID'], axis=1)
    return mrg_pf_soil_wtr_drp

# Feature eingineering for Test (2017) dataset
# for Performance
def fe4pfrm_t(pfrm):
    p1 = pfrm.Hybrid.str[0:5]
    p2 = pfrm.Hybrid.str[6:11]
    p1_df = p1.to_frame(name='p1')
    p2_df = p2.to_frame(name='p2')

    # Normalize Lat and Lon
    latlon_norm = pfrm[['Latitude', 'Longitude']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    latlon_norm = latlon_norm.rename(columns={'Latitude': 'Lat_n', 'Longitude': 'Lon_n'})

    mrg = pd.concat([pfrm, p1_df, p2_df, latlon_norm], axis=1, join_axes=[pfrm.index])
    mrg_drp = mrg#.drop(['Yield', 'Check_Yield'], axis=1)
    # One-hot Encoding
    p1_hot = pd.get_dummies(mrg_drp, prefix ='', prefix_sep='', columns=['p1'])
    p2_hot = pd.get_dummies(mrg_drp, prefix ='', prefix_sep='', columns=['p2'])
    p1_drp = p1_hot.drop(['Hybrid', 'Year', 'Maturity_Group', 'Test_Location_ID', 'Latitude', 'Longitude', 'Lat_n', 'Lon_n', 'p2'], axis=1)
    p2_drp = p2_hot.drop(['Hybrid', 'Year', 'Maturity_Group', 'Test_Location_ID', 'Latitude', 'Longitude', 'Lat_n', 'Lon_n', 'p1'], axis=1)
    parents_df = p1_drp.add(p2_drp, fill_value=0).astype(int)

    # Merge parents_df
    mrg_drp_seed = mrg_drp.drop(['Hybrid', 'p1', 'p2'], axis=1)
    mrg_drp_seed['M_Group_pow'] = mrg_drp_seed.Maturity_Group ** 2
    out = pd.concat([mrg_drp_seed, parents_df], axis=1)
    return out

# for Weather
def fe4wtr_t(wtr):
    # Chose May-October
    # Calc Rate of Change between month
    w_vars = []
    for i in range(1, 7):
        for m in range(5, 11): # Corn grow from May to October 5, 10
            w_vars.append('w_' + str(i) + '_' + str(m))
    wtr_sbt = wtr[['Test_Location_ID', 'Year'] + w_vars]
    for i in range(1, 7):
        # initialize sum list
        wtr_i_sum = wtr['w_' + str(i) + '_4']
        for m in range(5, 11): # Corn grow from May to October 5, 10
            wtr_sbt = wtr_sbt.join(pd.DataFrame(wtr['w_' + str(i) + '_' + str(m + 1)] - wtr['w_' + str(i) + '_' + str(m)], columns=['w_' + str(i) + '_' + str(m + 1) + 'minus' + str(m)]))
            wtr_sbt = wtr_sbt.join(pd.DataFrame(wtr['w_' + str(i) + '_' + str(m + 1)] * wtr['w_' + str(i) + '_' + str(m)], columns=['w_' + str(i) + '_' + str(m + 1) + 'multi' + str(m)]))
            wtr_sbt['w_' + str(i) + '_' + str(m) + 'pow'] = wtr['w_' + str(i) + '_' + str(m)] **2
            wtr_i_sum += wtr['w_' + str(i) + '_' + str(m)]
        wtr_i_mean = wtr_i_sum / 12
        wtr_sbt = wtr_sbt.join(wtr_i_sum.rename('w_sum_' + str(i)))
        wtr_sbt = wtr_sbt.join(wtr_i_mean.rename('w_mean_' + str(i)))
    return wtr_sbt

# for Soil
def fe4soil_t(soil):
    # Drop Latitude and Longitude
    soil_drp = soil.drop(['Latitude', 'Longitude'], axis=1)
    var_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    for v in var_list:
        soil_drp[v + 'pow'] = soil_drp[v] ** 2
    return soil_drp

# Merge Performance, Soil, Weather
def mrg_t(pfrm, wtr, soil):
    mrg_pf_soil = pd.merge(pfrm, soil, on=['Test_Location_ID'])
    mrg_pf_soil_wtr = pd.merge(mrg_pf_soil, wtr, on=['Test_Location_ID', 'Year'])
    mrg_pf_soil_wtr_drp = mrg_pf_soil_wtr.drop(['Test_Location_ID'], axis=1)
    return mrg_pf_soil_wtr_drp


## MODEL CREATION ###
# Baseline RMSE
# Predict average of Yield_Difference
base_rmse = ((((pfrm.Yield_Difference - pfrm.Yield_Difference.mean())**2).sum())*(1/len(pfrm.index)))**(1/2)
base_rmse = ((((df_valid.Yield_Difference - df_train.Yield_Difference.mean())**2).sum())*(1/len(df_valid.index)))**(1/2)
# base_rmse: 14.73

# w/ outliners dataset
#df = train_df_y[train_df.HighV_Flag == 0.0]
#df = train_df_full
df_train_f = df[df.Year < 2017].drop(['Year'], axis = 1)
df_train_f = df_train_f[df_train_f.HighV_Flag == 0.0].drop(['HighV_Flag'], axis=1)
#df_valid = df[df.Year == 2016].drop(['Year', 'HighV_Flag'], axis = 1)
df_valid_f = df[df.Year == 2016].drop(['Year'], axis = 1)
df_valid_f = df_valid_f[df_valid_f.HighV_Flag == 0.0].drop(['HighV_Flag'], axis=1)
df_17_f = df[df.Year == 2017].drop(['Year', 'HighV_Flag'], axis = 1)

# for training
X_train_f = df_train_f.drop(['Yield_Difference'], axis = 1).iloc[:, :].values
X_test_f = df_valid_f.drop(['Yield_Difference'], axis = 1).iloc[:, :].values
y_train_f = df_train_f.loc[:, 'Yield_Difference'].values
y_test_f = df_valid_f.loc[:, 'Yield_Difference'].values
# for test
X_17_f = df_17_f.drop(['Yield_Difference'], axis = 1).iloc[:, :].values

### Prediction
## (0) benchmark prediction RMSE (using average)
base_rmse = ((((df_valid.Yield_Difference - df_train.Yield_Difference.mean())**2).sum())*(1/len(df_valid.index)))**(1/2)

# Split Dataset for Train and Validation
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)

## (1) RANDOM FOREST
# Import Library of Randam Forest Reg
from sklearn.ensemble import RandomForestRegressor
# Using vanilla parameters
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
importances = forest.feature_importances_
# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
print('RMSE train : %.3f, test : %.3f' % (sqrt(mean_squared_error(y_train, y_train_pred)), sqrt(mean_squared_error(y_test, y_test_pred))))
# R^2
from sklearn.metrics import r2_score
print('R2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

## (2) XGBOOST
import xgboost as xgb
# Model Instance
mod = xgb.XGBRegressor()
mod.fit(X_train, y_train)
# CV for XGBoost
from sklearn.grid_search import GridSearchCV
params = {'max_depth': [10],
          'min_child_weight': [1],
          'gamma': [0],
          'learning_rate': [0.05, 0.08, 0.1],
          'subsample': [0.5, 0.8, 1.0],
          'colsample_bytree': [0.2, 0.5, 0.8]}
# Model Instance
mod = xgb.XGBRegressor()
# 10-fold Cross Validation
cv = GridSearchCV(mod, params, cv=10, scoring='neg_mean_squared_error', n_jobs=1, verbose=2)
cv.fit(X_train, y_train)

# RMSE of XGBoost
from sklearn.metrics import mean_squared_error
from math import sqrt
y_train_pred = cv.predict(X_train)
y_test_pred = cv.predict(X_test)
#importances = cv.importances_
print('RMSE train : %.3f, test : %.3f' % (sqrt(mean_squared_error(y_train, y_train_pred)), sqrt(mean_squared_error(y_test, y_test_pred))))
# R^2
from sklearn.metrics import r2_score
print('R2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

## (3) Light GBM
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 70,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5
               )

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)



### Weather prediction
def pred_by_mean(df_wide):
    pred_mean = df_wide[['2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1)
    pred_mean.reset_index(drop=True, inplace=True)
    df_wide.reset_index(drop=True, inplace=True)

    sabun = df_wide['2016'] - pred_mean
    rmse = (sum(sabun ** 2) / len(sabun)) ** (1/2)
    return round(rmse, 4)

def pred_by_randomforest(df, df17):
    df_x = df[['Latitude', 'Longitude', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    df_y = df['2016']
    X = df_x.values
    y = df_y.values
    # Split dataset for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.1, random_state=0)
    # Using vanilla parameters
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_test_pred))

    # apply randam forest for 2017
    df_x17 = df17[['Latitude', 'Longitude', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']]
    X17 = df_x17.values
    y17 = forest.predict(X17)

    return round(rmse, 4), y17

def wtr_proc(wtr):
    # pre-processing of weather data
    df = wtr[['Location_ID', 'Latitude', 'Longitude', 'Year', target]]
    df['Year'] = df.Year.astype(str)
    df_wide = df.pivot_table(index=['Location_ID', 'Latitude', 'Longitude'], columns='Year', values=target)
    df_wide.reset_index(inplace=True)
    df_wide = df_wide.drop(['2001', '2002', '2003'], axis=1).dropna()
    return df_wide

for i in range(1, 7):
    for m in range(1, 13):
        target = 'w_' + str(i) + '_' + str(m)
        df_wide = wtr_proc(wtr)
        df_wide17 = wtr_proc(wtr17)
        # use mean
        rmse_mean = pred_by_mean(df_wide)
        # use random forest regressor
        rmse_rfr, y17 = pred_by_randomforest(df_wide, df_wide17)

        rmse_list = pd.Series([rmse_mean, rmse_rfr], name=target)
        y17_series = pd.Series(y17, name=target)
        if i == 1 and m == 1:
            rmse_df = rmse_list
            y17_df = y17_series
        else:
            rmse_df = pd.concat([rmse_df, rmse_list], axis=1)
            y17_df = pd.concat([y17_df, y17_series], axis=1)

# RMSE comparison table
rmse_table = rmse_df.T
rmse_table.rename(columns={0: 'RMSE_avg', 1: 'RMSE_rfr'}, inplace=True)

# Bind predicted weather with Location_ID
wtr17 = pd.concat([df_wide17[['Location_ID']], y17_df], axis=1)
