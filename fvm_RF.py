# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:40:43 2020

@author: MikeyJW
"""
# TODO: Stratify player prices using pd split, sns.distplot(y) as justification

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

# Import model data
path = 'file:///C:/Users/micha/Documents/Quant/football_index/model_data.csv'
df = pd.read_csv(path)
df.set_index('PlayerName', inplace=True)

#df.drop('num_games_played', axis=1, inplace=True)

# Gen X and y matrices/vectors, leaving out forward to avoid indicator variable trap
y_col = 'CurrentPrice'
X_cols = ['ave_matchday_score',
          'Age',
          'num_games_played',
          'Midfielder',
          'Defender',
          'Goalkeeper']

y = df[y_col]
X = df[X_cols]


# TODO: JAM BOTH THESE INTO A SKLEARN PIPELINE
# Get polynomials, don't include bias (there is no point as it will be scaled to 0 later + we have a cons accounted for with indicators)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_X = poly.fit_transform(X)

# Feature scaling since we're using regularisation (+ is good to standardise when using polynomials)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(poly_X)

# Drop indicator variable squares and interactions
feature_names = poly.get_feature_names(X_cols)[:-6]
X = np.delete(scaled_X, slice(-6, None), axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# MODEL 1: Benchmark
lm = LinearRegression()
model = lm.fit(X_test, y_test)
y_pred = model.predict(X_test)
mean_absolute_error(y_test, y_pred)

ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
y_pred_ols = ols.predict(sm.add_constant(X_test))
mean_absolute_error(y_test, y_pred_ols)


# MODEL 2: LASSO

# Using non-stratified k-fold cv to find optimal alpha
lasso = Lasso()

# mae since we know we have outliers, we don't want to fit to these too much
param_grid = {'alpha': list(np.arange(0, 1, 0.005))}
model = GridSearchCV(lasso, param_grid, scoring='neg_mean_absolute_error')

results = model.fit(X_train, y_train)

np.mean(cross_val_score(lasso, X_train, y_train, scoring='neg_mean_absolute_error'))
results.cv_results_
results.best_params_

# Errr. gen a few more polynomials. Then we'll see. 
# benchmark mae from the OLS regs. Njoi boio
# Maybe this is working (just badly? test ols just to check the pipeline?)
optimised = Lasso(alpha=0.001) #TODO: Check fit intercept
coeffs = optimised.fit(X_train, y_train).coef_

coef_results = pd.Series(coeffs, index=feature_names)
# Feature importance is changing with alpha
feature_importance = pd.Series(coeffs, index=feature_names).abs().sort_values(ascending=False)

# Get your eval. metric from test set
y_pred = optimised.predict(X_test)
mean_absolute_error(y_test, y_pred)

# This is better/worse than my a priori, intuitive model, and means i would still prefer/not prefer the old model
# Try with/without giving it explicit polynomials
# CV and tune the hyper-params, see if you can get any more juice out of it. 
# We could do with stratifying, i borderline don't trust this result. Result looks good. get residuals from this???
# Just get another proj

rf = RandomForestRegressor()
model = rf.fit(X_test, y_test)
y_pred = model.predict(X_test)
mean_absolute_error(y_test, y_pred)
# 0.111 MAE

# Gboost for funsies???
gb = GradientBoostingRegressor()
model = gb.fit(X_test, y_test)
y_pred = model.predict(X_test)
mean_absolute_error(y_test, y_pred)