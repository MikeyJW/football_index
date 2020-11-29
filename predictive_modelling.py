# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:40:43 2020

@author: MikeyJW
"""
# TODO: Stratify player prices using pd split, sns.distplot(y) as justification (LATER)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Import model data
path = 'file:///C:/Users/micha/Documents/Quant/football_index/model_data.csv'
df = pd.read_csv(path)
df.set_index('PlayerName', inplace=True)

#df.drop('num_games_played', axis=1, inplace=True)

# Gen X and y matrices/vectors, leaving out forward to avoid multicollinearity problems
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
# Get polynomials, don't include bias (there is no point as it will be scaled to 0 later)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_X = poly.fit_transform(X)


# Feature scaling since we're using regularisation (+ is good to standardise when using polynomials)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(poly_X)

# Drop indicator variable squares and interactions
feature_names = poly.get_feature_names(X_cols)[:-6]
X = np.delete(scaled_X, slice(-6, None), axis=1)

# Checking multi-collinearity of new vars        
corr_table = np.corrcoef(np.transpose(X))
sns.heatmap(corr_table, annot=True, xticklabels=feature_names, 
            yticklabels=feature_names, cmap="YlGnBu")



# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Check stratification of train test since we have a skewed dataset and due to what 
# error metric we're using
fig, ax = plt.subplots()
sns.kdeplot(y_test, label='Training set', ax=ax)
sns.kdeplot(y_train, label='Testing set', ax=ax)


# MODEL 1: Benchmark 
lm = LinearRegression()
model_1 = lm.fit(X_train, y_train)
coeffs = model.coef_

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error'))

# Feature importance mapped out. Since we have scaling in the pipeline we can do this...
feature_importance = pd.Series(coeffs, index=feature_names).abs().sort_values(ascending=False)

# Plotting feature importance...
feature_importance.plot.bar()

# from plot above we want to see if trimming unimportant features may improve performance

# MODEL 2: LASSO

# Using non-stratified k-fold cv to find optimal alpha
lasso = Lasso()

# mae since we know we have outliers, we don't want to fit to these too much
param_grid = {'alpha': list(np.arange(0.005, 1, 0.005))}
model = GridSearchCV(lasso, param_grid, scoring='neg_mean_absolute_error')

results = model.fit(X_train, y_train)

print(results.best_params_)
print(results.best_score_)

# Clearly we dont have excessive model complexity here for our modest amount of training data
# no regularisation is better than some, so we decide to try increasing model complexity

# From the above, LASSO looks like it has some headroom, lets use a random forest to capture some more 
# subtle non-linearities and interaction here.

# Since rank deficiency doesn't matter for rf, we toss it these features as we want to bump complexity
# Since we've seen that the polys are mostly useful features, we pass these directly to our random forest model so it 
# can have them straight up, doesn't have to make them itself.


# First use default sklearn rf
rf = RandomForestRegressor(n_jobs=-1)
np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error'))


# Not much improvement, lets tune some hyperparams to see what we get...
# Grid search optimise the random forest

# Narrow down your options
random_grid = {'max_depth': np.arange(5, 50),
               'min_samples_split': np.arange(2, 50),
               'max_leaf_nodes': np.arange(5, 100)}  # wide net

model = RandomizedSearchCV(rf, random_grid, 
                           scoring='neg_mean_absolute_error', random_state=42,
                           n_iter=25)
results = model.fit(X_train, y_train)

print(model.best_params_)
print(model.best_score_)


# Final tune
param_grid = {'max_depth': np.arange(22, 33, 2),
               'min_samples_split': np.arange(8, 19, 2),
               'max_leaf_nodes': np.arange(63, 74, 2)}  # narrow net

model = GridSearchCV(rf, param_grid, scoring='neg_mean_absolute_error')
results = model.fit(X_train, y_train)
print(model.best_params_)
print(model.best_score_)

results = model.fit(X_train, y_train)


# Final results
# we choose polynomial lin reg model
final_model = LinearRegression().fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mean_absolute_error(y_test, y_pred_test)

# Prefer lin reg model, as occams protocol and explicit better than implicit.
# This actually generalises nicely as well.