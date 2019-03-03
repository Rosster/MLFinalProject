from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn import metrics
import numpy as np


RESPONSE_VARIABLE = 'sentiment'

def construct(train_df, algo, opts={}, remove_features=None):
  feature_cols = [col for col in train_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = train_df[feature_cols]
  y = train_df[RESPONSE_VARIABLE]

  if algo == 'RandomForestClassifier':
    model = RandomForestClassifier(
      n_estimators=1000,
      max_depth=opts.get('max_depth'),
      min_samples_split=opts.get('min_samples_split', 2),
      random_state=10,
      bootstrap=True,
      n_jobs=-1,
      max_features='sqrt', # If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
      verbose=1
    )
  elif algo == 'GradientBoostingClassifier':
    model = GradientBoostingClassifier(
      loss='deviance', # least squares
      learning_rate=opts.get('learning_rate', 0.1),
      subsample=1.0,
      n_estimators=1000,
      criterion='friedman_mse',
      max_depth=opts.get('max_depth', 3),
      min_samples_split=opts.get('min_samples_split', 3),
      min_samples_leaf=1,
      min_weight_fraction_leaf=0.0,
      random_state=10,
      max_features=None,
      max_leaf_nodes=None,
      verbose=1
    )
  
  model.fit(X, y)
  feature_importances = []
  # print('='*20, 'Feature Importance:')
  # importances = list(model.feature_importances_)
  # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_cols, importances)]
  # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
  # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

  in_sample_accuracy = model.score(X, y)
  print('in-sample accuracy --> ', in_sample_accuracy)
  return model, feature_importances


def get_predictions(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y_pred = model.predict(X)
  return y_pred


def mean_accuracy(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y = test_df[RESPONSE_VARIABLE]

  return model.score(X, y) # Mean accuracy, where single tree accuracy is 1 - missclassification rate

