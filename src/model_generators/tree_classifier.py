from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV


RESPONSE_VARIABLE = None

def grid_search_cv_tuning(train_X, train_y, algo):  
  X = train_X
  y = train_y

  if algo == 'RandomForestClassifier':
    model = RandomForestClassifier()
    parameters = {
      "min_samples_split": range(2, 5, 1),
      "min_samples_leaf": range(1, 5, 1),
      "max_depth":[3,8,10,15],
      "max_features":["log2","sqrt"],
      "criterion": ["gini",  "entropy"],
      "n_estimators":[100]
    }

  elif algo == 'GradientBoostingClassifier':
    model = GradientBoostingClassifier()
    parameters = {
      "loss":["deviance"],
      "learning_rate": [0.01, 0.05, 0.1, 0.2],
      "min_samples_split": range(2, 5, 1),
      "min_samples_leaf": range(1, 5, 1),
      "max_depth":[3,8,10,15],
      "max_features":["log2","sqrt"],
      "criterion": ["friedman_mse"],
      "subsample":[0.5, 0.85, 1.0],
      "n_estimators":[100]
    }

  elif algo == 'DecisionTreeClassifier':
    model = DecisionTreeClassifier()
    parameters = {
      "criterion": ["gini",  "entropy"],
      "min_samples_split": range(2, 5, 1),
      "min_samples_leaf": range(1, 5, 1),
      "max_depth":[3,8,10,15],
      "max_features":["log2","sqrt"],
    }

  grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)
  grid_search.fit(X, y) 
  
  print('best_params:', grid_search.best_params_)
  print('best_score:', grid_search.best_score_)
  return {'best_params': grid_search.best_params_, 'best_score': grid_search.best_score_}


def construct(train_X, train_y, algo, opts={}):
  X = train_X
  y = train_y

  if algo == 'RandomForestClassifier':
    model = RandomForestClassifier(
      criterion= opts.get('criterion', 'gini'),
      n_estimators=1000,
      max_depth=opts.get('max_depth', 15),
      min_samples_split=opts.get('min_samples_split', 3),
      min_samples_leaf=opts.get('min_samples_leaf', 3),
      random_state=1,
      bootstrap=True,
      n_jobs=-1,
      max_features=opts.get('max_features', 'log2'), # If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
      verbose=1
    )
  elif algo == 'GradientBoostingClassifier':
    model = GradientBoostingClassifier(
      loss='deviance', # Least squares
      learning_rate=0.005, 
      n_estimators=1000, 
      subsample=1.0, 
      criterion='friedman_mse', 
      min_samples_split=2, 
      min_samples_leaf=3,
      min_weight_fraction_leaf=0.0, 
      max_depth=10, 
      min_impurity_decrease=0.0, 
      min_impurity_split=None, 
      init=None, 
      random_state=None, 
      max_features='sqrt', 
      verbose=1, 
      max_leaf_nodes=None, 
      warm_start=False, 
      presort='auto', 
      validation_fraction=0.1, 
      n_iter_no_change=None, 
      tol=0.0001
    )
  elif algo == 'DecisionTreeClassifier':
    model = DecisionTreeClassifier(
      criterion='gini', 
      splitter='best',
      max_depth=8,
      min_samples_split=3,
      min_samples_leaf=2,
      min_weight_fraction_leaf=0.0,
      max_features='log2',
      random_state=1,
      max_leaf_nodes=None,
      min_impurity_decrease=0.0,
      min_impurity_split=None,
      class_weight=None,
      presort=False
    )

  model.fit(X, y)
  # feature_importances = []
  # print('='*20, 'Feature Importance:')
  # importances = list(model.feature_importances_)
  # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_cols, importances)]
  # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
  # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

  in_sample_accuracy = model.score(X, y)
  print('in-sample accuracy --> ', in_sample_accuracy)
  return model


def get_predictions(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y_pred = model.predict(X)
  return y_pred


def get_prediction_probability(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y_pred = model.predict_proba(X)
  return y_pred


def mean_accuracy(model, test_X, test_y):
  X, y = test_X, test_y
  return model.score(X, y) # Mean accuracy, where single tree accuracy is 1 - missclassification rate

