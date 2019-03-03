import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score

RESPONSE_VARIABLE = 'price'


def construct(train_df, k, remove_features=None):
  feature_cols = [col for col in train_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = train_df[feature_cols]
  y = train_df[RESPONSE_VARIABLE]

  model = KNeighborsRegressor(n_neighbors=k)
  model.fit(X, y)
  
  in_sample_accuracy = model.score(X, y)
  print('in-sample accuracy --> ', in_sample_accuracy)  
  return model


def mean_accuracy(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y = test_df[RESPONSE_VARIABLE]
  return model.score(X[:, np.newaxis], y)


def rmse(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y = test_df[RESPONSE_VARIABLE]
  
  # Evaluate the models using crossvalidation
  scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)
  return np.sqrt( -1 * scores.mean() )

