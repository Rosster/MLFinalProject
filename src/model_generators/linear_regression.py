import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics

RESPONSE_VARIABLE = None

def construct(X, y):
  lm = LinearRegression()
  model = lm.fit(X=X, y=y)
  return model


def rmse(model, test_df, independent_var, dependent_var):
  X = test_df[independent_var].values.reshape(-1, 1)
  y = test_df[dependent_var].values.reshape(-1, 1)
  y_pred = model.predict(X)
  rmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
  return rmse


def mean_accuracy(model, test_df, independent_var):
  X = test_df[independent_var]
  y = test_df[RESPONSE_VARIABLE]
  return model.score(X[:, np.newaxis], y)


def determine_outliers(df, col_name, m=3):
  y = df[col_name]
  X = np.arange(0, len(y), 1.0)
  X = X.reshape(-1, 1)
  model = construct(X=X, y=y)
  y_pred = [el[0] for el in [model.predict([i]) for i in X]]
  diffs = []
  for i in range(0, len(y)):
    diff = abs(y[i] - y_pred[i])
    diffs.append(diff)
  stdev, avg = np.std(diffs), np.mean(diffs)
  lower_bound, upper_bound = avg-(m*stdev), avg+(m*stdev)
  print({'lower_bound': lower_bound, 'upper_bound': upper_bound})
  outliers = [ el for el in diffs if el > upper_bound or el < lower_bound]
  return outliers

  
