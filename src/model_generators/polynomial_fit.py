import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Explore GridSearchCV 

RESPONSE_VARIABLE = 'price'

def construct(train_df, opts, independent_var):
  X = train_df[independent_var].values
  y = train_df[RESPONSE_VARIABLE].values

  poly_features = PolynomialFeatures(degree=opts.get('degree', 3), include_bias=False)
  linear_regression_model = LinearRegression()
  pipeline = Pipeline(
    [("polynomial_features", poly_features), ("linear_regression", linear_regression_model)
  ])
  pipeline.fit(X[:, np.newaxis], y)

  # Evaluate the models using crossvalidation
  scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=5)

  in_sample_accuracy = pipeline.score(X[:, np.newaxis], y)
  print('in-sample accuracy --> ', in_sample_accuracy)
  print('cross-validation in_sample_rmse', np.sqrt( -1 * scores.mean() ))
  return pipeline



def mean_accuracy(model, test_df, independent_var):
  X = test_df[independent_var]
  y = test_df[RESPONSE_VARIABLE]

  return model.score(X[:, np.newaxis], y)


def rmse(model, test_df, independent_var, dependent_var):
  X = test_df[independent_var].values
  y = test_df[dependent_var].values
  
  # Evaluate the models using crossvalidation
  scores = cross_val_score(model, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=5)
  return np.sqrt( -1 * scores.mean() )