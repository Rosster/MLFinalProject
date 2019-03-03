import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

'''
Logistic Regression is a Machine Learning classification 
algorithm that is used to predict the probability of a categorical dependent variable.
In logistic regression, the dependent variable is a binary variable that 
contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). 
In other words, the logistic regression model predicts P(Y=1) as a function of X.

Notes:
Binary logistic regression requires the dependent variable to be binary.
Only the meaningful variables should be included.
Logistic regression requires quite large sample sizes.
'''

RESPONSE_VARIABLE = 'sentiment'

def construct(train_df, opts={}, remove_features=None):
  feature_cols = [col for col in train_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = train_df[feature_cols]
  y = train_df[RESPONSE_VARIABLE]

  model = LogisticRegressionCV(
    fit_intercept=True, # Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    cv=10,
    solver='lbfgs', #For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss
    n_jobs=-1,
    random_state=27
  )

  model.fit(X, y)
  print('coeff, intercept -->', model.coef_, model.intercept_)
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


def mean_accuracy(model, test_df, remove_features=None):
  feature_cols = [col for col in test_df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  X = test_df[feature_cols]
  y = test_df[RESPONSE_VARIABLE]

  return model.score(X, y)