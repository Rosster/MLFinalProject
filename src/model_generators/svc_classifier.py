from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import pandas as pd

RESPONSE_VARIABLE = None

def grid_search_cv_tuning(train_df, remove_features=None, debug=False):
  if not RESPONSE_VARIABLE:
    raise ValueError('Must define Response Variable')

  feature_cols = __feature_cols(train_df, remove_features)
  X = train_df[feature_cols]
  y = train_df[RESPONSE_VARIABLE]
  parameters = {
    'C': [ 0.00001, 0.00005, 0.001, 0.005, 0.01, 0.1, 1, 5, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5, 6, 7, 8], # only used if kernel = poly,
    'gamma': [0.05, 0.5, 1, 2, 3, 4, 10, 20]
  }
  model = SVC()
  grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=-1, scoring='accuracy', return_train_score=True, verbose=2)
  grid_search.fit(X, y) 
  print('best_params:', grid_search.best_params_) if debug else None
  print('best_score:', grid_search.best_score_) if debug else None
  return {'best_params': grid_search.best_params_, 'best_score': grid_search.best_score_}



def construct(train_df, opts={}, remove_features=None):
  if not RESPONSE_VARIABLE:
    raise ValueError('Must define Response Variable')

  feature_cols = __feature_cols(train_df, remove_features)
  X = train_df[feature_cols]
  y = train_df[RESPONSE_VARIABLE]
  model = SVC(
    C=opts.get('C', 1),
    kernel=opts.get('kernel', 'rbf'),
    degree=opts.get('degree', 3),
    gamma=opts.get('gamma', 'auto')
  )
  model.fit(X, y)
  return model


def __feature_cols(df, remove_features=None):
  feature_cols = [col for col in df.columns if RESPONSE_VARIABLE not in col]
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  return feature_cols


def get_predictions(model, test_df, remove_features=None):
  feature_cols = __feature_cols(test_df, remove_features)
  X = test_df[feature_cols]
  y_pred = model.predict(X)
  return y_pred


def get_confusion_matrix(model, test_df, remove_features=None):
  y_pred = get_predictions(model, test_df, remove_features)
  y_test = test_df[RESPONSE_VARIABLE]
  confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=model.classes_, columns=model.classes_)
  return confusion_matrix_df


def mean_accuracy(model, test_df, remove_features=None):
  feature_cols = __feature_cols(test_df, remove_features)
  X = test_df[feature_cols]
  y = test_df[RESPONSE_VARIABLE]
  return model.score(X, y) # Mean accuracy, where single tree accuracy is 1 - missclassification rate