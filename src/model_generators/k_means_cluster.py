import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import pandas as pd

RESPONSE_VARIABLE = None

def construct(train_df, k, opts={}, remove_features=None):
  if RESPONSE_VARIABLE:
    train_df = train_df.drop(RESPONSE_VARIABLE, axis=1)
  feature_cols = train_df.columns
  print('WTF --> feature_cols --> ', feature_cols)
  if remove_features:
    feature_cols = [col for col in feature_cols if col not in remove_features]
  
  print('feature_cols --> ', feature_cols)
  X = train_df[feature_cols]
  model = KMeans(
    n_clusters=k,
    init='k-means++',
    random_state=17,
    n_jobs=-1
  ).fit(X)
  cluster_labels = model.predict(X)
  centroids = model.cluster_centers_
  return model, cluster_labels, centroids


