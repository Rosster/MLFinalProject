import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

## adapted from https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py


def svc_plot(model, X0, X1, y, xlabel, ylabel, title, dest_fqp):
  xx, yy = __make_meshgrid(X0, X1)
  figure, sub = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), dpi=120, facecolor='w', edgecolor='k')
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  ax = sub
  # pylint: disable=no-member
  __plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
  # pylint: disable=no-member
  ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xticks(())
  ax.set_yticks(())
  ax.set_title(title)
  plt.savefig(dest_fqp)
  plt.close(figure)


def __make_meshgrid(x, y, h=.02):
  """Create a mesh of points to plot in

  Parameters
  ----------
  x: data to base x-axis meshgrid on
  y: data to base y-axis meshgrid on
  h: stepsize for meshgrid, optional

  Returns
  -------
  xx, yy : ndarray
  """
  x_min, x_max = x.min() - 1, x.max() + 1
  y_min, y_max = y.min() - 1, y.max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
  return xx, yy


def __plot_contours(ax, clf, xx, yy, **params):
  """Plot the decision boundaries for a classifier.

  Parameters
  ----------
  ax: matplotlib axes object
  clf: a classifier
  xx: meshgrid ndarray
  yy: meshgrid ndarray
  params: dictionary of params to pass to contourf, optional
  """
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  out = ax.contourf(xx, yy, Z, **params)
  return out
