from ace import model  # https://pypi.org/project/ace/
from scipy.spatial.distance import pdist
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
import numpy as np
from sklearn.metrics import mutual_info_score


def variation_of_information(x, y, norm=False):
    """
    Computes the Variation of Information distance matrix used for feature clustering.
    """
    bXY = 5
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)  # mutual information
    hX = ss.entropy(np.histogram(x, bXY)[0])  # marginal
    hY = ss.entropy(np.histogram(y, bXY)[0])  # marginal
    vXY = hX + hY - 2 * iXY  # variation of information
    if norm:
        hXY = hX + hY - iXY  # joint
        vXY /= hXY  # normalized variation of information
    return vXY



def max_corr(x, y):
    """
    Computes the Maximum Correlation distance matrix used for feature clustering.
    """
    # Credit goes to https://mlfinlab.readthedocs.io/en/latest/implementations/codependence.html
    ace_model = model.Model()

    ace_model.build_model_from_xy([x], y)
    val = 1 - np.corrcoef(ace_model.ace.x_transforms[0], ace_model.ace.y_transform)[0][1]
    return val


def distance_corr(Xval, Yval, pval=False, nruns=500):
    """
    Computes the Distance Correlation distance matrix used for feature clustering.
    Based on Satra/distcorr.py (gist aa3d19a12b74e9ab7941)
    """
    X = np.atleast_1d(Xval)
    Y = np.atleast_1d(Yval)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        for i in range(nruns):
            Y_r = copy.copy(Yval)
            np.random.shuffle(Y_r)
            if distcorr(Xval, Y_r, pval=False) > dcor:
                greater += 1
        return (dcor, greater / float(nruns))
    else:

        return 1 - dcor
