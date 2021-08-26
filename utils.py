# Importing depedencies
import pandas as pd
import numpy as np
import datetime
from dask import delayed, compute
from numpy import loadtxt
import lhsmdu
from sklearn.model_selection import ParameterGrid
from metrics import spearman_rank,quartic_error
import datetime


# Submissions are scored by spearman correlation
# These are functions used in different parallel functions and hypeparameter tuning.

def kfold_era(num_folds, era):
    """
    Inputs: num_folds (int) Number of k-folds to be done
            era Eras to be used when finding each fold

    Outputs: Train-test indices for each fold

    This function makes k folds of the data keeping the eras together, eras are representing a unit of time.
    """
    fold_idx = []
    train = []
    test = []
    num_per_fold = len(np.unique(era)) // num_folds
    unique_era = np.unique(era)

    # Shuffle the eras to be split
    np.random.shuffle(unique_era)

    # Equally split the eras into num_folds
    unique = np.array_split(unique_era, num_folds)
    k = 0

    # Get the indices of each row within an era
    for i in unique:
        fold_idx.append(np.asarray([int(i) for i in era[era.isin(i)].index.tolist()]))
        k += 1

    # Append each index to the appropriate fold array
    for k in range(num_folds):
        test.append((np.asarray(fold_idx[k]).flatten()))
        if k == 0:
            train.append(np.concatenate(fold_idx[(k + 1):]).flatten())
        elif k == num_folds - 1:
            train.append(np.concatenate(fold_idx[:k]).flatten())
        else:
            train.append(np.concatenate(
                (np.concatenate(fold_idx[:(k)]).flatten(), np.concatenate(fold_idx[(k + 1):]).flatten())).flatten())

    return train, test


def payout(scores):
    """
    The hypothetical payout from Numerai given the Spearman Rank Correlation. (unused)
    """
    return ((scores - 0) / .2).clip(lower=-1, upper=1)


def timer(futures):
    """
    Inputs: futures (list) A list of Dask futures

    Outputs: None

    Takes in a list of Dask Futures objects, continually
    prints time elasped every 5 seconds, while printing out the
    finished, error, and pending Dask futures.
    """
    i = 0
    while (True):
        complete = 0
        error = 0
        pending = 0
        for f in futures:
            if f.status == 'finished':
                complete += 1
            elif f.status == 'error':
                error += 1
            else:
                pending += 1
        print(complete, error, pending)
        if pending == 0:
            break
        print(i * 5)
        i += 1
        time.sleep(5)
    return

def fit_predict(model, x_train, y_train, x_test, y_test, eras):
    """A helper function used in for submitting tasks to the cluster to fit, predict, and score the given model. Returns
    the Spearman Rank Correlation and Quartic Mean Error."""

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    s = spearman_rank(y_test, pred, eras)
    qme = quartic_error(y_test, pred)
    return s, qme

def LHS_RandomizedSearch(num_samples, params):
    """
    Inputs: num_samples (int) The number of samples to be taken from the ParameterGrid
             params (dict) A dictionary containing the hyperparameters and their potential values
    Output: parameters (list) A list containing num_samples number of different hyperparameter combinations

    This function takes in a dictionary of hyperparameter keys and potential values as values, then performs
    Latin Hypercube Sampling which is a more efficient random sampling method for hyperparameter tuning by
    only sampling positions in the ParameterGrid where only one sample exists in each axis-alligned hyperplane.
    """
    new = []
    # Uses the lhsmdu package found here: https://github.com/sahilm89/lhsmdu.
    # Creates array sized num_samples of Latin-Hypercube samples indices
    x = np.array(lhsmdu.sample(len(params[0].keys()), num_samples))

    # Form parameter grid to be sampled from usin the params dictionary
    parameter_grid = np.array(ParameterGrid(params))
    t = np.array([len(p) for p in params[0].values()])
    for i in range(len(t)):
        if t[i] == 1:
            new.append(np.zeros(len(x[i])))
        else:
            new.append(np.floor(x[i] * t[i]))
    idx = np.array(new).astype(int)
    k = (parameter_grid.reshape([len(p) for p in params[0].values()]))
    parameters = []

    # Append parameter samples to list
    for l in range(num_samples):
        j = 0
        for i in (idx[:, l]):

            if j == 0:
                ph = k[i]

            else:

                ph = ph[i]
            j += 1
        parameters.append(ph)
    return parameters
