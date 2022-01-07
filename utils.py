# Importing depedencies
import numpy as np
import lhsmdu
from sklearn.model_selection import ParameterGrid
import time


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


def fit_transform_dask(redux, train_x, num_fit_rows, num_splits, client):
    """
     Inputs: redux (sklearn object) A Dimensionality Reduction object from SKlearn
            train_x (2d array) The X matrix for training the model
            num_fit_rows (int) Number of rows used to fit the redux object
            num_splits (int) Number of times the data is split up to be transformed (smaller values lead to more
                             accurate transformations)
            client (Dask object) Used to submit jobs to the remote cluster


    Outputs: new_train_x (2d array) The transformed train_x dataset from the redux transform function.

    This function tunes the dimensionality reduction technique (redux) by first fitting the function
    with a subset of the training data then transforms the entire dataset by splitting the dataset by num_splits,
    transforming each split in parallel using Dask. Use this function when runtime becomes too long for transforming
    the dataset.
    """
    workers = np.array(list(client.get_worker_logs().keys()))
    num_workers = len(workers)
    train_subset = train_x[:num_fit_rows]
    redux.fit(train_subset)
    train_splits = len(train_x) // num_splits
    new_train_x = []
    for i in range(num_splits):
        if i == num_splits - 1:
            t_x = client.scatter(train_x[i * train_splits:], direct=True, workers=workers[i % num_workers])
            new_train_x.append(client.submit(redux.transform, t_x, workers=workers[i % num_workers]))

        else:
            t_x = client.scatter(train_x[i * train_splits:(i + 1) * train_splits], direct=True,
                                 workers=workers[i % num_workers])
            new_train_x.append(client.submit(redux.transform, t_x, workers=workers[i % num_workers]))

    new_train_x = client.gather(new_train_x, direct=True)

    return np.concatenate(new_train_x)
