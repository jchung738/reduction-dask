import pandas as pd
import numpy as np
from metrics import spearman_rank
from utils import timer
from utils import kfold_era
from utils import LHS_RandomizedSearch
from metrics import fit_predict
import scipy.stats as st


def tune_kfold_dask(model, train_x, train_y, eras, num_folds, params, num_samples, client, workers):
    """
    Inputs: model (sklearn Model Object) Any kind of sklearn model
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            num_folds (int) Number of folds used for cross-validation of each shuffled feature
            params (dict) A dictionary containing different hyper-parameters of the model as keys and ranges of
            potential hyper-parameter values used as values
            num_samples (int) number of samples to search for hyper-parameter tuning
            client (Dask object) Used to submit jobs to the remote cluster
            workers (list) List of worker ids given by the Dask cluster


    Output: final (DataFrame) A DataFrame of the hyper-parameter combinations tested ranked by their Spearman Rank Correlation

    This function uses Latin-Hypercube Sampling to search the hyper-parameter grid of any sklearn model and uses
    k-fold cross-validation (with respect to eras) to rank each combination by their Spearman rank correlation


    """

    val = pd.DataFrame()
    num_workers = len(workers)
    parameters = LHS_RandomizedSearch(num_samples, params)
    i = 0
    w = 0
    h = num_workers // num_folds
    kftrain, kftest = kfold_era(num_folds, eras)
    models = []
    test_indices = []
    train_indices = []
    x_s = []
    y_s = []
    x_t = []
    y_t = []
    e_t = []

    # Scatter each fold of data to each of the workers
    for train_idx, test_idx in zip(kftrain, kftest):
        x_train, x_test = train_x[train_idx], train_x[test_idx]
        y_train, y_test = train_y[train_idx], train_y[test_idx]
        work_id = workers[np.arange(int(w % num_workers), int(h + w % num_workers))].tolist()
        x_s.append(client.scatter(x_train, workers=work_id, broadcast=True))
        y_s.append(client.scatter(y_train, workers=work_id, broadcast=True))
        x_t.append(client.scatter(x_test, workers=work_id, broadcast=True))
        y_t.append(client.scatter(y_test, workers=work_id, broadcast=True))
        e_t.append(client.scatter(eras.iloc[test_idx], workers=work_id, broadcast=True))
        w += h
        test_indices.append(test_idx)
        train_indices.append(train_idx)
    print('Done Scattering')
    w = 0
    k = 0

    # For each hyper-parameter combination given from LHS random sampling, submit said model as a task to the cluster k
    # times, where k is the number of folds.
    for values in parameters:
        rf = model(**values)
        for f in range(num_folds):
            future = client.submit(fit_predict, rf, x_s[f], y_s[f], x_t[f], y_t[f], e_t[f],
                                   workers=workers[int(k % num_workers)].tolist())
            models.append(future)

            k += 1

        v = pd.DataFrame(values, index=[0])
        val = val.append(v, ignore_index=True)
        i += 1
        if i >= num_samples:
            break

    timer(models)
    results = client.gather(models, direct=True)

    s = [x[0] for x in results]
    q = [x[1] for x in results]
    scores = np.split(np.array(s), num_samples)
    qmes = np.split(np.array(q), num_samples)

    scores = np.mean(scores, axis=1)
    qmes = np.mean(qmes, axis=1)
    s = pd.DataFrame({'Spearman Rank Corr by ERA Mean': scores, 'Quartic Mean Error': qmes})
    final = pd.concat([val, s], axis=1)
    return final.sort_values(by='Spearman Rank Corr by ERA Mean',ascending=False)


def kfold_dask(model,train_x, train_y, eras, num_folds, client,  workers):
    """
    Inputs: model (sklearn Model Object) Any kind of sklearn model
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            num_folds (int) Number of folds for cross-validation
            client (Dask object) Used to submit jobs to the remote cluster
            workers (list) List of worker ids given by the Dask cluster

    Outputs: spear (tuple) A tuple containing the mean, 2.5% and 97.5% confidence intervals of the
                           Spearman Rank Correlation
             quart (tuple) A tuple containing the mean, 2.5% and 97.5% confidence intervals of the Quartic Mean Error

    This function performs K-fold cross-validation with respect to eras and returns the mean, 2.5% and 97.5% CI for both
    Spearman rank correlation and Quartic mean error.
    """
    train, test = kfold_era(num_folds, eras)
    models = []
    test_indices = []
    i = 0
    num_workers = len(workers)

    # Scatter each fold then submit task to cluster to train and score each fold for the model.
    for train_idx, test_idx in zip(train, test):
        x_train, x_test = train_x[train_idx], train_x[test_idx]
        y_train, y_test = train_y[train_idx], train_y[test_idx]
        e_test = eras.iloc[test_idx]
        x = client.scatter(x_train, workers=workers[int(i % num_workers)].tolist(), direct=True)
        y = client.scatter(y_train, workers=workers[int(i % num_workers)].tolist(), direct=True)
        x_t = client.scatter(x_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
        y_t = client.scatter(y_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
        e_t = client.scatter(e_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
        models.append(
            client.submit(fit_predict, model, x, y, x_t, y_t, e_t, workers=workers[int(i % num_workers)].tolist()))
        i += 1
        test_indices.append(test_idx)
    timer(models)

    # Gather all the completed futures
    scores = client.gather(models, direct=True)
    s = [x[0] for x in scores]
    q = [x[1] for x in scores]
    spear = (np.mean(s), st.t.interval(0.95, len(s) - 1, loc=np.mean(s), scale=st.sem(s)))
    quart = (np.mean(q), st.t.interval(0.95, len(q) - 1, loc=np.mean(q), scale=st.sem(q)))
    return spear, quart


def tune_reduction_dask(redux, model, train_x, train_y, eras, num_folds, params, num_samples, client, workers):
    """
    Inputs: redux (sklearn object) A Dimensionality Reduction object from SKlearn
            model (sklearn Model Object) Any kind of sklearn model
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            num_folds (int) Number of folds for cross-validation
            params (dict) A dictionary containing different hyperparamers of the redux function as keys and
            ranges of potential hyperparameter values used as values
            num_samples (int) Total number of hyperparameter combinations to search through
            client (Dask object) Used to submit jobs to the remote cluster
            workers (list) List of worker ids given by the Dask cluster


    Output: final (DataFrame) A DataFrame of the hyper-parameter combinations tested ranked by their Spearman Rank Correlation


    This function performs K-fold cross-validation with respect to eras and returns the mean, 2.5% and 97.5% CI for both
    Spearman rank correlation and Quartic mean error.
    """
    reductions = []
    val = pd.DataFrame()
    models = []
    num_workers = len(workers)
    parameters = LHS_RandomizedSearch(num_samples, params)
    kftrain, kftest = kfold_era(num_folds, eras)

    i = 0
    train_x = client.scatter(train_x, broadcast=True)

    def fit_predict(model, x_train, y_train, x_test, y_test, eras):
        model.fit(x_train, y_train)
        s = spearman_rank(y_test, model.predict(x_test), eras)
        return s

    # Transform the train_x array with each hyper-parameter combination used for the dimensionality reduction function
    for values in parameters:
        rf = redux(**values)
        fut = client.submit(rf.fit_transform, train_x, workers=workers[i % num_workers].tolist())
        reductions.append(fut)
        v = pd.DataFrame(values, index=[0])
        val = val.append(v, ignore_index=True)
        i += 1
        if i >= num_samples:
            break

    timer(reductions)
    print('gathered')
    reductions = client.gather(reductions, direct=True)

    # Scatter each fold and submit tasks for each fold to be trained and scored.
    i = 0
    for r in reductions:
        for train_idx, test_idx in zip(kftrain, kftest):
            x_train, x_test = r[train_idx], r[test_idx]
            y_train, y_test = train_y[train_idx], train_y[test_idx]
            e_test = eras.iloc[test_idx]
            x = client.scatter(x_train, workers=workers[int(i % num_workers)].tolist(), direct=True)
            y = client.scatter(y_train, workers=workers[int(i % num_workers)].tolist(), direct=True)
            x_t = client.scatter(x_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
            y_t = client.scatter(y_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
            e_t = client.scatter(e_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
            models.append(
                client.submit(fit_predict, model, x, y, x_t, y_t, e_t, workers=workers[int(i % num_workers)].tolist()))
            i += 1

    timer(models)
    gathered = client.gather(models, direct=True)
    models = np.split(np.array(gathered), num_samples)
    rf_means = np.mean(models, axis=1)
    s = pd.DataFrame({'Spearman Rank Corr by ERA Mean': rf_means})
    final = pd.concat([val, s], axis=1)
    return final.sort_values(by='Spearman Rank Corr by ERA Mean',ascending=False)


def tune_reduction_transform_dask(redux, model, train_x, train_y, eras, num_folds, params, num_samples, num_fit_rows,
                                  num_splits, client, workers):
    """
    Inputs: redux (sklearn object) A Dimensionality Reduction object from SKlearn
            model (sklearn Model Object) Any kind of sklearn model
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            num_folds (int) Number of folds for cross-validation
            params (dict) A dictionary containing different hyperparamers of the redux function as keys and
                          ranges of potential hyperparameter values used as values
            num_samples (int) Total number of hyperparameter combinations to search through
            num_fit_rows (int) Number of rows used to fit the redux object
            num_splits (int) Number of times the data is split up to be transformed (smaller values lead to more
                             accurate transformations)
            client (Dask object) Used to submit jobs to the remote cluster
            workers (list) List of worker ids given by the Dask cluster

    Outputs: final (DataFrame) A DataFrame that contains the average CV Spearman Rank Correlation of each set of
             hyperparameters

    This function tunes the dimensionality reduction technique (redux) by first fitting the function
    with a subset of the training data then transforms the entire dataset using the parameters given by the set of
    hyperparameters. This is done in case workers die due to memory constraints or to save on compute time.

    """
    reductions = []
    val = pd.DataFrame()
    models = []
    num_workers = len(workers)
    parameters = LHS_RandomizedSearch(num_samples, params)
    kftrain, kftest = kfold_era(num_folds, eras)

    i = 0
    train_x = client.scatter(train_x, broadcast=True)

    if train_y is not None:
        train_y = client.scatter(train_y, broadcast=True)

    def fit_predict(model, x_train, y_train, x_test, y_test, eras):
        model.fit(x_train, y_train)
        s = spearman_rank(y_test, model.predict(x_test), eras)
        return s

    def fit_transform(reduction, x_train):
        try:
            redux = reduction.fit(x_train[:num_fit_rows])
        except (ValueError, TypeError):
            return
        train_splits = len(x_train) // num_splits
        for i in range(num_splits):
            if i == 0:
                train_x = redux.transform(x_train[:train_splits])
            elif i < num_splits - 1:
                train_x = np.append(train_x, redux.transform(x_train[train_splits * i:train_splits * (i + 1)]), axis=0)
            else:

                train_x = np.append(train_x, redux.transform(x_train[train_splits * i:]), axis=0)

        return train_x

    for values in parameters:
        rf = redux(**values)
        fut = client.submit(fit_transform, rf, train_x, workers=workers[i % num_workers].tolist())
        reductions.append(fut)

        v = pd.DataFrame(values, index=[0])
        val = val.append(v, ignore_index=True)
        i += 1
        if i >= num_samples:
            break
    timer(reductions)
    print('gathered')
    reductions = client.gather(reductions, direct=True)
    train_y = client.gather(train_y)
    i = 0
    k = 0
    n_samples = 0
    drop = []
    for r in reductions:
        if r is None:
            drop.append(k)
            k += 1
            continue
        if np.isnan(np.sum(r)):
            drop.append(k)
            k += 1
            continue
        for train_idx, test_idx in zip(kftrain, kftest):
            x_train, x_test = r[train_idx], r[test_idx]
            y_train, y_test = train_y[train_idx], train_y[test_idx]
            e_test = eras.iloc[test_idx]
            x = client.scatter(x_train, workers=workers[int(i % num_workers)].tolist(), direct=True)
            y = client.scatter(y_train, workers=workers[int(i % num_workers)].tolist(), direct=True)
            x_t = client.scatter(x_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
            y_t = client.scatter(y_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
            e_t = client.scatter(e_test, workers=workers[int(i % num_workers)].tolist(), direct=True)
            models.append(
                client.submit(fit_predict, model, x, y, x_t, y_t, e_t, workers=workers[int(i % num_workers)].tolist()))
            i += 1
        k += 1
        n_samples += 1
    val = val.drop(drop, axis=0)
    val.reset_index(inplace=True)
    timer(models)
    gathered = client.gather(models, direct=True)
    models = np.split(np.array(gathered), n_samples)
    rf_means = np.mean(models, axis=1)
    s = pd.DataFrame({'Spearman Rank Corr by ERA Mean': rf_means})
    final = pd.concat([val, s], axis=1)
    return final.sort_values('Spearman Rank Corr by ERA Mean',ascending=False)


def hyperband(model, train_x, train_y, eras, num_folds, params, samples, eta, max_ratio, client, workers):
    """
    Inputs: model (sklearn Model Object) Any kind of sklearn model
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            num_folds (int) Number of folds for cross-validation
            params (dict) A dictionary containing different hyperparamers of the model as keys and ranges of
            potential hyperparameter values used as values
            samples (int) Total number of hyperparameter combinations to search through
            eta (float) Downsampling rate used for successive halving
            max_ratio (int) The maximum ratio of data to be used in training of models
            client (Dask object) Used to submit jobs to the remote cluster
            workers (list) List of worker ids given by the Dask cluster

    Outputs: best_score (float) Spearman Rank Correlation of the top performing hyperparameter combination
             best_param (dict) Best performing hyperparameters

    Use the hyperband algorithm to hyperparameter tune any model. This version uses Dask to parallelize model training
    for each round of successive halving.
    """

    num_workers = len(workers)
    parameters = LHS_RandomizedSearch(samples, params)
    w = 0
    h = num_workers // num_folds
    kftrain, kftest = kfold_era(num_folds, eras)
    test_indices = []
    train_indices = []
    x_s = []
    y_s = []
    x_t = []
    y_t = []
    e_t = []
    for train_idx, test_idx in zip(kftrain, kftest):
        x_train, x_test = train_x[train_idx], train_x[test_idx]
        y_train, y_test = train_y[train_idx], train_y[test_idx]
        work_id = workers[np.arange(int(w % num_workers), int(h + w % num_workers))].tolist()
        x_s.append(client.scatter(x_train, workers=work_id, broadcast=True))
        y_s.append(client.scatter(y_train, workers=work_id, broadcast=True))
        x_t.append(client.scatter(x_test, workers=work_id, broadcast=True))
        y_t.append(client.scatter(y_test, workers=work_id, broadcast=True))
        e_t.append(client.scatter(eras.iloc[test_idx], workers=work_id, broadcast=True))
        w += h
        test_indices.append(test_idx)
        train_indices.append(train_idx)
    print('Done Scattering')

    def score(Y_True, Y_Pred, era):
        """
        Inputs: Y_True (array-like) True targets
                Y_Pred (DataFrame) Predicted targets (should have an era column)
        """
        # Rank-correlation by era
        Y_Pred = pd.DataFrame(Y_Pred, index=Y_True.index)
        Y_Pred = Y_Pred.join(era).dropna()
        ranked_pred = Y_Pred.groupby('era').apply(lambda x: x.rank(pct=True, method="first")).values[:, 0]
        x = np.corrcoef(np.ravel(Y_True), ranked_pred)[0, 1]

        return x

    def fit_predict(model, x_train, y_train, x_test, y_test, era, ratio):

        model.fit(x_train[:np.int(np.ceil(ratio * len(x_train) / 100))],
                  y_train[:np.int(np.ceil(ratio * len(x_train) / 100))])
        pred = model.predict(x_test)
        s = score(y_test, pred, era)
        return s

    best_score = -1e8
    best_param = 0

    counter = 0
    log_eta = lambda x: np.log(x) / np.log(eta)
    s_max = int(log_eta(max_ratio))
    B = (s_max + 1) * max_ratio
    for s in reversed(range(s_max + 1)):

        # initial number of configurations
        n = int(np.ceil(B / max_ratio / (s + 1) * eta ** s))

        # initial number of iterations per config
        r = max_ratio * eta ** (-s)

        # n random configurations
        if len(parameters) < n:
            break
        T = parameters[:n]
        parameters = parameters[n:]
        k = 0
        for i in range((s + 1)):  # changed from s + 1

            # Run each of the n configs for <iterations>
            # and keep best (n_configs / eta) configurations

            n_configs = n * eta ** (-i)
            ratio = r * eta ** (i)

            print("\n*** {} configurations x {:.1f} ratio".format(n_configs, ratio))

            models = []

            for t in T:

                rf = model(**t)

                for f in range(num_folds):
                    future = client.submit(fit_predict, rf, x_s[f], y_s[f], x_t[f], y_t[f], e_t[f], ratio,
                                           workers=workers[int(k % num_workers)].tolist())
                    models.append(future)
                    k += 1

            timer(models)
            futures = client.gather(models, direct=True)
            models = np.split(np.array(futures), len(T))
            results = np.mean(models, axis=1)
            param = T[np.argmax(results)]
            score = np.max(results)
            # keeping track of the best result so far (for display only)
            # could do it be checking results each time, but hey
            if best_score < score:
                best_score = score
                best_param = param
            print("\n{} | current score: {} | best score so far: {:.4f} (parameters {})\n".format(counter, score,
                                                                                                  best_score,
                                                                                                  best_param))

            # select a number of best configurations for the next loop
            # filter out early stops, if any
            indices = np.argsort(results)
            T = [T[i] for i in indices]
            T = T[-int(n_configs / eta):]

    return best_score, best_param
