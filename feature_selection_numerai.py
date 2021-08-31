# Importing depedencies
import pandas as pd
import numpy as np
from metrics import spearman_rank
from utils import timer
from utils import kfold_era
from metrics import fit_predict
import ast
import re
import shap


def mean_decrease_accuracy(model, train_x, train_y, eras, num_folds, client, workers, features=None):
    """
    Inputs: model (Scikit-Learn Model object) The model to be used to rank features
            trainx (2d array) The X matrix for training the model
            trainy (1d array) The y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            num_folds (int) Number of folds used for cross-validation of each shuffled feature
            client (Dask object) Used to submit jobs to the remote cluster
            workers (list) List of worker ids given by the Dask cluster
            features (dict) An optional parameter which can take in a dictionary of clusters of features used in
            feature clustering
    Output: results (DataFrame) A DataFrame that is sorted by the MDA score for each feature or cluster of features

    MDA is a feature-importance algorithm which uses out-of-sample performance to rank each feature using predictive-importance.
    The predictive performance is found by taking the difference in score of a baseline model trained on the trainx and trainy data
    and the score of a model that is trained on trainx data with shuffled single or cluster of features. Dask is used to parallelize
    training of each shuffled feature (or cluster of features).
    Credit of MDA: Marcos Lopez De Prado, Advances in Financial Machine Learning
    """

    spearm = dict()
    # Get names of each feature for later

    names = pd.DataFrame(train_x).columns.values
    # Get indices for each k-fold
    kftrain, kftest = kfold_era(num_folds, eras)
    num_workers = len(workers)

    models = []
    val_indices = []

    # predict_score returns the Spearman Rank Correlation of the model prediction. Used when submitting tasks to Dask
    # cluster.
    def predict_score(model, x_test, y_test, era):
        pred = model.predict(x_test)
        s = spearman_rank(y_test, pred, era)
        return s

    # predict_mean returns the average Spearman Rank Correlation on all validation folds, where the models
    # are predicting with a single shuffled feature or a cluster of shuffled features.
    def predict_mean(model, x_test, y_test, acc, era):
        def perc(acc, shuff):
            return [(acc - shuff) / (1 - shuff)]

        scores = []
        i = 0
        for t in x_test:
            pred = model[i].predict(t.result())
            s = spearman_rank(y_test[i], pred, era[i])

            scores.append(perc(acc[i], s))

            i += 1
        return np.mean(scores)

    worker_idx = 0
    x_s = []
    y_s = []
    e = []

    # Scatter each training fold to the Dask cluster, as well as build val_indices and e array for validation
    for train_idx, test_idx in zip(kftrain, kftest):
        x_train = train_x[train_idx]
        y_train = train_y[train_idx]
        x_s.append(client.scatter(x_train, workers=workers[int(worker_idx % num_workers)].tolist(), broadcast=True))
        y_s.append(client.scatter(y_train, workers=workers[int(worker_idx % num_workers)].tolist(), broadcast=True))
        val_indices.append(test_idx)
        e.append(eras.iloc[test_idx])
        worker_idx += 1
    worker_idx = 0

    # Submit training for each model to Dask cluster using each fold from kfold_era
    for i in range(len(x_s)):
        models.append(client.submit(model.fit, x_s[i], y_s[i], workers=workers[int(worker_idx % num_workers)].tolist()))
        worker_idx += 1
    # Prints the current job status for each future
    timer(models)

    # Gather futures when all are finished
    models = client.gather(models, direct=True)
    i = 0
    accs = []

    # Scatter validation folds and submit tasks to Dask to get the base score for each fold. These scores are later
    # used when calculating the percentage change in performance from non-shuffled data to shuffled data.
    while i < len(models):
        val_x = client.scatter(train_x[val_indices[i]], workers=workers[int(i % num_workers)])
        val_y = client.scatter(train_y[val_indices[i]], workers=workers[int(i % num_workers)])
        era = client.scatter(e[i], workers=workers[int(i % num_workers)])
        m = models[i]
        accs.append(client.submit(predict_score, m, val_x, val_y, era, workers=workers[int(i % num_workers)]))
        i += 1

    # Gather futures done with computation of scoring each base validation fold.
    accs = client.gather(accs, direct=True)

    i = 0
    X_ts = []
    vals = []
    Y_ts = []

    # If there are no feature clusters inputted, then a 2d array is formed where each feature is its own cluster.
    # If feature clusters are inputted as a parameter, the values are taken from the dictionary and used for later
    # shuffling.
    if features is not None:
        feature_clusters = features.values()
    else:
        feature_clusters = np.array_split(names, train_x.shape[1])

    # For each validation fold, shuffle each feature or cluster of features (given by features parameter) and scatter
    # folds to the cluster.
    for t in val_indices:
        futs_x = []
        for v in feature_clusters:
            X_t = train_x[t].copy()
            for col in v:
                np.random.shuffle(X_t[:, col])
            futs_x.append(X_t)

        Y_ts.append(client.scatter(train_y[t], broadcast=True))
        X_ts.append(client.scatter(futs_x))
        i += 1
        vals.append(v)
    # For each feature cluster, submit all validation folds with said shuffled feature clusters using predict_mean as a
    # task.

    k = 0
    for v in feature_clusters:
        X_t = np.array([t[k] for t in X_ts])
        spearm[str(names[v])] = (
            client.submit(predict_mean, models, X_t, Y_ts, accs, e, workers=workers[int(k % num_workers)].tolist()))
        k += 1

    # Gather each feature cluster feature importance score
    spearm = client.gather(spearm, direct=True)

    # Returns each feature cluster and their score as a sorted DataFrame.
    print("Features sorted by their score:")
    sc = (sorted([(score, feat) for
                  feat, score in spearm.items()], reverse=True))
    results = pd.DataFrame(sc, columns=['Score', 'Feature'])
    return results


def mean_decrease_accuracy_tune(model, train_x, train_y, eras, feat_score, num_folds, n_components, client, workers,
                                is_clustered=False):
    """
    Inputs: model (Scikit-Learn Model object) The model that is used for the wrapper feature selection
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The Y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            feat_score (DataFrame) A DataFrame that contains each feature cluster with their corresponding MDA score (given from mean_decrease_accuracy)
            num_folds (int) Number of folds used for cross-validation of each shuffled feature
            n_components (int) The maximum number of features tried when tuning MDA using forward selection.
            client (Dask object) Used to submit jobs to the remote cluster
            num_workers (int) Number of workers used to run in the remote cluster
            is_clustered (boolean) Whether or not the feat_score uses single features (False) or feature clusters (True)

    Outputs: results (pandas DataFrame) A Dataframe containing the number of clusters tested, the average Spearman rank
    correlation and quartic mean error of the CV, sorted by the Spearman Rank Correlation.

    This function finds the optimal number of feature clusters from mean_decrease_accuracy using forward selection.
    Each subset of features is tested using kfold_era CV.
    """

    models = []
    num_workers = len(workers)
    # Get indices for cross-validation
    train, test = kfold_era(num_folds, eras)

    i = 0
    train_x_fut = []
    train_y_fut = []
    test_x_fut = []
    test_y_fut = []
    era_fut = []
    if n_components > len(feat_score):
        print('Cannot test more cluster combinations than number of clusters given in feat_score. Testing ' +
              str(len(feat_score)) + ' number of different cluster combinations')
        n_components = len(feat_score)

    # Forward selection of feature clusters ranked by mean_decrease_accuracy.
    for n in range(1, n_components):
        if is_clustered == False:
            idx = feat_score.sort_values('Score', ascending=False).iloc[:n, 1]
            rel_col = idx.apply(lambda x: int(x.strip('[]').split("' '")[0]))
            selected = train_x[:, rel_col]
        else:
            idx = (feat_score.sort_values('Score', ascending=False).iloc[:n, 1]).apply(
                lambda x: ast.literal_eval(re.sub(r'(?<=[\d])\s+(?=[\d])', ',', x)))
            rel_col = np.concatenate(idx.to_numpy())
            selected = train_x[:, rel_col]

        # Scatter each fold containing the features used in forward selection.
        for train_idx, test_idx in zip(train, test):
            x_train, x_test = selected[train_idx], selected[test_idx]
            y_train, y_test = train_y[train_idx], train_y[test_idx]
            e_test = eras.iloc[test_idx]

            x = client.scatter(x_train, workers=workers[int(i % num_workers)].tolist())
            y = client.scatter(y_train, workers=workers[int(i % num_workers)].tolist())
            x_t = client.scatter(x_test, workers=workers[int(i % num_workers)].tolist())
            y_t = client.scatter(y_test, workers=workers[int(i % num_workers)].tolist())
            e_t = client.scatter(e_test, workers=workers[int(i % num_workers)].tolist())

            train_x_fut.append(x)
            train_y_fut.append(y)
            test_x_fut.append(x_t)
            test_y_fut.append(y_t)
            era_fut.append(e_t)

            i += 1
    # Submit tasks for each subset of features from previous for-loop.
    for k in range(len(train_x_fut)):
        models.append(
            client.submit(fit_predict, model, train_x_fut[k], train_y_fut[k], test_x_fut[k], test_y_fut[k], era_fut[k],
                          workers=workers[int(k % num_workers)].tolist()))
    timer(models)
    # Gather futures once completed.
    scores = client.gather(models, direct=True)

    # Calculate mean of both Spearman rank correlation and Quartic mean error for each fold, each feature sub-selection
    # having their own k-folds.
    s = np.array([x[0] for x in scores])
    q = np.array([x[1] for x in scores])
    s = np.split(s, len(s) / num_folds)
    q = np.split(q, len(q) / num_folds)
    mean_s = np.mean(s, axis=1)
    mean_q = np.mean(q, axis=1)

    results = pd.DataFrame({'Number of Clusters': range(1, n_components),
                            'Spearman Rank Correlation': mean_s,
                            'Quartic Mean Error': mean_q})
    return results.sort_values('Spearman Rank Correlation', ascending=False)


def mean_decrease_accuracy_selector(train_x, feat_score, n_components, is_clustered=False):
    """
     Inputs: train_x (2d array) The X matrix for training the model
             feat_score (DataFrame) A DataFrame that contains each feature cluster with their corresponding MDA score (given from mean_decrease_accuracy)
             n_components (int) The number of feature clusters to be returned as the reduced X dataset
             is_clustered (boolean) Whether or not the feat_score uses single features (False) or feature clusters (True)

     Output: MDA_train_x (2d array) The X matrix after MDA feature selection

     This function will output the X matrix after MDA feature selection, using the optimal number of feature clusters given from
     mean_decrease_accuracy_tune.
     """
    rel_col = (feat_score.sort_values('Score', ascending=False).iloc[:n_components, :]).Feature
    if is_clustered == False:
        rel_col = rel_col.apply(lambda x: int(x.strip('[]').split("' '")[0]))
    else:
        rel_col = np.concatenate(rel_col.apply(lambda x: ast.literal_eval(re.sub(r'(?<=[\d])\s+(?=[\d])', ',', x))))

    MDA_train_x = train_x[:, rel_col]
    return MDA_train_x


def shapely_values(model, train_x, train_y):
    """
     Inputs: model (Scikit-Learn Model object) The model to be used to rank features
             trainx (2d array) The X matrix for training the model
     Output: results (DataFrame) A DataFrame that is sorted by the MDA score for each feature or cluster of features

     This function uses the Shapely values algorithm to rank the features by their game-theoretical contribution to the
     output of the model.
     Credit: https://github.com/slundberg/shap

    """
    shap.initjs()
    explainer = shap.TreeExplainer(model.fit(train_x, train_y), data=train_x[:1000])
    shap_values = explainer.shap_values(train_x[:1000])

    results = pd.DataFrame({'Feature': range(train_x.shape[1]), 'Score': np.abs(shap_values).mean(axis=0)}).sort_values(
        'Score', ascending=False)
    return results


def shap_tune(model, train_x, train_y, eras, feat_score, num_folds, n_components, client, workers):
    """
    Inputs: model (Scikit-Learn Model object) The model that is used for the wrapper feature selection
            train_x (2d array) The X matrix for training the model
            train_y (1d array) The Y array for training the model
            eras (1d array) The eras array which provide indices and eras for each row of the training data
            feat_score (DataFrame) A DataFrame that contains each feature cluster with their corresponding SHAP score
            (given from shapley_values)
            num_folds (int) Number of folds used for cross-validation of each shuffled feature
            n_components (int) The maximum number of features tried when tuning MDA using forward selection.
            client (Dask object) Used to submit jobs to the remote cluster
            num_workers (int) Number of workers used to run in the remote cluster

    Outputs: results (pandas DataFrame) A Dataframe containing the number of clusters tested, the average Spearman rank
    correlation and quartic mean error of the CV, sorted by the Spearman Rank Correlation.

    This function finds the optimal number of feature clusters from shapley_values using forward selection.
    Each subset of features is tested using kfold_era CV.
    """

    # A helper function that is used when submitting jobs to the Dask Cluster. Fits then scores the model on the
    # cluster.

    models = []
    num_workers = len(workers)
    # Get indices for cross-validation
    train, test = kfold_era(num_folds, eras)

    i = 0
    train_x_fut = []
    train_y_fut = []
    test_x_fut = []
    test_y_fut = []
    era_fut = []
    if n_components > len(feat_score):
        print('Cannot test more cluster combinations than number of clusters given in feat_score. Testing ' +
              str(len(feat_score)) + ' number of different cluster combinations')
        n_components = len(feat_score)

    # Forward selection of feature clusters ranked by shapley_values.
    for n in range(1, n_components):
        rel_col = feat_score.sort_values('Score', ascending=False).iloc[:n, 0]
        selected = train_x[:, rel_col]

        # Scatter each fold containing the features used in forward selection.
        for train_idx, test_idx in zip(train, test):
            x_train, x_test = selected[train_idx], selected[test_idx]
            y_train, y_test = train_y[train_idx], train_y[test_idx]
            e_test = eras.iloc[test_idx]

            x = client.scatter(x_train, workers=workers[int(i % num_workers)].tolist())
            y = client.scatter(y_train, workers=workers[int(i % num_workers)].tolist())
            x_t = client.scatter(x_test, workers=workers[int(i % num_workers)].tolist())
            y_t = client.scatter(y_test, workers=workers[int(i % num_workers)].tolist())
            e_t = client.scatter(e_test, workers=workers[int(i % num_workers)].tolist())

            train_x_fut.append(x)
            train_y_fut.append(y)
            test_x_fut.append(x_t)
            test_y_fut.append(y_t)
            era_fut.append(e_t)

            i += 1
    # Submit tasks for each subset of features from previous for-loop.
    for k in range(len(train_x_fut)):
        models.append(
            client.submit(fit_predict, model, train_x_fut[k], train_y_fut[k], test_x_fut[k], test_y_fut[k], era_fut[k],
                          workers=workers[int(k % num_workers)].tolist()))
    timer(models)
    # Gather futures once completed.
    scores = client.gather(models, direct=True)

    # Calculate mean of both Spearman rank correlation and Quartic mean error for each fold, each feature sub-selection
    # having their own k-folds.
    s = np.array([x[0] for x in scores])
    q = np.array([x[1] for x in scores])
    s = np.split(s, len(s) / num_folds)
    q = np.split(q, len(q) / num_folds)
    mean_s = np.mean(s, axis=1)
    mean_q = np.mean(q, axis=1)

    results = pd.DataFrame({'Number of Clusters': range(1, n_components),
                            'Spearman Rank Correlation': mean_s,
                            'Quartic Mean Error': mean_q})
    return results.sort_values('Spearman Rank Correlation', ascending=False)


def shap_selector(train_x, feat_score, n_components, is_clustered=False):
    """
     Inputs: train_x (2d array) The X matrix for training the model
             feat_score (DataFrame) A DataFrame that contains each feature cluster with their corresponding MDA score (given from shapley_values)
             n_components (int) The number of feature clusters to be returned as the reduced X dataset
             is_clustered (boolean) Whether or not the feat_score uses single features (False) or feature clusters (True)

     Output: SHAP_train_x (2d array) The X matrix after SHAP feature selection

     This function will output the X matrix after SHAP feature selection, using the optimal number of feature clusters given from
     mean_decrease_accuracy_tune.
     """
    rel_col = (feat_score.sort_values('Score', ascending=False).iloc[:n_components, :]).Feature
    SHAP_train_x = train_x[:, rel_col]
    return SHAP_train_x
