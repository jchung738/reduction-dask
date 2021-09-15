import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hac
from sklearn.metrics import silhouette_samples
import hdbscan
from sklearn.cluster import AgglomerativeClustering



def pairwise(x, dist_fn, client):
    """
    Input: x (numpy array) A numpy array containing transposed X training data
           dist_fn (distance function) Any function used to calculate a
           distance metric for clustering
           client (Dask object) Used to submit jobs to the remote cluster

    Output: dm (numpy array) The 2-d distance matrix containing the distances
            for each feature
    This function utilizes Dask to parallelize the distance computation for feature clustering.
    """

    distance_matrix = np.zeros((x.shape[0], x.shape[0]), dtype=object).tolist()
    k = 0
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                d = client.submit(dist_fn, x[i], x[j])
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
    dm = client.gather(distance_matrix)

    return dm


def get_optimal_clusters(distance_matrix, type):
    """
    Input: distance_matrix (2d numpy array) A 2-d distance matrix of the features of the X dataset
           type (string) The type of clustering to be used (singlelink: Single-linkage Agglomerative Clustering,
           kmedoids: K-Medoids, hdbscan: Hierarchical Density-Based Spatial Clustering and Application with Noise
    Output: clustDict (dictionary) A cluster label dictionary of the optimal cluster size based on the silhouette score

    This function will take in a 2-d feature distance matrix, test all possible # of clusters (for K-Medoids), distance
    thresholds (for Agglomerative clustering), and minimum cluster sizes (for HDBSCAN). The optimal parameter is chosen
    by testing the silhouette score for each parameter. This function will also print a plot of the silhouette scores with
    the parameter of choice.
    """
    square_matrix = distance_matrix

    if type == 'singlelink':
        link = hac.linkage(distance_matrix, 'single')
        distance_threshold = link[:, 2][1:]
        t_scores = np.array([])
        # Test all distance thresholds for hierarchical clustering
        for i in distance_threshold:
            aClustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single',
                                                  distance_threshold=i)
            labels = aClustering.fit_predict(square_matrix)
            sil_samples = silhouette_samples(square_matrix, labels, metric='precomputed')
            t_scores = np.append(t_scores, (np.mean(sil_samples) / np.var(sil_samples)))
        aClust_t_score_vals = pd.DataFrame({'Silhouette Score': t_scores, 'Distance Threshold': distance_threshold})
        aClust_t_score_vals.plot(x='Distance Threshold', y='Silhouette Score',
                                 title='Agglomerative Clustering Silhouette Scores')
        distOpt = aClust_t_score_vals.iloc[aClust_t_score_vals['Silhouette Score'].idxmax(), 1]

        clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single',
                                           distance_threshold=distOpt)
        clustLabels = clusters.fit_predict(square_matrix)
        names = range(distance_matrix.shape[0])
        clustNames = zip(clustLabels, names)
        clustDict = dict()
        for c, n in clustNames:
            if c not in clustDict.keys():
                clustDict[c] = [n]
            else:
                clustDict[c].append(n)

        return clustDict

    elif type == 'kmedoids':

        num_clusters = range(2, squareform(distance_matrix).shape[0])
        t_scores = np.array([])
        # Test all possible number of clusters
        for i in num_clusters:
            kmedoids = KMedoids(n_clusters=i, metric='precomputed', init='k-medoids++')
            labels = kmedoids.fit_predict(square_matrix)
            sil_samples = silhouette_samples(square_matrix, labels)
            t_scores = np.append(t_scores, (np.mean(sil_samples) / np.var(sil_samples)))
        kMeds_t_score_vals = pd.DataFrame({'Silhouette Score': t_scores, 'Number of Clusters': num_clusters})
        kMeds_t_score_vals.plot(x='Number of Clusters', y='Silhouette Score', title='K-Medoids Silhouette Scores')
        clustOpt = kMeds_t_score_vals.iloc[kMeds_t_score_vals['Silhouette Score'].idxmax(), 1]

        clusters = KMedoids(n_clusters=clustOpt, metric='precomputed', init='k-medoids++')
        clustLabels = clusters.fit_predict(square_matrix)
        names = range(distance_matrix.shape[0])
        clustNames = zip(clustLabels, names)
        clustDict = dict()
        for c, n in clustNames:
            if c not in clustDict.keys():
                clustDict[c] = [n]
            else:
                clustDict[c].append(n)

    elif type == 'hdbscan':
        clust_size = range(2,  squareform(distance_matrix).shape[0])
        t_scores = np.array([])
        # Test all minimum cluster sizes. If HDBSCAN returns only a single label, the loop is broken to avoid later errors.
        for c in clust_size:
            dbscan = hdbscan.HDBSCAN(min_cluster_size=c, metric='precomputed', n_jobs=-1)
            labels = dbscan.fit_predict(square_matrix)
            if len(np.unique(labels)) == 1:
                break
            sil_samples = silhouette_samples(square_matrix, labels)
            t_scores = np.append(t_scores, (np.mean(sil_samples) / np.var(sil_samples)))
        dbscan_t_score_vals = pd.DataFrame({'Silhouette Score': t_scores, 'Min Cluster Size': clust_size})
        dbscan_t_score_vals.plot(x='Min Cluster Size', y='Silhouette Score', title='HDBSCAN')
        clustOpt = dbscan_t_score_vals.iloc[dbscan_t_score_vals['Silhouette Score'].idxmax(), 1]
        clusters = hdbscan.HDBSCAN(min_cluster_size=int(clustOpt), metric='precomputed', n_jobs=-1)
        clustLabels = clusters.fit_predict(square_matrix)
        names = range(distance_matrix.shape[0])
        clustNames = zip(clustLabels, names)
        clustDict = dict()
        for c, n in clustNames:
            if c not in clustDict.keys():
                clustDict[c] = [n]
            else:
                clustDict[c].append(n)
    return clustDict


