# Gap Statistic code
import numpy as np
import random
from sklearn.cluster import KMeans

def dispersion (data, k):
    if k == 1:
        # get mean along the cols
        cluster_mean = np.mean(data, axis=0)
        # get sum of squared difference along the rows
        distances_from_mean = np.sum((data - cluster_mean)**2,axis=1)
        # get log of sum of distances from mean
        dispersion_val = np.log(sum(distances_from_mean))
    else:
        k_means_model_ = KMeans(n_clusters=k, max_iter=50, n_init=5, random_state=4).fit(data)
        # initialize list
        distances_from_mean = range(k)
        for i in range(k):
            distances_from_mean[i] = int()
            for idx, label in enumerate(k_means_model_.labels_):
                # if data row belongs to cluster 
                if i == label:
                    # get distance fromt point to centroid 
                    distances_from_mean[i] += sum((data[idx] - k_means_model_.cluster_centers_[i])**2)
        # get log of sum of distances from mean
        dispersion_val = np.log(sum(distances_from_mean))

    return dispersion_val

def reference_dispersion(data, num_clusters, num_reference_bootstraps):
    dispersions = [dispersion(generate_uniform_points(data), num_clusters) for i in range(num_reference_bootstraps)]
    mean_dispersion = np.mean(dispersions)
    stddev_dispersion = float(np.std(dispersions)) * np.sqrt(1. + 1. / num_reference_bootstraps) 
    return mean_dispersion, stddev_dispersion

def generate_uniform_points(data):
    mins = np.argmin(data, axis=0)
    maxs = np.argmax(data, axis=0)

    num_dimensions = data.shape[1]
    num_datapoints = data.shape[0]

    # initalize list 
    reference_data_set = np.zeros((num_datapoints,num_dimensions))
    for i in range(num_datapoints):
        for j in range(num_dimensions):
            reference_data_set[i][j] = random.uniform(data[mins[j]][j], data[maxs[j]][j])

    return reference_data_set   

def gap_statistic (data, nthCluster, referenceDatasets):
    # get dispersion for nthCluster
    actual_dispersion = dispersion(data, nthCluster)
    # get reference dispersion for nthCluster
    ref_dispersion, stddev_dispersion = reference_dispersion(data, nthCluster, num_reference_bootstraps)
    return actual_dispersion, ref_dispersion, stddev_dispersion