#!/usr/bin/env python3
"""This modlue preforms the k means on a data set
"""
import sklearn.cluster


def kmeans(X, k):
    """This functions preforms the k means on a data set
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
