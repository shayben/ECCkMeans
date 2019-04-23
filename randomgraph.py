import networkx as nx
import itertools
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from math import sqrt, log
import math, random
from copy import deepcopy
import time
from datetime import datetime


def GenerateGraph(proba, bigleavessizes):
    n = sum(bigleavessizes)
    G = nx.Graph()
    labels = []
    i = 0
    for lab in range(len(bigleavessizes)):
        for u in range(bigleavessizes[lab]):
            labels.append(lab)
            # print(lab)
            i += 1
    count = 0

    permuted = np.random.permutation(n)
    cpylabels = labels[:]
    labels = [cpylabels[permuted[x] - 1] for x in range(n)]

    for u, v in list(itertools.combinations_with_replacement(range(n), r=2)):
        clustu = labels[u]
        clustv = labels[v]
        coin = np.random.random()
        if coin < proba[clustu][clustv]:
            count = count + 1
            G.add_edge(u, v, weight=1.0)

    print("Number of edges:", count)
    return G, labels


def binarylimitsspecial(n, k, T):
    '''
    Compare the ECC basic approach to PCA on the stochastic block model with k equal-size clusters
    :param n: number of nodes
    :param k: number of clusters --> each cluster has size n/k
    :param T: length of the simple code
    :return: 
    '''

    print("PARAMS", n, k)
    global p, q
    Matrix = np.tile(q, [k, k])
    np.fill_diagonal(Matrix, p)

    bigleavessizes = [int(n / k)] * k

    print("Probability Matrix:")
    print(np.matrix(Matrix))

    G, labels = GenerateGraph(Matrix, bigleavessizes)
    labels = np.hstack([[i] * (int(n / k)) for i in range(k)])
    # print(labels)
    print("Generation done.")

    A = nx.adjacency_matrix(G)
    X = A.todense()

    np.random.seed(int(time.time()))
    random_subsets = np.random.randint(0, 2, (n, T))
    parity_bits = np.mod(A * random_subsets, 2)

    ### Error Correcting Code approach
    reduced_data = np.hstack([X, parity_bits])
    kmeans_clusters_alg = kmeans(reduced_data, k)

    ### Classic PCA approach
    reduced_data2 = X
    kmeans_clusters_standard = kmeans(reduced_data2, k)

    print("ALG: Accuracy=", metrics.classification.accuracy_score(labels, kmeans_clusters_alg))
    print(metrics.classification.classification_report(labels, kmeans_clusters_alg))
    print(metrics.confusion_matrix(labels, kmeans_clusters_alg))

    print("PCA: Accuracy=", metrics.classification.accuracy_score(labels, kmeans_clusters_standard))
    print(metrics.classification.classification_report(labels, kmeans_clusters_standard))
    print(metrics.confusion_matrix(labels, kmeans_clusters_standard))



def classif_error(labels):
    nb_correct_class = 0
    # for
    n = len(labels)
    for i in range(len(labels)):
        if i < n / 4 and labels[i] == 0:
            nb_correct_class += 1
        if i >= n / 2 and labels[i] == 1:
            nb_correct_class += 1
    return (nb_correct_class / n)


def check_labels(clustering, labels):
    nb_correct_class = 0
    for c in range(len(clustering)):
        if clustering[c] - 1 == labels[c]:
            nb_correct_class += 1
    return nb_correct_class


def kmeans(X, k):
    alg = KMeans(init='k-means++', n_clusters=k, n_init=10)
    alg.fit(X)
    #kmeans_clusters = {c: [] for c in range(k)}
    #for i in range(len(X)):
    #    kmeans_clusters[alg.labels_[i]].append(i)
    return alg.labels_


def check_clusters(clusters, labels):
    predict = [0] * len(labels)
    bad_guys = []

    nb_correctly_classified = 0
    for c in clusters:
        if clusters[c] == []: continue
        nbrs = {}
        for p in clusters[c]:
            if labels[p] in nbrs:
                nbrs[labels[p]] += 1
            else:
                nbrs[labels[p]] = 1
        M = 0
        digit = -1
        for n in nbrs:
            if nbrs[n] > M:
                digit = n
                M = nbrs[n]

        for p in clusters[c]:
            if labels[p] != digit:
                bad_guys.append(p)

        nb_correctly_classified += nbrs[digit]
        # if digit == -1:
        #     print(clusters[c], nbrs)
        print("Cluster", c, "represents digit", digit)
        for p in clusters[c]:
            predict[p] = digit
        print("           Classification accuracy = ",
              100 * nbrs[digit] / len(clusters[c]), "%.")
    print("Overall accuracy :",
          nb_correctly_classified / len(labels))
    return bad_guys


def digits(T=64, plotresult=False):
    '''
    compares the basic ECC approach to PCA on the classic digits dataset of scikit-learn
    (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)
    :param T:
    :param plotresult:
    :return:
    '''
    from sklearn import datasets
    digits = datasets.load_digits()
    labels = digits.target

    ### Classic PCA
    X = digits.data
    X_reduced = PCA(n_components=2).fit_transform(X)
    kmeans_clusters = kmeans(X_reduced, 10)
    print("Kmean classification:")
    check_clusters(kmeans_clusters, labels)

    ### Error Correcting Code approach
    n = len(labels)
    random.seed(datetime.now())
    vects = []
    for step in range(T):
        v = []
        for c in range(n):
            a = random.uniform(0, 1)
            if a == 1:
                v.append(1)
            else:
                v.append(0)
        vects.append(v)

    Y_large = digits.data
    Y_reduced = PCA(n_components=2).fit_transform(Y_large)

    Y_new = []
    for row in Y_reduced:
        # print(row)
        a = deepcopy(row)
        # print(row)
        for j in range(T):
            s = sum([row[i] for i in range(len(row))
                     if vects[j][i] == 1])
            a = a + [s % 199]
        Y_new.append(a)

    Y_new_reduced = Y_new
    kmeans_clusters_alg = kmeans(Y_new_reduced, 10)
    print("New Alg classification:")
    bad_guys = check_clusters(kmeans_clusters_alg, labels)

    ### PLOT Result
    ###
    if not plotresult: return
    import matplotlib.pyplot as plt
    colors = ['g', 'b', 'c', 'm', 'y', 'k']

    print(len(labels))
    for gc in kmeans_clusters_alg:
        for x in kmeans_clusters_alg[gc]:
            plt.plot(Y_new_reduced[x][0],
                     Y_new_reduced[x][1], colors[gc % 6] + 'o')

    for x in bad_guys:
        plt.plot(Y_new_reduced[x][0], Y_new_reduced[x][1],
                 "ro")

    plt.axis([-50, 50, -50, 50])
    plt.show()


def wrapper(n):
    runs = 1
    global p, q
    p = 1 / 10  # log(n)/sqrt(n)
    q = 1 / 200  # log(n)/(10*sqrt(n))
    for step in range(runs):
        binarylimitsspecial(n, 4, 10 * int(n))


### SBM
wrapper(400)

### Digits
# digits()
