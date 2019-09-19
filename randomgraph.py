"""Improving Classification via Checksum

Paper ID: 8976 submitted to NeurIPS2019
Shay Ben-Elazar - shaybenelazar@hotmail.com
Vincent Cohen-Addad - vcohenad@gmail.com
Karthik C. S. - karthik0112358@gmail.com (Primary)
Eylon Yogev - eylony@gmail.com

Abstract:
We consider the fundamental task of clustering a dataset.
The classic approach for this task is to perform dimensionality reduction
(via techniques for denoising the dataset such as PCA), and then classify using a fast clustering algorithm
such as k-means++. We propose a new approach to denoise data sets and improve classification
based on appending checksums, and is inspired by known constructions of error correcting codes.
We prove that when the data sets can be reasonably well classified, and have small noise and few outliers,
then our new approach performs better than the approaches known in literature.
We complement our results with an empirical analysis on both synthetic and real-world datasets.

"""
import time, datetime
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import generate_BX_graph as BX
import mushrooms as mush


# region data


def get_dataset(name, **kwargs):
    from sklearn.preprocessing import scale
    data = []

    if name == "SBM":
        X, labels = generate_block_stochastic_data(kwargs['n'], kwargs['k'], kwargs['p'], kwargs['q'])
        return X, X.shape[0], labels, len(set(labels))

    elif name == "mushrooms":
        X, labels = mush.ReadMushrooms()
        return X, X.shape[0], labels, len(set(labels))

    elif name == "cancer":
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()

    elif name == "digits":
        from sklearn.datasets import load_digits
        dataset = load_digits()

    elif name == "iris":
        from sklearn.datasets import load_iris
        dataset = load_iris()

    elif name == "boston":
        from sklearn.datasets import load_boston
        dataset = load_boston()
        data = dataset.data

    elif name == "KDD":
        from sklearn.datasets import fetch_kddcup99
        dataset = fetch_kddcup99(subset='None')
        data = dataset.data[:, 4:].astype(float)
        labmap = dict([(v, id) for id, v in enumerate(set(dataset.target))])
        print(labmap)
        dataset.target = [labmap[v] for v in dataset.target]

    else:
        print("Unknown name of dataset")
        exit(-1)

    labels = dataset.target
    if data == []:
        data = scale(dataset.data)
    n_samples, n_features = data.shape

    return data, n_samples, labels, len(set(labels))


def generate_block_stochastic_data(n, k, p, q):
    """
    Generates binary block stochastic community graph data
    :param n: number of nodes in graph
    :param k: number of clusters, n/k nodes per cluster
    :param p: pr of edge within cluster
    :param q: pr of edge between clusters
    :return: X adj. matrix, labels ground truth cluster ids
    """
    bigleavessizes = [int(n / k)] * k
    intra = sp.linalg.block_diag(
        *[distance.squareform(np.random.binomial(1, p, int(d * (d - 1) / 2))) for _, d in enumerate(bigleavessizes)])
    intra_mask = sp.linalg.block_diag(*[np.ones((d, d)) for _, d in enumerate(bigleavessizes)])
    intra_fin = np.multiply(intra, intra_mask)
    inter = distance.squareform(np.random.binomial(1, q, int(n * (n - 1) / 2)))
    inter_mask = np.ones((n, n)) - intra_mask
    inter_fin = np.multiply(inter, inter_mask)
    X = intra_fin + inter_fin
    labels = np.hstack([[i] * (int(n / k)) for i in range(k)])
    return X, labels


def get_digits():
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    digits = load_digits()
    data = scale(digits.data)
    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target
    return data, n_digits, labels


# endregion

# region algorithms

def correct_label_assignment(cluster_labels, true_labels):
    """
    Transforms cluster ids to ground truth class ids using the hungarian maximum matching algorithm
    :param cluster_labels:
    :param true_labels:
    :return: confusion matrix used for the mapping
    """
    conf_mat = metrics.confusion_matrix(true_labels, cluster_labels)
    # hungarian algorithm to assign optimal cluster ids to original labels
    _, idmap = linear_sum_assignment(1 / (1 + conf_mat))
    nbrs = dict([(id, v) for v, id in enumerate(idmap)])
    return np.asarray([nbrs[v] for v in cluster_labels]), conf_mat


def subsampled_kmeans(X, k, fraction, is_adj=True):
    """ runs kmeans on row-subsampled matrix
    :param X: n X m matrix
    :param k: k for kmeans
    :param fraction: fraction of n to sample
    :param is_adj: is adjacency matrix (n X n)
    :return: KMeans object and the indices of selected samples (rows)
    """
    n = X.shape[0]
    smpl = int(n * fraction)
    smplindices = np.random.randint(0, n, smpl)
    km = KMeans(init='k-means++', n_clusters=k, n_init=10)
    if is_adj:
        km.fit(X[np.ix_(smplindices, smplindices)])
    else:
        km.fit(X[smplindices, :])
    return km, smplindices


def compute_alpha_beta(k, p, q):
    """

    :param k: kmeans
    :param p: intra-cluster edge probability
    :param q: inter-cluster edge probability
    :return:
    """
    alpha = 1 / k * (2 * p ** 2 - 2 * p + 1) + (1 - 1 / k) * (2 * q ** 2 - 2 * q + 1)
    beta = 2 * p * q - p - q + 1
    return alpha, beta


def compute_t_D(n, alpha, beta):
    """
    :param n: number of points
    :param alpha: intra-cluster normalized Hamming similarity
    :param beta: inter-cluster normalized Hamming similarity
    :return: t_star - optimal subset sizes, D - number of added checksum coordinates
    """
    t_star = np.log(n * (beta - 1) * np.log(beta) / np.log(alpha)) / np.log(alpha / beta)
    D = 1 / np.power(alpha, t_star)
    return t_star, D


def kmeans_subsample_density_estimator(X, labels, sample_ratio=0.2):
    """
    performs a subsampled k-means, estimate p,q parameters and use these to determine t, D according to paper
    :param X:
    :param labels:
    :param sample_ratio:
    :return:
    """
    synthetic_data = np.array_equal(X, np.transpose(X))
    k = np.unique(labels).shape[0]
    n = X.shape[0]
    print('n = %d k = %d sample_ratio = %.3f' % (n, k, sample_ratio))
    km, idx = subsampled_kmeans(X, k, sample_ratio, synthetic_data)
    kmeans_clusters_alg, conf_mat = correct_label_assignment(km.labels_, labels[idx])
    acc_alg = metrics.classification.accuracy_score(labels[idx], kmeans_clusters_alg)
    print('Subsampled k-means accuracy = %.3f' % acc_alg)
    clustidx = [np.where(km.labels_ == i)[0] for i in range(k)]  # indices per cluster
    sub_X = X[np.ix_(idx, idx)] if synthetic_data else X[idx, :]
    sub_n = len(idx)
    # Estimate alpha, beta, p, q parameters
    if synthetic_data:
        intra_clust_sums = [np.sum(sub_X[np.ix_(tidx, tidx)]) for tidx in clustidx] if synthetic_data else [
            np.sum(sub_X[tidx, :]) for tidx in clustidx]
        p_hat = np.mean([intra_clust_sums[i] / (len(tidx) ** 2) for i, tidx in enumerate(clustidx)])
        q_hat = np.mean(
            [(np.sum(sub_X[tidx, :]) - intra_clust_sums[i]) / (len(tidx) * (sub_n - len(tidx))) for i, tidx in
             enumerate(clustidx)])
        # total_mse = 0.5 * np.sqrt((p - p_hat) ** 2 + (q - q_hat) ** 2)
        # print('p_hat = %.3f q_hat = %.3f mse = %.3f' % (p_hat, q_hat, total_mse))
        alpha, beta = compute_alpha_beta(p_hat, q_hat)
    else:
        alpha = np.mean([np.mean(1.0 - distance.pdist(sub_X[tidx, :], 'Hamming')) for tidx in clustidx])
        beta = np.mean(
            [np.mean(1.0 - distance.cdist(sub_X[tidx, :], sub_X[np.setdiff1d(range(1, sub_n), tidx), :], 'Hamming')) for
             tidx in clustidx])
    print('alpha = %.3f beta = %.3f' % (alpha, beta))
    t, D = compute_t_D(n, alpha, beta)
    print('t = %.3f D = %.3f' % (t, D))
    return t, D


# endregion

# region deprecated
def ecc_kmeans(X, T, k, s, labels, verbose=True):
    ### Error Correcting Code approach
    n = X.shape[0]
    tic = time.time()
    np.random.seed(int(time.time()))
    print(s, n, T)
    random_subsets = np.random.binomial(1, s, (n, T))
    parity_bits = np.mod(np.matmul(X, random_subsets), 2)
    reduced_data = np.hstack([X, parity_bits])
    subtime = time.time() - tic
    kmeans_clusters_alg, conf_mat = correct_label_assignment(kmeans(reduced_data, k), labels)
    acc_alg = metrics.classification.accuracy_score(labels, kmeans_clusters_alg)
    time_alg = time.time() - tic
    print("ALG took %.3s seconds. Accuracy=%s" % (time_alg, acc_alg))
    if verbose:
        print(metrics.classification.classification_report(labels, kmeans_clusters_alg))
        print(metrics.confusion_matrix(labels, kmeans_clusters_alg))
    return reduced_data, acc_alg, time_alg, subtime


def condition_on_T_BX():
    runs = 20
    p, q = 0.01, 0.003
    s = 0.5
    T = [1, 5, 10, 20, 30, 40, 50, 80, 100, 150, 200]  # int(n*(n-1)/2)
    X, mapping, inv_mapping, books = BX.Wrapper()
    res = list()
    for t in T:
        subres = list()
        for step in range(runs):
            print('T=%s iter #%s' % (t, step))
            # X, labels = generate_block_stochastic_data(n, 4, p, q)
            clust_alg = ecc_kmeans_books(X, t, 4, s, False)
            save_clusters(t, clust_alg, books, inv_mapping)
        #     subres.append(acc_alg)
        # res.append(np.asarray(subres))
    # res = np.asarray(res)
    # errs = np.std(res, axis=1)
    # means = np.mean(res, axis=1)
    # plt.errorbar(T, means, yerr=errs)
    # plt.ylabel('Clustering accuracy')
    # plt.xlabel('ECC code size')
    # plt.title(('ECC accuracy on %s runs at n=%s'+'\n'+'p=%s q=%s s=%s') % (runs, n, p, q, s))
    # plt.savefig('results_conditioned_onT_%s_%s_%s.png' % (n, p, q))
    # plt.show()


def condition_on_T_mush(n):
    runs = 20
    p, q = 0.01, 0.003
    s = 0.5
    T = [1, 5, 10, 20, 30, 40, 50, 80, 100]  # , 150, 200]# int(n*(n-1)/2)
    X, labels = mush.ReadMushrooms()
    res = list()
    for t in T:
        subres = list()
        for step in range(runs):
            print('T=%s iter #%s' % (t, step))
            _, acc_alg, _, _ = ecc_kmeans(X, t, 2, s, labels, False)
            subres.append(acc_alg)
        res.append(np.asarray(subres))
    res = np.asarray(res)
    errs = np.std(res, axis=1)
    means = np.mean(res, axis=1)
    plt.errorbar(T, means, yerr=errs)
    plt.ylabel('Clustering accuracy')
    plt.xlabel('ECC code size')
    plt.title(('ECC accuracy on %s runs at n=%s' + '\n' + 'p=%s q=%s s=%s') % (runs, n, p, q, s))
    plt.savefig('results_conditioned_onT_%s_%s_%s.png' % (n, p, q))
    plt.show()


# def ecc_kmeans_reals(X, T, k, s, labels, verbose=True):
#     ### Error Correcting Code approach
#     n = X.shape[0]
#     tic = time.time()
#     np.random.seed(int(time.time()))
#     random_subsets = np.random.binomial(1, s, (n, T))
#     reduced_data = np.hstack([X, parity_bits])
#     subtime = time.time() - tic
#     kmeans_clusters_alg, conf_mat = correct_label_assignment(kmeans(reduced_data, k), labels)
#     acc_alg = metrics.classification.accuracy_score(labels, kmeans_clusters_alg)
#     time_alg = time.time() - tic
#     print("ALG took %.3s seconds. Accuracy=%s" % (time_alg, acc_alg))
#     if verbose:
#         print(metrics.classification.classification_report(labels, kmeans_clusters_alg))
#         print(metrics.confusion_matrix(labels, kmeans_clusters_alg))
#     return reduced_data, acc_alg, time_alg, subtime

if False:
    print(metrics.classification.classification_report(labels, kmeans_clusters_standard))
    print(metrics.confusion_matrix(labels, kmeans_clusters_standard))
    plt.subplot(1, 2, 1)
    tmp = distance.squareform(distance.pdist(X))
    np.fill_diagonal(tmp, np.mean(tmp))
    plt.imshow(tmp)
    plt.subplot(1, 2, 2)
    tmp = distance.squareform(distance.pdist(reduced_data))
    np.fill_diagonal(tmp, np.mean(tmp))
    plt.imshow(tmp)


# endregion

# def print_data(X):
#     for i in range(len(X)):
#         for j in range(len(X[i])):
#             print(X[i][j]+" ", end='')
#         print("\n")


def ecc_kmeans_v2(X, s, T):
    def sample_row(row, s, T):
        """
        treats samples as polynom coefficients and evaluates at random point.
        :param row: row from apply_along_axis
        :param s: sample rate transformed into sample size
        :param T: number of added bits
        :return: returns T parity bits
        """

        def as_polynom(entries):
            t = entries.shape[1]
            s = np.random.rand()
            res = entries.tolist()[0]
            for p in range(t):
                res[p] = res[p] * s ** p
            return np.expand_dims(np.asarray(res), 0)

        return np.sum(np.asarray(
            [as_polynom(row[0, i]) for i in np.random.randint(0, row.shape[1], (T, int(s * row.shape[1])))]).squeeze(),
                      axis=1)

    ### Error Correcting Code approach
    tic = time.time()
    np.random.seed(int(time.time()))
    parity_bits = np.apply_along_axis(sample_row, 1, X, s, T)
    reduced_data = np.hstack([X, parity_bits])
    subtime = time.time() - tic
    kmeans_clusters_alg, conf_mat = correct_label_assignment(kmeans(reduced_data, k), labels)
    acc_alg = metrics.classification.accuracy_score(labels, kmeans_clusters_alg)
    time_alg = time.time() - tic
    print("ALG took %.3s seconds. Accuracy=%s" % (time_alg, acc_alg))
    return reduced_data, acc_alg, time_alg, subtime


def randsphere(n, d):
    # samples n points uniformly from within a d-dimensional sphere using Muller method (#20):
    # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    u = np.random.normal(0, 1, (d, n))  # an array of d normally distributed random variables
    norm = sp.linalg.norm(u, axis=0)
    r = np.power(np.random.rand(n), 1.0 / d)
    return ((r * u) / norm).transpose()


def matrix_form_ecc_sampling(X, T):
    #np.random.seed(int(time.time()))

    n, d = X.shape
    T = int(T)
    # print(X.shape[1], X.shape[0], T)
    # T=2
    if False:
        p_i = 20.0/n
        sample = np.random.random_sample(n)
    #    print(sample)
        sampled_scaled_ = []
        for i in range(len(sample)):
            u = [0]*n
            if sample[i] < p_i:
                u[i] = 1
            sampled_scaled_.append(np.array(u))

        # sample_scaled_ = np.mat([np.array([0]*(i-1)+[1]+[0]*(n-i))
        #                          for i in range(len(sample))])
        X_no_outliers_ = np.dot(np.mat(sampled_scaled_), X)
    #    print("Outliers shapes", X_no_outliers_)

    medX = np.median(X, axis=0)
    dsts = distance.cdist([medX], X)
    main_mass = (dsts < np.percentile(dsts, 90)).flatten()
    X_no_outliers_ = X[main_mass, :]

    Xmean = np.mean(X_no_outliers_, axis=0)
    centeredX = X_no_outliers_ - Xmean
    Xnorm = np.max(np.linalg.norm(centeredX, axis=1))
    
    A = np.square(np.linalg.norm(X, axis=1))
    Z = randsphere(T, d) * Xnorm + Xmean # np.random.rand(T, d) * Xnorm + Xmean
    B = np.square(np.linalg.norm(Z, axis=1))
    C = np.matmul(X, Z.transpose())
    A_B = np.broadcast_to(np.expand_dims(A, axis=1), (n, T)) + np.broadcast_to(np.expand_dims(B, axis=1), (T, n)).transpose()
    D = np.sqrt(A_B - 2*C)
    return D


def matrix_form_simhash(X, T):
    n, d = X.shape
    T = int(T)
    Xmean = np.mean(X, axis=0)
    centeredX = X - Xmean
    r = np.random.rand(d, T)
    D = np.sign(np.dot(centeredX, r))
    return D


def loop_form_ecc_sampling(X, T):
    n, d = X.shape
    # print(X.shape[1], X.shape[0], T)
    # T=2
    D = np.zeros((n, T))
    for i in range(T):
        vect = np.random.binomial(1, 1, d)
        sizevect = sum(vect)
        r = np.random.rand(sizevect)
        for l in range(n):
            dp = 0
            count = 0
            for j in range(d):
                if vect[j] == 1:
                    dp += (1 / float(sizevect)) * ((r[count] - X[l, j]) ** 2)
                    count += 1
            D[l, i] = np.sqrt(dp)
    return D


def ecc_kmeans_v2_reals(X, T, labels, s=1):
    """
    Computes euclidean distance to T random points
    :param X:
    :param s:
    :param T:
    :return:
    """
    ### Error Correcting Code approach
    tic = time.time()
    #D1 = matrix_form_ecc_sampling(X, T)
    #D2 = loop_form_ecc_sampling(X, T)
    D3 = matrix_form_simhash(X, T)
    D = D3

    reduced_data = np.hstack([X, D])
    # print(X_reduced)

    #    parity_bits = np.apply_along_axis(sample_row, 1, X, s, T)
    subtime = time.time() - tic
    kmeans_clusters_alg, conf_mat = correct_label_assignment(kmeans(reduced_data, k), labels)
    acc_alg = metrics.classification.accuracy_score(labels, kmeans_clusters_alg)
    time_alg = time.time() - tic
    print("ALG took %.3s seconds. Accuracy=%s" % (time_alg, acc_alg))
    return reduced_data, acc_alg, time_alg, subtime


def ecc_kmeans_iterations(X, s, T, labels, iter=1):
    """
    Runs the algorithm several iterations to mitigate any randomness related issues.
    :param X: 
    :param s: 
    :param T: 
    :param labels: 
    :param iter: 
    :return: 
    """
    reduced_data_lst, acc_alg_lst, time_alg_lst, subtime_lst = [], [], [], []
    for i in range(iter):
        reduced_data, acc_alg, time_alg, subtime = ecc_kmeans_v2_reals(X, T, labels, s)
        reduced_data_lst.append(reduced_data)
        acc_alg_lst.append(acc_alg)
        time_alg_lst.append(time_alg)
        subtime_lst.append(subtime)
    return np.asarray(reduced_data_lst), np.asarray(acc_alg_lst), np.asarray(time_alg_lst), np.asarray(subtime_lst)


def compute_all_kmeans(X, T, k, s, labels):
    """
    Computes a comparison between different algorithms on X,labels dataset
    :param X: array shape=(n_samples, n_features)
    :param T: ECC code length
    :param k: for k-means
    :param labels: ground truth cluster ids
    :param verbose: print details
    :return: acc_alg, acc_pca, acc_vanilla
    """
    n = X.shape[0]

    ### Error Correcting Code approach
    reduced_data, acc_alg, time_alg, subtime = ecc_kmeans_v2_reals(X, T, labels, s)

    ### Classic PCA approach
    tic = time.time()
    reduced_data2 = PCA().fit_transform(X)
    kmeans_clusters_standard, conf_mat2 = correct_label_assignment(kmeans(reduced_data2, k), labels)
    acc_pca = metrics.classification.accuracy_score(labels, kmeans_clusters_standard)
    time_pca = time.time() - tic
    print("PCA (%s components) took %.3s seconds. Accuracy=%s" % (n, time_pca, acc_pca))

    ### ECC-PCA approach
    tic = time.time()
    reduced_data3 = PCA().fit_transform(reduced_data)
    kmeans_clusters_eccpca, conf_mat3 = correct_label_assignment(kmeans(reduced_data3, k), labels)
    acc_eccpca = metrics.classification.accuracy_score(labels, kmeans_clusters_eccpca)
    time_eccpca = time.time() - tic + subtime
    print("ECC-PCA (%s components) took %.3s seconds. Accuracy=%s" % (n, time_eccpca, acc_eccpca))

    ### Vanilla k-means approach
    tic = time.time()
    reduced_data4 = X
    kmeans_clusters_vanilla, conf_mat3 = correct_label_assignment(kmeans(reduced_data4, k), labels)
    acc_vanilla = metrics.classification.accuracy_score(labels, kmeans_clusters_vanilla)
    time_vanilla = time.time() - tic
    print("Vanilla k-means took %.3s seconds. Accuracy=%s" % (time_vanilla, acc_vanilla))

    return acc_alg, acc_pca, acc_eccpca, acc_vanilla, time_alg, time_pca, time_eccpca, time_vanilla


def kmeans(X, k):
    alg = KMeans(init='random', n_clusters=k, n_init=10, n_jobs=5)
    alg.fit(X)
    # kmeans_clusters = {c: [] for c in range(k)}
    # for i in range(len(X)):
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


def wrapper(n):
    runs = 20
    # p = 1 / 10  # log(n)/sqrt(n)
    # q = 1 / 200  # log(n)/(10*sqrt(n))
    p, q = 0.01, 0.003
    # p, q = 0.01, 0.003
    s = 0.5  # sample size
    T = 5  # int(n*(n-1)/2)
    res = list()
    timeres = list()
    for step in range(runs):
        print('iter #%s' % step)
        X, labels = generate_block_stochastic_data(n, k, p, q)
        iterout = compute_all_kmeans(X, T, k, s, labels)
        res.append(iterout[:4])
        timeres.append(iterout[4:])
    res = np.asarray(res)
    timeres = np.asarray(timeres)
    plt.subplot(2, 1, 1)
    plt.boxplot(res, notch=True, labels=['', '', '', ''])
    plt.ylabel('Clustering accuracy')
    anovap = stats.f_oneway(*res.transpose())
    plt.title(('Comparing %s runs at n=%s' + '\n' + 'p=%s q=%s T=%s s=%s' + '\n' + 'p-value(anova)=%.2E') % (
        runs, n, p, q, T, s, anovap.pvalue))
    plt.subplot(2, 1, 2)
    plt.boxplot(timeres, notch=True, labels=['ALG', 'PCA', 'ECC-PCA', 'VANILLA'])
    plt.ylabel('Runtime(sec)')
    plt.savefig('results_%s_%s_%s_%s.png' % (n, p, q, T))
    plt.show()


def condition_on_T(X, labels, k, t, D, p, iter=1):
    runs = 20
    p, q = 0.01, 0.003
    s = 0.5
    T = [1, 5, 10, 20, 30, 40, 50, 80, 100, 150, 200]  # int(n*(n-1)/2)
    res = list()
    for t in T:
        subres = list()
        for step in range(runs):
            print('T=%s iter #%s' % (t, step))
            X, labels = generate_block_stochastic_data(n, 4, p, q)
            _, acc_alg, _, _ = ecc_kmeans(X, t, 4, s, labels, False)
            subres.append(acc_alg)
        res.append(np.asarray(subres))
    res = np.asarray(res)
    errs = np.std(res, axis=1)
    means = np.mean(res, axis=1)
    plt.errorbar(T, means, yerr=errs)
    plt.ylabel('Clustering accuracy')
    plt.xlabel('ECC code size')
    plt.title(('ECC accuracy on %s runs at n=%s' + '\n' + 'p=%s q=%s s=%s') % (runs, n, p, q, s))
    plt.savefig('results_conditioned_onT_%s_%s_%s.png' % (n, p, q))
    plt.show()


def evaluate_dataset_plot(X, labels, k, t, D, p, iter=1):
    """
    :param X: dataset
    :param labels: ground truth labels
    :param k: k clusters
    :param t: subset size
    :param D: #added dimensions
    :param p: evaluated norm
    :param iter: #iterations to run per param set
    :return:
    """

    res = list()
    timeres = list()
    for T in D:
        iter_vals, iter_times = [], []
        for itr in range(iter):
            print('Iteration #%d' % (itr))
            iterout = compute_all_kmeans(X, T, k, t / X.shape[1], labels)
            iter_vals.append(iterout[:4])
            iter_times.append(iterout[4:])
        res.append(np.mean(iter_vals, axis=0))
        timeres.append(np.mean(iter_times, axis=0))
    res = np.asarray(res)
    timeres = np.asarray(timeres)
    res = np.asarray(res)
    plt.imshow(res)
    plt.yticks(range(len(D)), D)
    plt.xticks(range(4), ['ALG', 'PCA', 'ECC-PCA', 'VANILLA'])
    cbar = plt.colorbar()
    cbar.set_label('Accuracy')
    plt.xlabel('Algorithm')
    plt.ylabel('Added parity coordinates')

    for i in range(len(D)):
        for j in range(4):
            text = plt.text(j, i, '{:.4f}'.format(res[i, j]).replace('0.', '.'), ha="center", va="center", color="w")

    plt.savefig('Result_' + datetime.datetime.now().strftime('%Y%m%d%H%M') + '.png')
    plt.ion()
    plt.show(block=False)
    print('Evaluated.')


def plot_norm_density(X, name):
    n, d = X.shape
    norms = np.linalg.norm(X, axis=1)
    density = stats.gaussian_kde(norms)
    xs = np.linspace(np.min(norms), np.max(norms), 200)
    plt.plot(xs, density(xs), label=name)
    plt.xlabel('l2 norm')


def debug_plot_densities(X, name):
    plot_norm_density(X, name)
    plot_norm_density(np.random.rand(n, X.shape[1]), 'random vector')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # "iris"
    # "digits"
    # "boston"
    # "cancer"
    # "KDD"
    # "mushrooms"
    # "SBM" with explicit parameters n,k,p,q
    name = "cancer"

    X, n, labels, k = get_dataset(name) #, n=600, k=4, p=0.6, q=0.2)
    #debug_plot_densities(X, name)

    # t, D = kmeans_subsample_density_estimator(X, labels, sample_ratio=0.2)

    print('n=%d k=%d' % (n, k))

    t = X.shape[1]
    D = list(range(105, 106, 1)) #[25, 50, 75, 100]
    p, iters = 2, 10
    evaluate_dataset_plot(X, labels, k, t, D, p, iters)

    # condition_on_T(600)
    # wrapper(600)

    print('done.')
