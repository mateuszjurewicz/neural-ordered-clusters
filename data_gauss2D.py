"""
All credit for implementations and their adjustments goes to Ari Pakman et al. and Juho Lee et al., as per their cited papers.
"""

import random

import numpy as np
import torch
import math

from copy import deepcopy
from utils import relabel, get_ordered_cluster_assignments
from sklearn.metrics import v_measure_score, mean_squared_error
from scipy.stats import spearmanr, kendalltau
from collections import Counter


def get_generator(params):
    if params['generative_model'] == 'Gauss2D' and not params['max_cardinality']:
        return Gauss2D_generator_with_order_simple(params)
    elif params['generative_model'] == 'Gauss2D' and params['max_cardinality']:
        return Gauss2D_generator_with_order_max_cardinality(params)
    else:
        raise NameError('Unknown generative model ' + params['generative_model'])


class Gauss2D_generator():

    def __init__(self, params):
        self.params = params

    def generate(self, N=None, batch_size=1):
        lamb = self.params['lambda']
        sigma = self.params['sigma']
        x_dim = self.params['x_dim']

        clusters, N, num_clusters = generate_CRP(self.params, N=N)
        cumsum = np.cumsum(clusters)
        data = np.empty([batch_size, N, x_dim])
        cs = np.empty(N, dtype=np.int32)
        for i in range(num_clusters):

            # generate cluster points
            mu = np.random.normal(0, lamb, size=[x_dim * batch_size, 1])
            samples = np.random.normal(mu, sigma, size=[x_dim * batch_size, clusters[i + 1]])
            samples = np.swapaxes(samples.reshape([batch_size, x_dim, clusters[i + 1]]), 1, 2)

            # update data and cs
            data[:, cumsum[i]:cumsum[i + 1], :] = samples
            cs[cumsum[i]:cumsum[i + 1]] = i + 1

        # %shuffle the assignment order
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]

        data = data[:, arr, :]

        # relabel cluster numbers so that they appear in order
        cs = relabel(cs)

        # normalize data
        # means = np.expand_dims(data.mean(axis=1),1 )
        medians = np.expand_dims(np.median(data, axis=1), 1)

        data = data - medians
        # data = 2*data/(maxs-mins)-1        #data point are now in [-1,1]

        return data, cs, clusters, num_clusters


class Gauss2D_generator_with_order_simple():

    def __init__(self, params):
        self.params = params

    @staticmethod
    def get_crp_from_chosen_clusters(cs_list):
        """
        Take a list of cluster labels, turn into a CRP output that can be used to generate an example
        """

        crp_clusters = np.asarray(cs_list)
        crp_num_clusters = len(crp_clusters)
        crp_N = sum(crp_clusters)
        crp_clusters = np.pad(crp_clusters, (1, crp_N - crp_num_clusters + 1), 'constant') # pad to N + 2
        crp = dict(clusters=crp_clusters,
                   N=crp_N,
                   num_clusters=crp_num_clusters)
        return crp

    def generate(self, N=None, batch_size=1, crp=None):
        lamb = self.params['lambda']
        sigma = self.params['sigma']
        x_dim = self.params['x_dim']

        # use give number of clusters and their cardinality
        if crp:
            clusters = crp['clusters']
            N = crp['N']
            num_clusters = crp['num_clusters']
        else:
            clusters, N, num_clusters = generate_CRP(self.params, N=N)

        cumsum = np.cumsum(clusters)
        data = np.empty([batch_size, N, x_dim])
        cs = np.empty(N, dtype=np.int32)

        s = []
        origin = np.zeros([batch_size, x_dim])
        for i in range(num_clusters):
            # get cluster seed points, batched
            mu = np.random.normal(0, lamb, size=[x_dim * batch_size, 1])
            s.append(mu)

        # reshape to (num_clusters, batch, x_dim)
        batched_seeds = np.stack(s).reshape(num_clusters, batch_size, x_dim)

        # get distances
        d = batched_seeds - origin
        distances = np.hypot(d[:, :, 0], d[:, :, 1])

        # get sorted indexes for distances
        sorted_ind = np.argsort(np.transpose(distances), axis=1)

        # reshape
        batched_seeds = np.swapaxes(batched_seeds, 0, 1)

        # sort them
        sorted_seeds = np.asarray([batched_seeds[i][sorted_ind[i]] for i in range(sorted_ind.shape[0])])

        for i in range(num_clusters):

            # take the proper, ordered cluster seeds
            mu = sorted_seeds[:, i, :].reshape(batch_size * x_dim, 1)

            # generate cluster points
            samples = np.random.normal(mu, sigma, size=[x_dim * batch_size, clusters[i + 1]])
            samples = np.swapaxes(samples.reshape([batch_size, x_dim, clusters[i + 1]]), 1, 2)

            # update data and cs
            data[:, cumsum[i]:cumsum[i + 1], :] = samples
            cs[cumsum[i]:cumsum[i + 1]] = i + 1

        # %shuffle the assignment order
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]

        data = data[:, arr, :]

        # relabel cluster numbers so that they appear in order
        cs_relabelled = relabel(cs)
        cs_ordered = cs - 1  # make it zero indexed

        # Y_att
        # get ordered-to-relabelled
        o2r = {r: cs_ordered[i] for i, r in enumerate(cs_relabelled)}

        # use o2r to get the y_att
        y_ci = torch.zeros(num_clusters)
        y_att = torch.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            proper_cluster_index_for_ith_cluster = o2r[i]
            y_ci[i] = proper_cluster_index_for_ith_cluster
            y_att[i][proper_cluster_index_for_ith_cluster] = 1

        # repeat it for the entire batch
        Y_att = torch.tile(y_att, (batch_size, 1, 1))
        Y_ci = torch.tile(y_ci, (batch_size, 1))

        # turn to torch
        # data = torch.from_numpy(data).float()
        # cs_relabelled = torch.from_numpy(cs_relabelled)

        return data, cs_relabelled, clusters, num_clusters, cs_ordered, Y_att, Y_ci


class Gauss2D_generator_with_order_max_cardinality():

    def __init__(self, params):
        self.params = params

    @staticmethod
    def get_crp_from_chosen_clusters(cs_list):
        """
        Take a list of cluster labels, turn into a CRP output that can be used to generate an example
        """

        crp_clusters = np.asarray(cs_list)
        crp_num_clusters = len(crp_clusters)
        crp_N = sum(crp_clusters)
        crp_clusters = np.pad(crp_clusters, (1, crp_N - crp_num_clusters + 1), 'constant') # pad to N + 2
        crp = dict(clusters=crp_clusters,
                   N=crp_N,
                   num_clusters=crp_num_clusters)
        return crp

    def generate(self, N=None, batch_size=1, crp=None):
        lamb = self.params['lambda']
        sigma = self.params['sigma']
        x_dim = self.params['x_dim']
        max_cardinality = self.params['max_cardinality']

        # use give number of clusters and their cardinality
        if crp:
            clusters = crp['clusters']
            N = crp['N']
            num_clusters = crp['num_clusters']
        else:
            clusters, N, num_clusters = generate_CRP(self.params, N=N)

        data = np.empty([batch_size, N, x_dim])
        cs = np.empty(N, dtype=np.int32)

        s = []
        origin = np.zeros([batch_size, x_dim])
        for i in range(num_clusters):
            # get cluster seed points, batched
            mu = np.random.normal(0, lamb, size=[x_dim * batch_size, 1])
            s.append(mu)

        clusters, cumsum, num_clusters, s = enforce_max_cardinality(max_cardinality, clusters, s)

        # reshape to (num_clusters, batch, x_dim)
        batched_seeds = np.stack(s).reshape(num_clusters, batch_size, x_dim)

        # get distances
        d = batched_seeds - origin
        distances = np.hypot(d[:, :, 0], d[:, :, 1])

        # get sorted indexes for distances
        sorted_ind = np.argsort(np.transpose(distances), axis=1)

        # reshape
        batched_seeds = np.swapaxes(batched_seeds, 0, 1)

        # sort them
        sorted_seeds = np.asarray([batched_seeds[i][sorted_ind[i]] for i in range(sorted_ind.shape[0])])

        for i in range(num_clusters):

            # take the proper, ordered cluster seeds
            mu = sorted_seeds[:, i, :].reshape(batch_size * x_dim, 1)

            # generate cluster points
            samples = np.random.normal(mu, sigma, size=[x_dim * batch_size, clusters[i + 1]])
            samples = np.swapaxes(samples.reshape([batch_size, x_dim, clusters[i + 1]]), 1, 2)

            # update data and cs
            data[:, cumsum[i]:cumsum[i + 1], :] = samples
            cs[cumsum[i]:cumsum[i + 1]] = i + 1

        # %shuffle the assignment order
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]

        data = data[:, arr, :]

        # relabel cluster numbers so that they appear in order
        cs_relabelled = relabel(cs)
        cs_ordered = cs - 1  # make it zero indexed

        # Y_att
        # get ordered-to-relabelled
        o2r = {r: cs_ordered[i] for i, r in enumerate(cs_relabelled)}

        # use o2r to get the y_att
        y_ci = torch.zeros(num_clusters)
        y_att = torch.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            proper_cluster_index_for_ith_cluster = o2r[i]
            y_ci[i] = proper_cluster_index_for_ith_cluster
            y_att[i][proper_cluster_index_for_ith_cluster] = 1

        # repeat it for the entire batch
        Y_att = torch.tile(y_att, (batch_size, 1, 1))
        Y_ci = torch.tile(y_ci, (batch_size, 1))

        # # turn to torch
        # data = torch.from_numpy(data).float()
        # cs_relabelled = torch.from_numpy(cs_relabelled)

        return data, cs_relabelled, clusters, num_clusters, cs_ordered, Y_att, Y_ci


def generate_CRP(params, N, no_ones=False):
    alpha = params['alpha']  # dispersion parameter of the Chinese Restaurant Process
    keep = True

    while keep:
        if N is None or N == 0:
            N = np.random.randint(params['Nmin'], params['Nmax'])

        clusters = np.zeros(N + 2)
        clusters[0] = 0
        clusters[1] = 1  # we start filling the array here in order to use cumsum below
        clusters[2] = alpha
        index_new = 2
        for n in range(N - 1):  # we loop over N-1 particles because the first particle was assigned already to cluster[1]
            p = clusters / clusters.sum()
            z = np.argmax(np.random.multinomial(1, p))
            if z < index_new:
                clusters[z] += 1
            else:
                clusters[index_new] = 1
                index_new += 1
                clusters[index_new] = alpha

        clusters[index_new] = 0
        clusters = clusters.astype(np.int32)

        if no_ones:
            clusters = clusters[clusters != 1]
        N = int(np.sum(clusters))
        keep = N == 0

    K = np.sum(clusters > 0)

    return clusters, N, K


def get_metrics(a_model, a_data_generator, log,
                clustering_metric=v_measure_score, ordering_metric=kendalltau, max_cardinality=None,
                m_examples=50, n_infer_samples=100):
    """
    Take a model, a data generator and two metrics (one for clustering, one for ordering clusters).
    Generate a specified number of examples and make predictions using the model.
    Report the metrics from those predictions.
    :param a_model: dpmm + set2seq or such
    :param a_data_generator: e.g. Gauss2D with ordering
    :param clustering_metric: v-score or such
    :param ordering_metric: ?
    :param max_cardinality: whether (and what) the max-cardinality cap was on the training data
    :param m_examples: number of examples to measure metrics on
    :param n_infer_samples: number of samples to generate from dpmm
    :return: a clustering score and an ordering score
    """
    # sampling works on single examples, so this will not be parallel
    clustering_scores = []
    ordering_scores = []
    max_card_scores = []

    # infer on m_examples
    for i in range(m_examples):

        if i % 10 == 0:
            log.info('Metrics calculation for example {} / {} ...'.format(i+1, m_examples))

        # generate single example
        x, cs_target, clusters, num_clusters, cs_ordered, Y_att, Y_ci = a_data_generator.generate(None, batch_size=1)

        # infer on it
        cs_pred, co_pred = a_model.infer(x, n_samples=n_infer_samples)

        # calculate clustering score
        c_score = clustering_metric(cs_target, cs_pred)
        clustering_scores.append(c_score)

        # if max-cardinality cap, calculate adherence percentage (0-1)
        if max_cardinality:
            max_card_score = mca_metric(cs_pred, max_cardinality)
            max_card_scores.append(max_card_score)

        # calculate ordering score
        # because of the way tau/rho handle repeated/shared ranks
        # we need to append a meaningless rank at the end of both, if either is just one rank
        correlation_rank_token = 999
        if len(set(cs_ordered)) == 1 or len(set(cs_pred)) == 1:
            cs_ordered = np.append(cs_ordered, correlation_rank_token)
            cs_pred = np.append(cs_pred, correlation_rank_token)

        o_score, _ = ordering_metric(cs_ordered, cs_pred)
        ordering_scores.append(o_score)


    # reduce
    avg_clustering_score = sum(clustering_scores) / len(clustering_scores)
    avg_ordering_score = sum(ordering_scores) / len(ordering_scores)

    # construct results dictionary
    r = dict(
        cs_score=avg_clustering_score,
        co_score=avg_ordering_score
    )

    if max_cardinality:
        avg_max_card_score = sum(max_card_scores) / len(max_card_scores)
        r['mc_score'] = avg_max_card_score

    return r

def mca_metric(pred_cs, max_card):
    """Take a prediction of cluster labels and the max cardinality allowed, return 1 if no clusters exceeded, 0 if any did"""
    counts = Counter(pred_cs)

    for cluster, cardinality in counts.items():
        if cardinality > max_card:
            return 0
    return 1

def enforce_max_cardinality(max_cardinality, cluster_cardinalities, mus):
    """Take cluster cardinalities, cumulative sum, number of clusters and generated centroids (mus), adjust to adhere by max cardinality"""

    # placeholders
    new_clusters = []
    s_tracker = []

    # loop over cardinalities
    for i, c in enumerate(cluster_cardinalities):

        if c > max_cardinality:
            # find the new optimal sub-cardinalities (splits)
            split_cardinalities = get_split_cardinalities(c, max_cardinality)

            # add them
            new_clusters.extend(split_cardinalities)

            # remember the numbers, to know where to duplicate centroids (mus), and how many times
            s_tracker.append((i-1, len(split_cardinalities)))  # -1 because we're referring to mus, which are not zero padded

        else:
            new_clusters.append(c)

    # recreate proper objects based on new clusters
    # recreate new mus:
    new_s = duplicate_mus(mus, s_tracker)

    # recrete cumsum
    new_cumsum = np.cumsum(new_clusters)

    # adjust to proper type
    new_clusters = np.asarray(new_clusters)

    # count n_clusters
    new_num_clusters = np.count_nonzero(new_clusters)

    return new_clusters, new_cumsum, new_num_clusters, new_s


def get_split_cardinalities(given_cardinality, max_cardinality):
    """Take an actual cardinality of a cluster, split it into equal parts as much as possible, depending on max cardinality"""
    # placeholder
    new_splits = []

    # try to break it into as few parts as possible, such that all parts are smaller or equal to max cardinality
    for divisor in range(2, given_cardinality):

        # add new splits
        for i in range(divisor):
            new_splits.append(given_cardinality // divisor)

        # handle leftover
        leftover = given_cardinality % divisor

        # distribute lefover equally
        for j in range(leftover):
            # guard against being out of new_splits to add to and having to circle back
            idx =  (j + 1) % (len(new_splits) - 1)
            new_splits[idx] += 1

        # check if all splits meet requirement
        requirement_met = True
        for split in new_splits:
            if split > max_cardinality:
                requirement_met = False

        # either return or reset placeholder
        if requirement_met:
            return new_splits
        else:
            new_splits = []


def duplicate_mus(original_mus, split_tracker):
    """
    Take the original mus (centroids), as a list of ndarrays, and the split tracker (list of tuples with indices and num splits,
    Return the properly duplicated mus.
    """
    # need to know how many splits we've already inserted
    new_s = deepcopy(original_mus)

    n_added_splits = 0
    for i_n in split_tracker:
        original_idx = i_n[0] + n_added_splits
        current_mu = new_s[original_idx]
        num_splits = i_n[1]

        # start with 1, cause 1 original mu is already there
        for j in range(1, num_splits):
            new_s.insert(original_idx, current_mu)
            n_added_splits += 1

    return new_s