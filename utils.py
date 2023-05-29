import logging
import sys
import tqdm
import uuid
import numpy as np


class TqdmStream(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, 'flush', lambda: None)()


def get_run_id(max_length=10):
    """
    Get a unique string id for an experiment run.
    :param max_length: id length (in characters), default 10
    :return: the string id
    """
    run_id = str(uuid.uuid4().fields[-1])[:max_length]
    return run_id


def get_logger(path, level=logging.INFO, run_id=''):
    """Get a basic logger that can handle tqdm"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format=f'\n%(asctime)s | %(levelname)s | {run_id} | %(message)s\n',
        handlers=[
            logging.FileHandler(filename=path, encoding='utf-8'),
            # log file
            logging.StreamHandler(sys.stdout)  # stdout
        ])
    log = logging.getLogger(__name__)
    logging.root.handlers[0].stream = TqdmStream(
        logging.root.handlers[0].stream)
    return log


def get_parameters(generative_model, arguments):
    params = {}

    if generative_model == 'Gauss2D':
        params['generative_model'] = 'Gauss2D'
        params['max_cardinality'] = arguments['max_cardinality']
        params['alpha'] = .7
        params['sigma'] = 1  # std for the Gaussian noise around the cluster mean
        params['lambda'] = 10  # std for the Gaussian prior that generates de centers of the clusters
        params['Nmin'] = 5
        params['Nmax'] = 100

        # elementwise
        if arguments['model'] == 'ENOC':
            params['x_dim'] = 2
            params['h_dim'] = 256
            params['g_dim'] = 512
            params['H_dim'] = 128
        elif arguments['model'] == 'CNOC':
            params['x_dim'] = 2
            params['h_dim'] = 128
            params['g_dim'] = 128
            params['H_dim'] = 128
            params['e_dim'] = 128
            params['z_dim'] = 128
            params['use_attn'] = True
            params['n_heads'] = 4
            params['n_inds'] = 32
            params['s2s_c_dim'] = 256
        elif arguments['model'] == 'NOC':
            params['x_dim'] = 2
            params['h_dim'] = 128
            params['g_dim'] = 256 # can be 128 or 256 for cardinality conditioned on G
            params['H_dim'] = 128
            params['e_dim'] = 128
            params['z_dim'] = 128
            params['use_attn'] = True
            params['n_heads'] = 4
            params['n_inds'] = 32
            params['s2s_c_dim'] = 256
            params['condition_cardinality_on_assigned'] = arguments['cardinality_conditioned_on_assigned']

    elif generative_model == 'SyntheticStructures':
        params['vocab_size'] = 5 if arguments['remove_start_end_markers'] else 7
        params['add_set_repr_to_elems'] = arguments['add_set_repr_to_elems']
        params['generative_model'] = 'SyntheticStructures'
        params['alpha'] = .7
        params['sigma'] = 1  # std for the Gaussian noise around the cluster mean
        params['lambda'] = 10  # std for the Gaussian prior that generates de centers of the clusters
        params['Nmin'] = 5
        params['Nmax'] = 100

        # elementwise
        if arguments['model'] == 'ENOC':
            params['x_dim'] = 1
            params['h_dim'] = 256
            params['g_dim'] = 512
            params['H_dim'] = 128
        elif arguments['model'] == 'CNOC':
            params['x_dim'] = 1
            params['h_dim'] = 128
            params['g_dim'] = 128
            params['H_dim'] = 128
            params['e_dim'] = 128
            params['z_dim'] = 128
            params['use_attn'] = True
            params['n_heads'] = 4
            params['n_inds'] = 32
            params['s2s_c_dim'] = 256
        elif arguments['model'] == 'FNOC':
            params['x_dim'] = 1
            params['h_dim'] = 128
            params['g_dim'] = 128 # can be 128 or 256 for cardinality conditioned on G
            params['H_dim'] = 128
            params['e_dim'] = 128
            params['z_dim'] = 128
            params['use_attn'] = True
            params['n_heads'] = 4
            params['n_inds'] = 32
            params['s2s_c_dim'] = 256
            params['condition_cardinality_on_assigned'] = arguments['cardinality_conditioned_on_assigned']
    else:
        raise NameError('Unknown generative model ' + generative_model)

    return params


def relabel(cs):
    cs = cs.copy()
    d = {}
    k = 0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k += 1
        cs[i] = d[j]

    return cs


def count_params(model, return_string=False):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = '{:,}'.format(params)
    if return_string:
        return params, 'The model has {} trainable parameters'.format(params)
    else:
        print('The model has {} trainable parameters'.format(params))


def get_ordered_cluster_assignments(cluster_assignments, cluster_order):
    """
    Take the predicted cluster assignments (per point) and the predicted
    order of clusters. Turn it into a per-point assignment of cluster index (as per predicted cluster order).
    :param cluster_assignments: np.array of size N, containing predicted cluster labels (order doesn't matter)
    :param cluster_order: torch.Tensor of size K_p (predicted number of clusters), where 0th element is the index
           of the cluster that should be first. Refers to the cluster order from cluster_assignments.
    :return: an np.array of length N, where assigned cluster labels match cluster order.
    """
    # obtain map of cluster labels to ordered cluster indices
    i2p = {int(e): i for i, e in enumerate(list(cluster_order))}
    ordered_cluster_assignments = np.asarray([i2p[e] for e in list(cluster_assignments)])
    return ordered_cluster_assignments
