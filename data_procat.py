"""
Script for all utilities regarding data & its generation.
"""
import ast
import copy
import csv
import datasets
import itertools
import numpy as np
import random
import statistics
import torch
import logging

from scipy.spatial import distance
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertTokenizer, AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
from copy import deepcopy
from matplotlib import rcParams
from random import choice, shuffle, randint, seed
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import v_measure_score, mean_squared_error
from scipy.stats import spearmanr, kendalltau


class ProcatData:
    """
    Object capable of turning a PROCAT dataset into a dataloader.
    """

    def __init__(self, dataset_type, logger, config, tokenizer):
        self.dataset_type = dataset_type
        self.logger = logger
        self.config = config
        self.tokenizer = tokenizer

    def get_dataloader(self):
        # load as a datasets object
        dataset = datasets.load_dataset(split=self.dataset_type,
                                        download_mode="reuse_dataset_if_exists",  # "reuse_dataset_if_exists" or "force_redownload" for debug
                                        **{'path': 'data/PROCAT/procat_dataset_loader.py'})

        # make smaller for debugging, if necessary
        if self.config['debug_dataset_sizes']:
            dataset = dataset.select(range(100))

        # map convert_to_features batch wise
        dataset = dataset.map(self.convert_to_features, batched=True,
                              batch_size=100)

        columns = ['s{}_input_ids'.format(i + 1) for i in range(200)] + \
                  ['s{}_attention_mask'.format(i + 1) for i in range(200)] + \
                  ['s{}_token_type_ids'.format(i + 1) for i in range(200)] + ['label']
        dataset.set_format(type="torch", columns=columns)

        # dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config['batch_size'])

        return dataloader

    def convert_to_features(self, batch):
        """A version that works with batching and dataloaders"""
        batch_y = batch['label']

        batch_x = [
            self.tokenizer(s, max_length=self.config['max_sentence_length'],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt", ) for s in
            batch['shuffled_sentences']]

        # placeholder
        b = {}

        # extract
        b_input_ids = {'s{}_input_ids'.format(i + 1): [e['input_ids'][i].tolist() for e in batch_x] for i in range(200)}
        b_attention_masks = {'s{}_attention_mask'.format(i + 1): [e['attention_mask'][i].tolist() for e in batch_x] for i in range(200)}
        b_token_type_ids = {'s{}_token_type_ids'.format(i + 1): [e['token_type_ids'][i].tolist() for e in batch_x] for i in range(200)}

        # update
        b.update(b_input_ids)
        b.update(b_attention_masks)
        b.update(b_token_type_ids)
        b['labels'] = batch_y

        return b

    def convert_to_features_single(self, batch):
        """A version that works with 1-elem batches"""
        batch_y = batch['label']

        batch_x = [
            self.tokenizer(s, max_length=self.config['max_sentence_length'],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt", ) for s in
            batch['shuffled_sentences']]

        b = {
            "input_ids": [e["input_ids"].tolist() for e in batch_x],
            "attention_mask": [e["attention_mask"].tolist() for e in batch_x],
            "token_type_ids": [e["token_type_ids"].tolist() for e in batch_x],
            "labels": batch_y,
        }
        return b


def get_procat_dataloader(dataset_type, logger, config, tokenizer):
    """
    Create the dataloaders for the procat datasets,
    per language model.
    """
    available_dataset_types = ['train', 'test', 'validation']
    if dataset_type not in available_dataset_types:
        error_msg = 'Incorrect dataset type specified, ' \
                    'try {}'.format(available_dataset_types)
        logging.exception(error_msg)
        raise Exception(error_msg)

    # handle each dataset separately (move if to function at some point)
    if config['dataset_name'] == 'PROCAT':
        # load the csv
        logger.info('Handling {} csv files'.format(config['dataset_name']))

        # load datasets
        dset = ProcatData(dataset_type, logger, config, tokenizer)
        dloader = dset.get_dataloader()

        return dloader


def get_batch(dataloaders_list, single_example_batch=False):
    """Take a list of dataloaders, choose one randomly, return objects expected by NOC"""
    # randomly choose a dataloader (n will differ)
    b = next(iter(choice(dataloaders_list)))

    if not single_example_batch:
        # if we want a full batch
        data = np.asarray(b['data'])
        cs_relabelled = np.asarray(b['cs_relabelled'])
        clusters = np.asarray(b['clusters'])
        K = b['K']
        cs_ordered = np.asarray(b['cs_ordered'])
        Y_att = b['Y_att']
        Y_ci = b['Y_ci']
    else:
        # otherwise, we want single example batch
        data = np.asarray(b['data'][0].unsqueeze(0))
        cs_relabelled = np.asarray(b['cs_relabelled'][0])
        clusters = np.asarray(b['clusters'][0])
        K = int(b['K'][0])
        cs_ordered = np.asarray(b['cs_ordered'][0])
        Y_att = b['Y_att'][0].unsqueeze(0)
        Y_ci = b['Y_ci'][0].unsqueeze(0)
    return data, cs_relabelled, clusters, K, cs_ordered, Y_att, Y_ci


def get_parameters_procat(_, arguments):
    params = {}

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
    elif arguments['model'] == 'NOC':
        params['x_dim'] = 1
        params['h_dim'] = 128
        params['g_dim'] = 128  # can be 128 or 256 for cardinality conditioned on G
        params['H_dim'] = 128
        params['e_dim'] = 128
        params['z_dim'] = 128
        params['use_attn'] = True
        params['n_heads'] = 4
        params['n_inds'] = 32
        params['s2s_c_dim'] = 256
        params['condition_cardinality_on_assigned'] = arguments['cardinality_conditioned_on_assigned']

    return params


def get_model_basic_procat_metrics(a_model: object, dataloaders: list, rulesets: dict, n_examples: int, n_samples: int, logger: object,
                                   clustering_metric: callable = v_measure_score, ordering_metric: callable = kendalltau) -> dict:
    """
    Take all objects needed to get n_examples predicted on, return basic clustering and cluster ordering scores.
    :param a_model: e.g. model_fnoc_batched.FNOC
    :param n_examples: how many examples to make predictions on
    :param dataloaders: list of dataloaders, from which to draw n_examples
    :param rulesets: loaded json dictionary with structural & compositional rules for catalog creation
    :param n_samples: how many samples to generate during inference
    :param logger: a logger object
    :param clustering_metric: a callable function returning 0-1 metric for clustering
    :param ordering_metric: a callable function returning 0-1 metric for cluster ordering
    :return: dict with clustering and ordering scores
    """
    # sampling works on single examples, so this will not be parallel
    clustering_scores = []
    ordering_scores = []

    logger.info('Beginning the inference for basic metrics ...')

    for i in range(n_examples):

        if i % 10 == 0:
            logger.info('Basic metrics calculation for example {} / {} ...'.format(i + 1, n_examples))

        # get example
        x, cs_target, clusters, num_clusters, cs_ordered, Y_att, Y_ci = get_batch(dataloaders,
                                                                                  single_example_batch=True)  # replace None with manual_clusters

        # predict
        y_hat_cs, y_hat_co = a_model.infer(x, n_samples=n_samples)

        # calculate clustering score
        c_score = clustering_metric(cs_target, y_hat_cs)
        clustering_scores.append(c_score)

        # calculate ordering score
        # because of the way tau/rho handle repeated/shared ranks
        # we need to append a meaningless rank at the end of both, if either is just one rank
        correlation_rank_token = 999
        if len(set(cs_ordered)) == 1 or len(set(y_hat_cs)) == 1:
            cs_ordered = np.append(cs_ordered, correlation_rank_token)
            y_hat_cs = np.append(y_hat_cs, correlation_rank_token)

        o_score, _ = ordering_metric(cs_ordered, y_hat_cs)
        ordering_scores.append(o_score)

    # reduce
    avg_clustering_score = sum(clustering_scores) / len(clustering_scores)
    avg_ordering_score = sum(ordering_scores) / len(ordering_scores)

    return {'CLUSTERING_SCORE_AVG': avg_clustering_score,
            'ORDERING_SCORE_AVG': avg_ordering_score}


def get_procat_tokenizer(language_model):
    """Instantiate the right tokenizer for a given language model"""
    known_language_models = ['danish-bert-botxo']

    # catch unknown models
    if language_model not in known_language_models:
        error_msg = 'Unknown language model provided: {}, try {}'.format(
            language_model, known_language_models)
        logging.exception(error_msg)
        raise Exception(error_msg)
    # BERT cased
    elif language_model == 'danish-bert-botxo':
        tokenizer = AutoTokenizer.from_pretrained(language_model, from_tf=True)
    else:
        error_msg = 'Language model known but not handled ' \
                    'in get_sentence_ordering_tokenizer().'
        logging.exception(error_msg)
        raise Exception(error_msg)

    return tokenizer


def decode_procat_batch(a_batch, a_tokenizer, a_logger):
    """Use the tokenizer to turn an offer vector back into readable text"""

    # get single offer's input ids
    single_input_ids = a_batch['s1_input_ids'][0]
    a_logger.info('Original input ids [:20]: {}'.format(single_input_ids[:20]))

    # detokenize
    single_readable = a_tokenizer.decode(single_input_ids)
    a_logger.info(single_readable)