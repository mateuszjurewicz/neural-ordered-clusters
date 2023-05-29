import json
import os
import torch

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from collections import Counter
from copy import deepcopy
from matplotlib import rcParams
from random import choice, shuffle, randint, seed
from utils import relabel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import v_measure_score, mean_squared_error
from scipy.stats import spearmanr, kendalltau


class SyntheticStructureDataset(Dataset):
    def __init__(self, data, cs_relabelled, clusters, K, cs_ordered, Y_ci, Y_att):
        self.data = data
        self.cs_relabelled = cs_relabelled
        self.clusters = clusters
        self.K = K
        self.cs_ordered = cs_ordered
        self.Y_ci = Y_ci
        self.Y_att = Y_att

    def __getitem__(self, index):
        data = self.data[index]
        cs_relabelled = self.cs_relabelled[index]
        clusters = self.clusters[index]
        K = self.K[index]
        cs_ordered = self.cs_ordered[index]
        Y_ci = self.Y_ci[index]
        Y_att = self.Y_att[index]

        return {'data': data,
                'cs_relabelled': cs_relabelled,
                'clusters': clusters,
                'K': K,
                'cs_ordered': cs_ordered,
                'Y_ci': Y_ci,
                'Y_att': Y_att,
                }

    def __len__(self):
        return len(self.K)


def create_catalogs(ruleset: dict, num_catalogs: int) -> list:
    """
    Take a ruleset in the form of a dictionary read from a configuration json, and a number of
    catalogs to generate according to that ruleset.
    :param ruleset:
    :param num_catalogs:
    :return: list of catalogs, each in the form of a list of sections, each as a list of token strings
    """
    # placeholder
    n_catalogs = []

    # unpack the ruleset
    input_sets = ruleset['input_set_compositions']
    sections = ruleset['sections']

    for i in range(num_catalogs):

        # new catalog
        new_catalog_as_sections = []

        # randomly choose an input set composition for current catalog
        isc_id, isc = choice(list(input_sets.items()))

        # unpack it's properties
        available_tokens = set(isc['token_set'])
        unused_tokens = available_tokens
        available_sections = isc['valid_sections']
        max_n_sections = isc['num_sections_max']

        # pick at least one sections from each valid order group
        ordered_sections = sorted(isc['valid_order'].items())
        for place_in_order, section_ids in ordered_sections:

            # handle empty order sections (catch-all for other sections whose order doesn't matter)
            # by skipping them
            if len(section_ids) == 0:
                continue

            s_id = choice(section_ids)

            # make sure section is composable from available tokens
            check_if_composable(s_id, sections, available_tokens, isc_id=isc_id)

            # mark those tokens as used
            used_tokens = set(sections[s_id])
            unused_tokens = available_tokens - used_tokens

            # add section to current catalog
            new_catalog_as_sections.append(s_id)

        # go through remaining tokens
        for token in unused_tokens:

            # shuffle the available sections
            shuffle(available_sections)

            # go through each possible section
            for s_id in available_sections:

                # checking if this section contains the current token
                if is_contained(token, s_id, sections):
                    # confirm its composable
                    check_if_composable(s_id, sections, available_tokens, isc_id=isc_id)

                    # add section to current catalog and exit loop
                    new_catalog_as_sections.append(s_id)
                    break

        # check if we have exceeded max sections, throw error
        check_if_too_long(new_catalog_as_sections, max_n_sections, isc_id)

        # randomly choose a number of pages to add, within limits
        max_pages_to_add = max_n_sections - len(new_catalog_as_sections)
        n_pages_to_add = randint(0, max_pages_to_add)

        # create and add as many pages
        for _ in range(n_pages_to_add):
            s_id = choice(available_sections)
            check_if_composable(s_id, sections, available_tokens, isc_id=isc_id)
            new_catalog_as_sections.append(s_id)

        # now we have to re-order the pages according to the structural rules
        # from the ISC's valid order info
        new_catalog_as_ordered_sections = order_catalog_sections(new_catalog_as_sections, ordered_sections)

        # turn sections into lists of tokens
        new_catalog_as_tokens = turn_sections_to_tokens(new_catalog_as_ordered_sections, sections)

        # add finished catalog to all catalogs
        n_catalogs.append(new_catalog_as_tokens)

    # return the placeholder
    return n_catalogs


def check_if_composable(section_id: str, section_definitions: dict, available_tokens: set, isc_id: str = None):
    """
    Check if a section is composable from available tokens, based on section definitons
    from the rulesets. Raise exception if not.
    """
    section_tokens = set(section_definitions[section_id])

    for st in section_tokens:
        if st not in available_tokens:
            raise (Exception(f"ERROR | Section {section_id} not composable from tokens {available_tokens} for isc {isc_id}. Check rulesets for inconsistency."))


def check_if_too_long(a_catalog: list, max_length: int, isc_id: str = None):
    current_length = len(a_catalog)
    if current_length > max_length:
        raise (Exception(f"ERROR | Current catalog length {current_length} for isc {isc_id} exceeds {max_length}."))


def is_contained(token_id: str, section_id: str, section_definitions: dict) -> bool:
    """Check if a section id's corresponding section contains the given token"""
    section_tokens = set(section_definitions[section_id])
    if token_id in section_tokens:
        return True
    return False


def order_catalog_sections(catalog_as_sections: list, valid_order: list) -> list:
    """
    Take a catalog as a list of section ids, the valid order for its ISC, return a reordered list.
    """
    catalog_copy = deepcopy(catalog_as_sections)
    catalog_reordered = []

    for relative_order, possible_sections in sorted(valid_order):

        # go through every section in the possible ones
        shuffle(possible_sections)
        for s_id in possible_sections:

            # find all instances of this section in original catalog's copy and add them to the reordered one
            all_instances_found = False
            while not all_instances_found:

                if s_id not in catalog_copy:
                    all_instances_found = True
                else:
                    catalog_reordered.append(s_id)
                    catalog_copy.remove(s_id)

    # sanity check
    assert sorted(catalog_reordered) == sorted(catalog_as_sections)

    return catalog_reordered


def turn_sections_to_tokens(catalog_as_sections: list, sections: dict) -> list:
    """Turn a list of section ids into a list of lists with token ids. Use the dictionary of section ids to token lists."""
    catalog_as_tokens = []
    for section_id in catalog_as_sections:
        section_as_tokens = sections[section_id]
        catalog_as_tokens.append(section_as_tokens)
    return catalog_as_tokens


def load_ruleset(path):
    with open(path) as f:
        r = json.load(f)
        return r


def get_catalog_length_distribution(catalogs) -> list:
    catalogs_per_length = dict()
    for c in catalogs:
        # get number of tokens in catalog
        n_tokens = 0
        for s in c:
            n_tokens += len(s)
        if n_tokens not in catalogs_per_length.keys():
            catalogs_per_length[n_tokens] = 1
        else:
            catalogs_per_length[n_tokens] += 1

    # sort
    catalogs_per_length = sorted(catalogs_per_length.items(), key=lambda x: x[1], reverse=True)
    return catalogs_per_length


def group_catalogs_by_length(catalogs: list) -> dict:
    r = dict()
    for c in catalogs:
        current_length = sum([len(s) for s in c])
        if current_length in r.keys():
            r[current_length].append(c)
        else:
            r[current_length] = [c, ]
    return r


def plot_catalog(catalog_as_tokens,
                 img_map, figsizes=(21, 2),
                 save_path=None,
                 title=None):
    """Take a catalog as tokens and plot it, potentially saving the figure with a given title."""

    # get catalog sequence, by adding section breaks
    seq = []
    for section_tokens in catalog_as_tokens:
        seq.extend(section_tokens)
        seq.append('SECTION_BREAK')

    # remove last section break
    del seq[len(seq) - 1]

    # turn to a sequence of img paths
    paths = [img_map[e] for e in seq]

    # read in the images
    images = [mpimg.imread(p) for p in paths]

    # display images
    rcParams['figure.figsize'] = figsizes
    fig, ax = plt.subplots(1, len(images))

    # adjust distance between plots horizontally
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0,
                        hspace=0)

    # hide axis, legend etc.
    for a in ax:
        a.axis('off')

    # show
    for i, a in enumerate(ax):
        a.imshow(images[i])

    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=399)

    # close all figures
    # added to prevent "RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory"
    plt.close('all')


def get_equal_datasets(rulesets: dict, n_examples_per_dl: int, set_cardinalities: list, n_tries: int = 5) -> dict:
    """
    Generate dataloaders with equal number of examples in them, each dataloader
    consisting of examples with one of the chosen input set cardinalities.
    Number of tries to generate equal dataloders is controlled, Exception
    thrown if objective not met. Increase n_tries to increase chances of equal
    dataloaders at the cost of time.
    :param rulesets: dictionary with structural and compositional rules
    :param n_examples_per_dl: how many examples should be in each dataloader
    :param set_cardinalities: length of examples, per dataloader
    :param n_tries: number of tries to generate equal dataloaders
    :return: list of len(set_cardinalities) dataloaders
    """

    # placeholder
    datasets = {k: [] for k in set_cardinalities}
    minimum_n_catalogs = n_examples_per_dl * len(set_cardinalities)
    is_done = True

    # try to create enough equal datasets
    for i in range(n_tries):

        # create minimum catalogs
        catalogs = create_catalogs(rulesets, minimum_n_catalogs)

        # group by input set cardinality (example length)
        catalogs_grouped = group_catalogs_by_length(catalogs)

        # update current datasets
        for cardinality in datasets.keys():

            # find the right examples (guard against no catalogs of this cardinality having been generated)
            if cardinality in catalogs_grouped.keys():
                examples = catalogs_grouped[cardinality]

                # add them, if needed
                if len(datasets[cardinality]) < n_examples_per_dl:
                    datasets[cardinality].extend(examples)

        # check if we have enough
        is_done = True
        for c, d in datasets.items():
            if len(d) < n_examples_per_dl:
                is_done = False
                break
        # if all criteria are met, stop
        if is_done:
            break

    # if we've done n_tries iterations and still haven't met the criterion,
    # raise an exception
    if not is_done:
        raise Exception(f"ERROR | Unable to generate equal dataloaders, desite {n_tries} tries!")

    # cut datasets down to n_examples_per_dl
    datasets = {k: examples[:n_examples_per_dl] for k, examples in datasets.items()}

    return datasets


def get_cs(catalog_as_sections_of_ints):
    r = []
    current_cluster_label = 0
    for s in catalog_as_sections_of_ints:
        for i in s:
            r.append(current_cluster_label)
        current_cluster_label += 1
    return np.asarray(r)


def get_clusters(a_cs):
    """Take cluster labels, return both-side zero padded (N+2) cardinalities per cluster"""
    N = len(a_cs)
    clusters = np.zeros(N)
    for c in a_cs:
        clusters[c] += 1

    # pad on both sides
    clusters = np.pad(clusters, (1, 1), 'constant', constant_values=(0, 0))
    return clusters


def get_k(a_cs):
    """Get num clusters (K)"""
    return len(set(a_cs))


def get_flat_catalog_as_int(catalog_as_list_of_ints):
    flat_c = []
    for s in catalog_as_list_of_ints:
        flat_c.extend(s)
    flat_c = np.expand_dims(np.asarray(flat_c), 0)
    return flat_c


def get_shuffled_cs_and_data(a_cs, a_data):
    """Shuffle cs and data according to the same """
    N = len(a_cs)
    arr = np.arange(N)
    np.random.shuffle(arr)
    a_cs = a_cs[arr]
    a_data = a_data[:, arr]
    return a_cs, a_data


def get_y_ci_and_y_att(a_cs_ordered, a_cs_relabelled, num_clusters, batch_size=1):
    y_ci = torch.zeros(num_clusters)
    y_att = torch.zeros((num_clusters, num_clusters))
    o2r = {o: a_cs_relabelled[i] for i, o in enumerate(a_cs_ordered)}
    for i in range(num_clusters):
        proper_cluster_index_for_ith_cluster = o2r[i]
        y_ci[i] = proper_cluster_index_for_ith_cluster
        y_att[i][proper_cluster_index_for_ith_cluster] = 1

    # repeat it for the entire batch
    Y_ci = torch.tile(y_ci, (batch_size, 1))
    Y_att = torch.tile(y_att, (batch_size, 1, 1))

    return Y_ci, Y_att


def tokens_to_integers(catalog_as_tokens: list, token_to_int_dict: dict):
    r = []
    for s_tok in catalog_as_tokens:
        s_int = [token_to_int_dict[tok][1] for tok in s_tok]  # the token integer is second in the value tuple in ruleset
        r.append(s_int)
    return r


def integers_to_tokens(catalog_as_integers:list , integer_to_token_map: dict):
    r = []
    for s in catalog_as_integers:
        s_tok = []
        for e in s:
            s_tok.append(integer_to_token_map[e])
        r.append(s_tok)
    return r


def get_xy_synthetic_clustered(catalog_as_tokens, t2i, debug=False):
    """Turn a catalogs as list of lists (sections) of tokens into X and Y data fror clustering"""
    c_as_int = tokens_to_integers(catalog_as_tokens, t2i)

    # get cs
    cs = get_cs(c_as_int)

    # get clusters
    clusters = get_clusters(cs)

    # get K
    K = get_k(cs)

    # we now have to randomly shuffle
    data = get_flat_catalog_as_int(c_as_int)
    cs, data = get_shuffled_cs_and_data(cs, data)

    # relabel
    cs_relabelled = relabel(cs)
    cs_ordered = cs

    # get y_ci
    Y_ci, Y_att = get_y_ci_and_y_att(cs_ordered, cs_relabelled, K, batch_size=1)

    return data, cs_relabelled, clusters, K, cs_ordered, Y_ci, Y_att


def to_datasets(cardinality_data_dict):
    """Take data split by cardinality and merge it into datasets"""
    dsets = []

    # unpack the data for each cardinality
    for cardinality, raw_data in cardinality_data_dict.items():

        # guard against empty cardinalities
        if len(raw_data['K']) > 0:
            data = np.concatenate(raw_data['data'], axis=0)
            cs_relabelled = np.stack(raw_data['cs_relabelled'], axis=0)
            clusters = np.stack(raw_data['clusters'], axis=0)
            K = raw_data['K']
            cs_ordered = np.stack(raw_data['cs_ordered'], axis=0)
            Y_ci = torch.concat(raw_data['Y_ci'], axis=0)
            Y_att = torch.concat(raw_data['Y_att'], axis=0)

            # instantiate a Synthetic Dataset using this data
            dset = SyntheticStructureDataset(data, cs_relabelled, clusters, K, cs_ordered, Y_ci, Y_att)
            dsets.append(dset)

    return dsets


def get_dataloaders(rulesets: dict, n_examples_per_dl: int, min_input_cardinality: int, max_input_cardinality: int,
                    batch_size: int = 64, section_padding_int: int = -999, num_workers=0) -> list:
    """
    Take a ruleset, the target number of examples per dataloader, cardinalities range per dataloader and batch size,
    generate data, return it as a list of dataloaders, with padding for Y_ci and Y_att due to varying number of
    target sections.
    :param rulesets: a loaded json dict with structural & compositional rules & more
    :param n_examples_per_dl: number of examples of the same cardinality / input length to generate
    :param min_input_cardinality: minimum input set cardinality (example length) [inclusive]
    :param max_input_cardinality: maximum input set cardinality (example length) [inclusive]
    :param batch_size: number of examples per batch
    :param section_padding_int: what integer to pad Y_ci and Y_att due to varying section numbers per example
    :return: a list of dataloaders with batches of examples of different lengths
    """
    # create cardinalities as list
    set_cardinalities = [integer for integer in range(min_input_cardinality, max_input_cardinality + 1)]

    # calculate max possible sections count
    max_sections = max([isc["num_sections_max"] for _, isc in rulesets["input_set_compositions"].items()])

    # generate examples, split per cardinality
    examples_by_cardinality = get_equal_datasets(rulesets, n_examples_per_dl, set_cardinalities)

    # create placeholder for raw data
    data_by_cardinality = {c: {
        'data': [], 'cs_relabelled': [], 'clusters': [], 'K': [], 'cs_ordered': [], 'Y_ci': [], 'Y_att': []
    } for c in range(min(examples_by_cardinality), max(examples_by_cardinality) + 1)}

    # turn every example of raw data (as token lists) into their integer representation, flattened
    for cardinality, catalogs_as_tokens in examples_by_cardinality.items():
        for c_as_t in catalogs_as_tokens:
            data, cs_relabelled, clusters, K, cs_ordered, Y_ci, Y_att = get_xy_synthetic_clustered(c_as_t, rulesets['tokens'])

            # update placeholders
            data_by_cardinality[cardinality]['data'].append(data)
            data_by_cardinality[cardinality]['cs_relabelled'].append(cs_relabelled)
            data_by_cardinality[cardinality]['clusters'].append(clusters)
            data_by_cardinality[cardinality]['K'].append(K)
            data_by_cardinality[cardinality]['cs_ordered'].append(cs_ordered)

            # we have to pad, since we have a varying number of sections
            n_sections_current = Y_ci.size(1)
            to_pad = max_sections - n_sections_current
            Y_ci_padded = F.pad(Y_ci, (0, to_pad), "constant", section_padding_int)
            Y_att_padded = F.pad(Y_att, (0, to_pad, 0, to_pad), "constant", section_padding_int)

            # continue with padded Y for set to sequence
            data_by_cardinality[cardinality]['Y_ci'].append(Y_ci_padded)
            data_by_cardinality[cardinality]['Y_att'].append(Y_att_padded)

    # torch datasets
    datasets = to_datasets(data_by_cardinality)

    # torch dataloaders
    dataloaders = [DataLoader(d, batch_size=batch_size, num_workers=num_workers) for d in datasets]

    return dataloaders


def prediction_to_integers(x: torch.Tensor, cluster_labels: torch.Tensor, cluster_order: torch.Tensor,
                           section_padding_int: int =-999):
    catalog_as_integers = []

    # remove unnecessary dimension
    x = x.squeeze(0)

    # start by turning element cluster labels into ordered ones
    cluster_order = cluster_order.squeeze(0).int().tolist()
    cluster_labels = cluster_labels.squeeze(0).int().tolist()

    # remove padding from cluster order
    cluster_order = [e for e in cluster_order if e != section_padding_int]

    cluster_label_to_order_assignment = {v: i for i, v in enumerate(cluster_order)}

    # change labels into ordered ones
    cluster_labels_in_order = [cluster_label_to_order_assignment[e] for e in cluster_labels]

    # group offers into sections, in order
    sections = {k: [] for k in cluster_label_to_order_assignment.keys()}
    for i, e in enumerate(x):
        # find element's section index
        current_element_section_idx = cluster_labels_in_order[i]

        # add the current element to its section
        sections[current_element_section_idx].append(int(e))

    # reconstruct a full catalog as indices from ordered groups
    for i in range(len(sections)):
        catalog_as_integers.append(sections[i])

    return catalog_as_integers


def identify_isc(catalog_as_tokens, input_set_compositions):
    """Take a catalog, return its ISC identifier string"""

    # flatten
    catalog_flat = [t for s in catalog_as_tokens for t in s]

    # turn to set
    catalog_set = set(catalog_flat)

    # identify the isc
    for isc_id, isc in input_set_compositions.items():
        if catalog_set == set(isc['token_set']):
            return isc_id
    return None


def identify_section(section_as_tokens: list, valid_sections: dict, invalid_id: str='INVALID_SECTION') -> str:
    """Take a section and valid sections' dict, return matching section id string or unknown."""
    # in-section order of tokens doesn't matter, just their relative numbers
    section_as_tokens = deepcopy(section_as_tokens)  # otherwise we update the valid_sections too
    counter_current = Counter(section_as_tokens)
    for s_id, valid_section in valid_sections.items():
        counter_candidate = Counter(valid_section)
        if counter_current == counter_candidate:
            return s_id
    return invalid_id


def get_catalog_metrics(catalogs_as_tokens: list, rulesets: dict) -> dict:
    """
    Take a list of catalogs as lists of token strings and a rulesets dictionary,
    report metrics on how well the catalogs adhere to these rules.
    """
    invalid_section_id = 'INVALID_SECTION'
    invalid_isc_id = 'INVALID_ISC'

    # we only need the input set compositions
    iscs = rulesets['input_set_compositions']
    sections = rulesets['sections']
    n_catalogs = len(catalogs_as_tokens)

    # placeholder metrics
    metrics = dict(isc_ids=[],
                   composition=[[] for _ in range(n_catalogs)],
                   structure=[[] for _ in range(n_catalogs)],
                   num_sections=[])

    # iterate over all catalogs, updating metrics
    for n, c in enumerate(catalogs_as_tokens):

        # take a catalog and identify the ISC
        current_isc = identify_isc(c, iscs)

        # 1. ISC Validity
        # might need to guard against unidentified isc
        if not current_isc:
            raise(Exception(f"ERROR | Invalid ISC found in catalog: {c} \nfor rulesets {rulesets}"))
        else:
            metrics['isc_ids'].append(current_isc)

        # keep track of identified sections for structural metrics later
        catalog_as_sections = []

        # 2. Compositional Metrics
        for s in c:
            # identify the section id
            s_id = identify_section(s, sections, invalid_section_id)

            # check if it's in valid and update the catalog as sections
            if s_id == invalid_section_id:
                metrics['composition'][n].append(0)
            else:
                metrics['composition'][n].append(1)

            # append it to the tracker for later structural / ordering
            catalog_as_sections.append(s_id)

        # 3. Structural Metrics
        # get the valid order for this isc
        ordered_sections_groups = [sections for _, sections in iscs[current_isc]['valid_order'].items()]
        max_group_index = len(ordered_sections_groups) - 1
        current_sections_group_idx = 0

        for i, s_id in enumerate(catalog_as_sections):

            # first one has to be right
            if i == 0:
                if s_id in ordered_sections_groups[current_sections_group_idx]:
                    metrics['structure'][n].append(1)
                else:
                    metrics['structure'][n].append(0)
                    if current_sections_group_idx < max_group_index:
                        current_sections_group_idx += 1

            # next one might already belong to the next order group
            else:
                if s_id in ordered_sections_groups[current_sections_group_idx]:
                    metrics['structure'][n].append(1)
                elif current_sections_group_idx < max_group_index:
                    if s_id in ordered_sections_groups[current_sections_group_idx + 1]:
                        metrics['structure'][n].append(1)
                        if current_sections_group_idx < max_group_index:
                            current_sections_group_idx += 1
                else:
                    metrics['structure'][n].append(0)
                    if current_sections_group_idx < max_group_index:
                        current_sections_group_idx += 1

        # 4. Num Sections
        if len(catalog_as_sections) <= iscs[current_isc]['num_sections_max']:
            metrics['num_sections'].append(1)
        else:
            metrics['num_sections'].append(0)

    # aggregate
    raw_metrics = metrics
    agg_metrics = aggregate_metrics(raw_metrics)

    return agg_metrics


def aggregate_metrics(raw_metrics):
    """Take raw metrics per catalog and aggregate them per each ISC and jointly"""
    # construct placeholder for aggregated metrics for each ISC and jointly
    am = dict(ISC_ALL=dict(composition=0, structure=0, num_sections=0))
    for isc_id in set(raw_metrics['isc_ids']):
        am[isc_id] = dict(composition=0, structure=0, num_sections=0)

    # update entries for total and individual ISC
    # per catalog
    n_catalogs = len(raw_metrics['isc_ids'])
    for i in range(n_catalogs):

        # current isc
        isc_id = raw_metrics['isc_ids'][i]

        # current composition score
        composition_scores = raw_metrics['composition'][i]
        composition_score = sum(composition_scores) / len(composition_scores)

        # current structure score
        structure_scores = raw_metrics['structure'][i]
        structure_score = sum(structure_scores) / len(structure_scores)

        # num sections
        num_sections_score = raw_metrics['num_sections'][i]

        # update the trackers for current ISC
        am[isc_id]['composition'] += composition_score
        am[isc_id]['structure'] += structure_score
        am[isc_id]['num_sections'] += num_sections_score

        # update trackers for all ISCs jointly
        am['ISC_ALL']['composition'] += composition_score
        am['ISC_ALL']['structure'] += structure_score
        am['ISC_ALL']['num_sections'] += num_sections_score

    # track number of catalogs per isc
    catalogs_per_isc = Counter(raw_metrics['isc_ids'])

    # aggregate (average)
    for isc_id in am:

        # find n catalogs for that isc
        if isc_id == 'ISC_ALL':
            n_relevant_catalogs = sum([counts for counts in catalogs_per_isc.values()])
        else:
            n_relevant_catalogs = catalogs_per_isc[isc_id]

        # divide the summed score to get averages for each metrics
        for k, v in am[isc_id].items():
            am[isc_id][k] = v / n_relevant_catalogs

    # include total counts in the metrics
    am['CATALOGS_PER_ISC'] = catalogs_per_isc

    # return
    return am


def get_model_basic_metrics(a_model: object, dataloaders: list, rulesets: dict, n_examples: int, n_samples: int, logger: object,
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
            logger.info('Basic metrics calculation for example {} / {} ...'.format(i+1, n_examples))

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


def get_model_functional_metrics(a_model: object, dataloaders: list, rulesets: dict, n_examples: int, n_samples: int, logger: object) -> dict:
    """
    Take all objects needed to get n_examples predicted on, return full functional metrics
    :param a_model: e.g. model_fnoc_batched.FNOC
    :param n_examples: how many examples to make predictions on
    :param dataloaders: list of dataloaders, from which to draw n_examples
    :param rulesets: loaded json dictionary with structural & compositional rules for catalog creation
    :param n_samples: how many samples to generate during inference
    :param logger: a logger object
    :return: dict with full scores
    """
    predicted_catalogs_as_tokens = []

    logger.info('Beginning the inference for functional metrics ...')
    for i in range(n_examples):

        if i % 10 == 0:
            logger.info('Functional metrics calculation for example {} / {} ...'.format(i+1, n_examples))

        # get example
        x, cs, clusters, num_clusters, cs_ordered, Y_att, Y_ci = get_batch(dataloaders,
                                                                           single_example_batch=True)  # replace None with manual_clusters

        # predict
        y_hat_cs, y_hat_co = a_model.infer(x, n_samples=n_samples)

        # turn to sequence of element indices from dict
        predicted_c_as_i = prediction_to_integers(torch.from_numpy(x),
                                                  torch.from_numpy(y_hat_cs).unsqueeze(0),
                                                  y_hat_co)

        # turn to tokens
        int_to_token = {int_tup[1]: tok for tok, int_tup in rulesets['tokens'].items()}
        predicted_c_as_tokens = integers_to_tokens(predicted_c_as_i, int_to_token)

        # append
        predicted_catalogs_as_tokens.append(predicted_c_as_tokens)

    # get metrics
    logger.info('Obtaining functional catalog metrics from predicted catalogs ...')
    metrics = get_catalog_metrics(predicted_catalogs_as_tokens, rulesets)

    return metrics


def aggregate_functional_metrics(metric_scores: dict):
    """Take a metrics dictionary, calculate aggregated totals (weighted by number of catalogs per ISC)
    :param metric_scores: dictionary of metrics, per isc
    :return: a dictionary with totals"""

    total_catalog_count = 0
    total_composition_score = 0
    total_structure_score = 0
    for isc_id, catalog_count in metric_scores['CATALOGS_PER_ISC'].items():
        total_composition_score += metric_scores[isc_id]['composition'] * catalog_count
        total_structure_score += metric_scores[isc_id]['structure'] * catalog_count
        total_catalog_count += catalog_count

    # reduce
    total_composition_score /= total_catalog_count
    total_structure_score /= total_catalog_count

    return {'TOTAL_COMPOSITION_SCORE': total_composition_score,
            'TOTAL_STRUCTURE_SCORE': total_structure_score}


def get_parameters_synthetic(generative_model, arguments):
    params = {}

    if generative_model == 'SyntheticStructures':
        params['vocab_size'] = len(arguments['rulesets']['tokens'])
        params['add_set_repr_to_elems'] = arguments['add_set_repr_to_elems']
        params['generative_model'] = 'SyntheticStructures'

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


if __name__ == '__main__':
    # visuals
    path_visuals = os.path.join(os.getcwd(), 'visual')

    red = os.path.join(path_visuals, 'red_offer.png')
    blue = os.path.join(path_visuals, 'blue_offer.png')
    yellow = os.path.join(path_visuals, 'yellow_offer.png')
    green = os.path.join(path_visuals, 'green_offer.png')
    purple = os.path.join(path_visuals, 'purple_offer.png')
    section_break = os.path.join(path_visuals, 'page_break.png')

    # map human-readable catalog elem names to img paths
    img_path_map = {
        "T_001": red,
        "T_002": yellow,
        "T_003": blue,
        "T_004": green,
        "T_005": purple,
        "SECTION_BREAK": section_break
    }

    # config
    cfg = dict()
    cfg['batch_size'] = 64
    cfg['n_examples_per_dataloaders'] = 10
    cfg['section_padding_int'] = -999
    cfg['min_cardinality'] = 35
    cfg['max_cardinality'] = 50
    cfg['num_workers'] = 0
    cfg['seed'] = 10
    cfg['rulesets_path'] = os.path.join('run_configs', 'synthetic_rulesets.json')

    # seed
    seed(cfg['seed'])

    # load chosen rulesets config
    rulesets = load_ruleset(cfg['rulesets_path'])

    # generate data
    catalogs = create_catalogs(rulesets, 1000)

    # get metrics (predicted catalogs get turned into this form prior too)
    metrics = get_catalog_metrics(catalogs, rulesets)
    print(metrics)

    # show
    n_plots = 10
    for i in range(n_plots):
        c_as_tokens = choice(catalogs)
        plot_catalog(c_as_tokens, img_path_map, save_path=f'figures/SyntheticStructures/test_{i+1}')

    # count catalog lengths
    print(get_catalog_length_distribution(catalogs))

    # generate data
    dataloaders = get_dataloaders(rulesets,
                                  n_examples_per_dl=cfg['n_examples_per_dataloaders'],
                                  min_input_cardinality=cfg['min_cardinality'],
                                  max_input_cardinality=cfg['max_cardinality'],
                                  batch_size=cfg['batch_size'],
                                  section_padding_int=cfg['section_padding_int'],
                                  num_workers=cfg['num_workers']
                                  )

    # experiment on random batch
    import random
    b = next(iter(random.choice(dataloaders)))
    x = b['data'][0].unsqueeze(0)
    cs = b['cs_relabelled'][0].unsqueeze(0)
    co = b['Y_ci'][0].unsqueeze(0)

    # we need to construct an int_to_token map from the ruleset
    i2t = i2t = {int_tup[1]: tok for tok, int_tup in rulesets['tokens'].items()}
    c_as_int = prediction_to_integers(x, cs, co)
    print(c_as_int)



