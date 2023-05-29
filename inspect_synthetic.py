import json
import logging
import os
import torch

from data_synthetic import get_parameters_synthetic, load_ruleset, get_dataloaders, \
    get_batch, prediction_to_integers, integers_to_tokens, plot_catalog
from utils import get_logger

# configure
model_name = 'FNOC_5911180726_700_best_over_all_comp_0.34640_struct_0.45200'  # no .pt
add_set_repr_to_elems = True  # has to match how model was trained
rulesets_path = 'run_configs/synthetic_rulesets.json'
batch_size = 64
n_examples_per_dataloaders = 10
section_padding_int = -999
min_cardinality = 35
max_cardinality = 50
n_inference_samples = 100

# load
model_type = model_name[:4]  # ['CNOC', 'ENOC', 'FNOC"]
model = torch.load(f'saved_models/SyntheticStructures/{model_name}.pt',
                   map_location=torch.device('cpu'))

# gpu models need this for sampler to work
model.device = torch.device('cpu')
model.params['device'] = torch.device('cpu')

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

# data
log = get_logger(path='logs/inspect_synthetic_discardable', level=logging.WARNING)
rulesets = load_ruleset(rulesets_path)
args = dict(
    model=model_type,
    add_set_repr_to_elems=add_set_repr_to_elems,
    rulesets_path=rulesets_path,
    rulesets=rulesets,
    cardinality_conditioned_on_assigned=model.condition_cardinality_on_assigned,
    batch_size=batch_size,
    n_examples_per_dataloaders=n_examples_per_dataloaders,
    section_padding_int=section_padding_int,
    min_cardinality=min_cardinality,
    max_cardinality=max_cardinality

)
params = get_parameters_synthetic('SyntheticStructures', args)

# integer-to-token map from ruleset
i2t = {int_tup[1]: tok for tok, int_tup in rulesets['tokens'].items()}

# generate dataloaders
test_dataloaders = get_dataloaders(rulesets,
                                   n_examples_per_dl=args['n_examples_per_dataloaders'],
                                   min_input_cardinality=args['min_cardinality'],
                                   max_input_cardinality=args['max_cardinality'],
                                   batch_size=args['batch_size'],
                                   section_padding_int=args['section_padding_int'],
                                   num_workers=0)
# metrics [optional]
from data_synthetic import get_model_basic_metrics
metrics_basic = get_model_basic_metrics(a_model=model,
                                        dataloaders=test_dataloaders,
                                        rulesets=rulesets,
                                        n_examples=10,
                                        n_samples=10,
                                        logger=log)
cs_score = metrics_basic['CLUSTERING_SCORE_AVG']
co_score = metrics_basic['ORDERING_SCORE_AVG']

for i in range(20):
    # example
    x, cs, clusters, num_clusters, cs_ordered, _, Y_ci = get_batch(test_dataloaders, single_example_batch=True)

    # # single inference
    y_hat_cs, y_hat_co = model.infer(x, n_samples=n_inference_samples)

    # turn to sequence of element indices from dict
    predicted_c_as_i = prediction_to_integers(torch.from_numpy(x),
                                              torch.from_numpy(y_hat_cs).unsqueeze(0),
                                              y_hat_co)
    predicted_c_as_t = integers_to_tokens(predicted_c_as_i, integer_to_token_map=i2t)

    # show and save image
    fig_name = f'./figures/SyntheticStructures/inspect_{model_name}_{i + 1}.pdf'
    log.info(f'Saving figure {i + 1} at {fig_name}...')
    plot_catalog(predicted_c_as_t, img_path_map, save_path=fig_name)
    print()
