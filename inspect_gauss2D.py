import torch

from data_gauss2D import get_generator
from show import show_all
from utils import get_parameters

# configure
max_cardinality = 30
model_name = 'FNOC_2373723707_mc_30_2800_best_clusters_cl_0.55113_or_0.19707'
model_type = model_name[:4]  # ['CNOC', 'ENOC']

# load
model = torch.load(f'saved_models/Gauss2D/{model_name}.pt',
                   map_location=torch.device('cpu'))

# gpu models need this for sampler to work
model.device = torch.device('cpu')
model.params['device'] = torch.device('cpu')

# data
args = dict(
    max_cardinality=max_cardinality,
    model=model_type,
    cardinality_conditioned_on_assigned=True  # optional for certain FNOC models
)
params = get_parameters('Gauss2D', args)
data_generator = get_generator(params)

for i in range(20):
    # example
    x, cs, clusters, num_clusters, cs_ordered, _, Y_ci = data_generator.generate(None, batch_size=1)

    # single inference
    y_hat_cs, y_hat_co = model.infer(x, n_samples=100)

    # show
    show_all(x, cs_ordered, y_hat_cs, title=f'Neural Ordered Clusters with Max Cardinality n={max_cardinality}, {i + 1}',
             save_path=f'figures/Gauss2D/scratch_max_cardinality_{max_cardinality}_{model_name}_{i + 1}.pdf',
             s=5, marker='o', show_cardinalities=True,
             fixed_scale=False)
    print()
