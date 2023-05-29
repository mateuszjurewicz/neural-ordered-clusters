import argparse
import torch
from data_synthetic import load_ruleset, get_dataloaders, prediction_to_integers, integers_to_tokens, \
    plot_catalog, get_parameters_synthetic, get_batch, get_model_basic_metrics, \
    get_model_functional_metrics, aggregate_functional_metrics
from model_noc import NOC, get_NOC_synth_encoder
from utils import get_logger, get_run_id, count_params
import random
import numpy as np
import os
import time


def main(args):

    # logs
    log = get_logger(path=args.logs_path, run_id=args.run_id)
    log.info(f'Starting run {args.run_id}')

    # log args
    log.info(f'Arguments: {args}')

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  # random is used to select the dataloader

    # additional tracking of data params
    train_dloader_name = 'seed_{}_ncat_{}_minc_{}_maxc_{}_spad_{}'.format(args.seed, args.n_examples_per_train_dataloaders,
                                                                          args.min_cardinality, args.max_cardinality, args.section_padding_int)
    test_dloader_name = 'seed_{}_ncat_{}_minc_{}_maxc_{}_spad_{}'.format(args.seed, args.n_examples_per_test_dataloaders,
                                                                         args.min_cardinality, args.max_cardinality, args.section_padding_int)

    # get data
    if args.generate_new_data:
        log.info('Generating train dataloaders...')
        train_dataloaders = get_dataloaders(args.rulesets,
                                            n_examples_per_dl=args.n_examples_per_train_dataloaders,
                                            min_input_cardinality=args.min_cardinality,
                                            max_input_cardinality=args.max_cardinality,
                                            batch_size=args.batch_size,
                                            section_padding_int=args.section_padding_int,
                                            num_workers=args.num_workers
                                            )
        # persist
        log.info('Persisting train dataloaders...')
        for i, td in enumerate(train_dataloaders):
            torch.save(td, f'data/synthetic/train_dataloader_{train_dloader_name}_card_{args.min_cardinality + i}.pth')

        log.info('Generating test dataloaders...')
        test_dataloaders = get_dataloaders(args.rulesets,
                                           n_examples_per_dl=args.n_examples_per_test_dataloaders,
                                           min_input_cardinality=args.min_cardinality,
                                           max_input_cardinality=args.max_cardinality,
                                           batch_size=args.batch_size,
                                           section_padding_int=args.section_padding_int,
                                           num_workers=args.num_workers
                                           )
        log.info('Persisting test dataloaders...')
        for i, td in enumerate(test_dataloaders):
            torch.save(td, f'data/synthetic/test_dataloader_{test_dloader_name}_card_{args.min_cardinality + i}.pth')

    # reload
    log.info('Reloading train dataloaders...')
    train_dataloaders = []
    for i in range(args.min_cardinality, args.max_cardinality + 1):
        td = torch.load(f'data/synthetic/train_dataloader_{train_dloader_name}_card_{i}.pth')
        train_dataloaders.append(td)
    log.info('Reloading test dataloaders...')
    test_dataloaders = []
    for i in range(args.min_cardinality, args.max_cardinality + 1):
        td = torch.load(f'data/synthetic/test_dataloader_{test_dloader_name}_card_{i}.pth')
        test_dataloaders.append(td)

    # check dataloader
    batch = next(iter(random.choice(train_dataloaders)))
    log.info(
        f'Checking random train dataloader...'
        f'\ndata: {batch["data"].shape}'
        f'\ncs_relabelled: {batch["cs_relabelled"].shape}'
        f'\nclusters: {batch["clusters"].shape}'
        f'\nK: {len(batch["K"])}'
        f'\ncs_ordered: {batch["cs_ordered"].shape}'
        f'\nY_ci: {batch["Y_ci"].size()}'
        f'\nY_att: {batch["Y_att"].size()}'
    )

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

    # inspect a single example
    log.info('Checking one example for the dataloaders visually...')
    x = batch['data'][0].unsqueeze(0)
    cs = batch['cs_relabelled'][0].unsqueeze(0)
    co = batch['Y_ci'][0].unsqueeze(0)

    # turn to a list of integers
    catalog_as_integers = prediction_to_integers(x, cs, co)

    # turn into a list of tokens (need a map from rulesets)
    i2t = {int_tup[1]: tok for tok, int_tup in args.rulesets['tokens'].items()}
    catalog_as_tokens = integers_to_tokens(catalog_as_integers, i2t)
    log.info(f'Single example as integers: {catalog_as_integers}')
    log.info(f'Single example as tokens:   {catalog_as_tokens}')

    # plot a single example from a randomly chosen dataloader
    test_save_path = 'figures/SyntheticStructures/test'
    log.info(f'Plotting one example for the dataloaders at {test_save_path} ...')
    plot_catalog(catalog_as_tokens, img_path_map, save_path=test_save_path)

    # unpack params
    max_it = args.iterations
    params = get_parameters_synthetic('SyntheticStructures', vars(args))
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")
    log.info('Device: {}'.format(str(params['device'])))
    end_name = args.model + '_' + str(args.run_id)

    model = None
    if args.model == 'NOC':
        encoder = get_NOC_synth_encoder(params)
        model = NOC(params, encoder).to(params['device'])

    # log params
    n_params_report = count_params(model, return_string=True)
    log.info(n_params_report[1])

    # add directiories for saved models ...
    if not os.path.isdir(f'saved_models/SyntheticStructures'):
        os.makedirs(f'saved_models/SyntheticStructures')
    # ... and figures
    if not os.path.isdir(f'figures/SyntheticStructures'):
        os.makedirs(f'figures/SyntheticStructures')

    # define containers to collect statistics - -   §   `§
    ordering_criterion = torch.nn.CrossEntropyLoss()
    losses_clustering = []  # NLLs
    losses_ordering = []
    losses_cardinality = []
    losses_final = []
    accs_or_elbos = []  # Accuracy or elbo of the classification prediction

    it = 0  # iteration counter

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate_initial, weight_decay=args.weight_decay)

    # set model to train, clear grads
    model.train()
    optimizer.zero_grad()

    # track best model scores and save those models
    best_composition_score = 0
    best_structure_score = 0

    # main loop
    while True:
        t_start = time.time()
        it += 1

        if it == max_it:
            break

        data, labels, cardinalities, K, labels_ordered, _, Y_ci = get_batch(train_dataloaders, single_example_batch=False)
        N = data.shape[1]

        # get clustering loss
        loss_training, acc_or_elbo = model.train_clustering_loss_masked(data, labels)  # masked for different label per example in batch

        # unpack NOC losses
        if args.model == 'NOC':
            loss_clustering = loss_training['loss_clustering']
            loss_cardinality = loss_training['loss_cardinality']
        else:
            loss_clustering = loss_training

        # get ordering loss
        loss_ordering = model.train_ordering_loss_masked(data, labels, Y_ci, ordering_criterion)

        # apply any multiplication factors
        loss_ordering = loss_ordering * args.ordering_loss_multiplier

        # combine losses
        loss_final = loss_clustering + loss_ordering

        # loss cardinality (NOC only)
        if args.model == 'NOC':
            loss_cardinality = loss_cardinality * args.cardinality_loss_multiplier
            loss_final += loss_cardinality

        # backwards step
        loss_final.backward()

        # update trackers
        losses_clustering.append(loss_clustering.item())
        losses_ordering.append(loss_ordering.item())
        losses_final.append(loss_final.item())
        accs_or_elbos.append(acc_or_elbo.item())  # variable batch size

        # NOC loss tracking
        if args.model == 'NOC':
            losses_cardinality.append(loss_cardinality.item())

        optimizer.step()
        optimizer.zero_grad()

        # report per example
        if args.model == 'ENOC':
            log.info(
                '{0:4d}  N:{1:2d}  K:{2}  Clustering Loss:{3:.3f}  Ordering Loss:{4:.3f}  Total Loss:{5:.3f}  Mean Acc/Elbo:{6:.3f}  Mean Time/Iteration: {7:.1f}' \
                    .format(it, N, set(K.tolist()), np.mean(losses_clustering[-50:]), np.mean(losses_ordering[-50:]), np.mean(losses_final[-50:]),
                            np.mean(accs_or_elbos[-50:]), (time.time() - t_start)))
        elif args.model == 'CNOC':
            log.info('{0:4d}  N:{1:2d}  K:{2}  Clustering Loss:{3:.3f}  Ordering Loss:{4:.3f}  Total Loss:{5:.3f}  Mean Time/Iteration: {6:.1f}' \
                     .format(it, N, set(K.tolist()), np.mean(losses_clustering[-50:]), np.mean(losses_ordering[-50:]), np.mean(losses_final[-50:]),
                             (time.time() - t_start)))
        elif args.model == 'NOC':
            log.info(
                '{0:4d}  N:{1:2d}  K:{2}  Clustering Loss:{3:.3f}  Ordering Loss:{4:.3f}  Cardinality Loss: {5:.3f} Total Loss:{6:.3f}  Mean Time/Iteration: {7:.1f}' \
                    .format(it, N, set(K.tolist()), np.mean(losses_clustering[-50:]), np.mean(losses_ordering[-50:]), np.mean(losses_cardinality[-50:]),
                            np.mean(losses_final[-50:]), (time.time() - t_start)))

        if it % args.save_every == 0 and it > 0:
            if 'fname' in vars():
                os.remove(fname)
            model.params['it'] = it
            fname = 'saved_models/' + 'SyntheticStructures' + '/' + end_name + '_' + str(it) + '.pt'
            log.info(f'Saving model at args.save_every, {fname}')
            torch.save(model, fname)

        if it % args.validate_every == 0 and it > 0:

            torch.cuda.empty_cache()
            model.eval()

            # 1. Generate 5 examples and show their predictions
            log.info('Visualizing 5 predictions...')
            for i in range(1):
                # get example
                x, cs, clusters, num_clusters, cs_ordered, Y_att, Y_ci = get_batch(train_dataloaders,
                                                                                   single_example_batch=True)  # replace None with manual_clusters

                # predict
                y_hat_cs, y_hat_co = model.infer(x, n_samples=args.inference_n_samples)

                # turn to sequence of element indices from dict
                predicted_c_as_i = prediction_to_integers(torch.from_numpy(x),
                                                          torch.from_numpy(y_hat_cs).unsqueeze(0),
                                                          y_hat_co)

                # turn to tokens
                predicted_c_as_tokens = integers_to_tokens(predicted_c_as_i, i2t)

                # show and save image
                fig_name = f'./figures/SyntheticStructures/{args.run_id}_samples_{it:04d}' + '_' + str(i + 1) + '.pdf'
                log.info(f'Saving figure {i + 1} at {fig_name}...')
                plot_catalog(predicted_c_as_tokens, img_path_map, save_path=fig_name, title=f'Run {run_id} Iteration {it}')

            # 2. Basic Metrics
            log.info(f'Obtaining basic metrics for {args.examples_per_metrics} examples,'
                     f' with {args.inference_n_samples} samples each...')
            metrics_basic = get_model_basic_metrics(a_model=model,
                                                    dataloaders=train_dataloaders,
                                                    rulesets=args.rulesets,
                                                    n_examples=args.examples_per_metrics,
                                                    n_samples=args.inference_n_samples,
                                                    logger=log)
            cs_score = metrics_basic['CLUSTERING_SCORE_AVG']
            co_score = metrics_basic['ORDERING_SCORE_AVG']

            # cs_score, co_score = get_metrics_synthetic(model, test_dataloaders, log, m_examples=args.examples_per_metrics)
            log.info('Metrics: Clustering Score: {:.5f}  |  Ordering Score: {:.5f}'.format(cs_score, co_score))

            # 3. Functional Metrics
            log.info(f'Obtaining functional metrics for {args.examples_per_metrics} examples,'
                     f' with {args.inference_n_samples} samples each...')
            metrics_functional = get_model_functional_metrics(a_model=model,
                                                              dataloaders=train_dataloaders,
                                                              rulesets=args.rulesets,
                                                              n_examples=args.examples_per_metrics,
                                                              n_samples=args.inference_n_samples,
                                                              logger=log)
            # extract total scores
            log.info('Combining total metrics for composition and structure ...')
            metrics_functional_aggregated = aggregate_functional_metrics(metrics_functional)
            log.info(f'Functional results (full): {metrics_functional_aggregated}')

            # unpack
            comp_score = metrics_functional_aggregated['TOTAL_COMPOSITION_SCORE']
            struct_score = metrics_functional_aggregated['TOTAL_STRUCTURE_SCORE']
            log.info(f'Total functional composition score: {comp_score}')
            log.info(f'Total functional structure score: {struct_score}')

            # 6. Save the models if their performance is better than previous ones
            if (comp_score + struct_score) > (best_composition_score + best_structure_score):
                # have to guard against first time the metrics are obtained and there's no previous model to remove
                if 'best_over_all_path' in vars():
                    os.remove(best_over_all_path)
                best_over_all_path = 'saved_models/' + 'SyntheticStructures' + '/' + end_name + '_' + str(it) + '_' \
                                     + 'best_over_all' + '_comp_' + '{:.5f}'.format(comp_score) + '_struct_' + '{:.5f}'.format(struct_score) + '.pt'
                log.info(f'Saving best overall model as {best_over_all_path}')
                torch.save(model, best_over_all_path)

            if comp_score > best_composition_score:
                if 'best_composition_path' in vars():
                    os.remove(best_composition_path)
                best_composition_path = 'saved_models/' + 'SyntheticStructures' + '/' + end_name + '_' + str(it) + '_' \
                                     + 'best_composition' + '_comp_' + '{:.5f}'.format(comp_score) + '_struct_' + '{:.5f}'.format(struct_score) + '.pt'
                log.info(f'Saving best clusters model as {best_composition_path}')
                torch.save(model, best_composition_path)
                # update new best
                best_composition_score = comp_score

            if struct_score > best_structure_score:
                if 'best_structure_path' in vars():
                    os.remove(best_structure_path)
                best_structure_path = 'saved_models/' + 'SyntheticStructures' + '/' + end_name + '_' + str(it) + '_' \
                                     + 'best_structure' + '_comp_' + '{:.5f}'.format(comp_score) + '_struct_' + '{:.5f}'.format(struct_score) + '.pt'
                log.info(f'Saving best ordering model as {best_structure_path}')
                torch.save(model, best_structure_path)
                # update new best
                best_structure_score = struct_score

            # return model to train mode
            model.train()

        if it in args.learning_rates_consecutive:
            log.info(f'Adjusting consecutive learning rate at iteration {it}, from {optimizer.defaults["lr"]} to {args.learning_rates_consecutive[it]} ...')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rates_consecutive[it], weight_decay=args.weight_decay)


if __name__ == '__main__':
    run_id = get_run_id()
    parser = argparse.ArgumentParser(description='Neural Ordered Clusters')
    parser.add_argument('--model', type=str, default='NOC', metavar='S')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Examples per batch, when creating new dataloaders (default: 64)')
    parser.add_argument('--n-examples-per-train-dataloaders', type=int, default=20000,
                        help='Examples per train dataloader (each of different cardinality) (default: 20000)')
    parser.add_argument('--n-examples-per-test-dataloaders', type=int, default=5000,
                        help='Examples per test dataloader (each of different cardinality) (default: 5000)')
    parser.add_argument('--section-padding-int', type=int, default=-999,
                        help='What to pad Y_ci and Y_att with due to varying number of sections in examples. (default: -999)')
    parser.add_argument('--min-cardinality', type=int, default=35,
                        help='Minimum cardinality of input set, aka example length. (default: 35)')
    parser.add_argument('--max-cardinality', type=int, default=50,
                        help='Maximum cardinality of input set, aka example length. (default: 50)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers. (default: 0)')
    parser.add_argument('--iterations', type=int, default=250000, metavar='N',
                        help='number of iterations (examples) to train (default: 250000)')
    parser.add_argument('--learning-rate-initial', type=float, default=1e-4, metavar='N',
                        help='The starting learning rate, which can then be adjusted at specified iterations via separate argument (default: 1e-4)')
    parser.add_argument('--learning-rates-consecutive', type=dict, default={100000: 5e-5, 200000: 1e-5}, metavar='N',
                        help='A dictionary of iterations as keys and consecutive learning rates as values. (default: {100000: 5e-5, 200000: 1e-5})')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='N',
                        help='Weight decay for the optimizer (default: 0.001)')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--run-id', type=int, default=run_id, metavar='N',
                        help='auto-generated run-id')
    parser.add_argument('--logs-path', type=str, default=f'logs/synth_struct_{run_id}', metavar='S',
                        help='path to the log file created during run')
    parser.add_argument('--rulesets-path', type=str, default=f'run_configs/synthetic_rulesets.json', metavar='S',
                        help='path to the rulesets configuration file for the run, which defines structural and compositional rules for the synthetic catalogs.')
    parser.add_argument('--generate-new-data', type=str, default=False, metavar='B',
                        help='whether to generate new synthetic data or not')
    parser.add_argument('--save-every', type=int, default=100,
                        help="Save model every n-th iteration, (default: 100)")
    parser.add_argument('--validate-every', type=int, default=100, metavar='N',
                        help='Validate model every n-th iteration (default: 100)')
    parser.add_argument('--examples-per-metrics', type=int, default=1, metavar='N',
                        help='How many test/validation examples to generate for the metric function (default: 50)')
    parser.add_argument('--inference-n-samples', type=int, default=100, metavar='N',
                        help='How many samples to generate during inference (default: 100)')
    parser.add_argument('--cardinality-conditioned-on-assigned', type=bool, default=True,
                        help='Whether to use the G joint representation of all '
                             'unassigned points when predicting the current cluster cardinality. (default: True)')
    parser.add_argument('--add-set-repr-to-elems', type=bool, default=True, metavar='B')
    parser.add_argument('--ordering-loss-multiplier', type=float, default=15.0, metavar='N',
                        help='A weighing factor by which to multiply the cluster ordering loss. (default: 15.0)')
    parser.add_argument('--cardinality-loss-multiplier', type=float, default=0.1, metavar='N',
                        help='A weighing factor by which to multiply the cluster cardinality loss. (default: 0.1)')
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.rulesets = load_ruleset(args.rulesets_path)

    main(args)
