import argparse
import os
import time

import numpy as np
import torch
from data_gauss2D import get_generator, get_metrics
from model_noc import NOC, get_noc_mog_encoder
from show import plot_avgs, plot_samples_2d, show_all
from utils import relabel, get_parameters, get_logger, get_run_id, count_params


def main(args):

    # logs
    log = get_logger(path=args.logs_path, run_id=args.run_id)
    log.info(f'Starting run {args.run_id}')

    # log args
    log.info(f'Arguments: {args}')

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # unpack params
    batch_size = args.batch_size
    max_it = args.iterations
    params = get_parameters('Gauss2D', vars(args))
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")
    end_name = args.model + '_' + str(args.run_id) + '_mc_' + str(args.max_cardinality)

    # log params
    log.info(f'Params: {params}')
    log.info('Device: {}'.format(str(params['device'])))

    # initialize the data generator
    data_generator = get_generator(params)

    # get encoder
    model = None
    if args.model == 'NOC':
        encoder = get_noc_mog_encoder(params)
        model = NOC(params, encoder).to(params['device'])

    # log params
    n_params_report = count_params(model, return_string=True)
    log.info(n_params_report[1])

    # add directiories for saved models
    if not os.path.isdir(f'saved_models/Gauss2D'):
        os.makedirs(f'saved_models/Gauss2D')
    if not os.path.isdir(f'figures/Gauss2D'):
        os.makedirs(f'figures/Gauss2D')

    # define containers to collect statistics
    ordering_criterion = torch.nn.CrossEntropyLoss()
    losses_clustering = []  # NLLs
    losses_ordering = []
    losses_cardinality = []
    losses_final = []
    accs_or_elbos = []  # Accuracy or elbo of the classification prediction

    it = 0  # iteration counter

    # optimizer uses learning rates and weight decay from parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate_initial, weight_decay=args.weight_decay)

    # set model to train, clear grads
    model.train()
    optimizer.zero_grad()

    # track best model scores and save those models
    best_cs_score = 0
    best_co_score = 0

    # training loop
    while True:
        t_start = time.time()
        it += 1

        if it == max_it:
            break

        # generate a batch of data
        data, cs, clusters, K, cs_ordered, _, Y_ci = data_generator.generate(None, batch_size)
        N = data.shape[1]

        # # Debug test plot
        # for i in range(20):
        #     batch_size = 1
        #     data, cs, clusters, num_clusters, cs_ordered = data_generator.generate(None, batch_size=batch_size)
        #     for b in range(batch_size):
        #         show_prediction(np.expand_dims(data[b], axis=0), cs_ordered)
        #         show_x_via_colormap(data[b])
        #         plot_target(np.expand_dims(data[b], axis=0), cs_ordered, clusters, num_clusters)

        try:

            # get clustering loss
            labels = np.tile(cs, batch_size).reshape(batch_size, cs.size)
            loss_training, acc_or_elbo = model.train_clustering_loss_masked(data, labels) # loss is the difference between logprob assigned to true cluster index

            # unpack FNOC losses
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

            # backward step
            loss_final.backward()

            losses_clustering.append(loss_clustering.item())
            losses_ordering.append(loss_ordering.item())
            losses_final.append(loss_final.item())
            accs_or_elbos.append(acc_or_elbo.mean())

            # FNOC loss tracking
            if args.model == 'NOC':
                losses_cardinality.append(loss_cardinality.item())

            optimizer.step()  # the gradients used in this step are the sum of the gradients for each permutation
            optimizer.zero_grad()

            # report per example
            if args.model == 'NOC':
                log.info('{0:4d}  N:{1:2d}  K:{2}  Clustering Loss:{3:.3f}  Ordering Loss:{4:.3f}  Cardinality Loss: {5:.3f} Total Loss:{6:.3f}  Mean Time/Iteration: {7:.1f}' \
                         .format(it, N, K, np.mean(losses_clustering[-50:]), np.mean(losses_ordering[-50:]), np.mean(losses_cardinality[-50:]), np.mean(losses_final[-50:]), (time.time()-t_start)))

        except RuntimeError:
            bsize = int(.75 * data.shape[0])
            if bsize > 2:
                log.info('RuntimeError handled  ', 'N:', N, ' K:', K, 'Trying batch size:', bsize)
                data = data[:bsize, :, :]
            else:
                break

        if it % args.validate_every == 0 and it > 0:

            torch.cuda.empty_cache()
            # plot_avgs(losses_final, accs_or_elbos, empty_placeholder, 50, save_name='./figures/train_avgs_' + end_name + '.pdf')

            if params['generative_model'] == 'Gauss2D':

                # 0. [optional] Manually choose clusters
                # manual_clusters = [24, 5, 17, 9, 1, 41, 53]
                # manual_crp = data_generator.get_crp_from_chosen_clusters(manual_clusters)
                model.eval()

                # 1. Generate data
                x, cs, clusters, num_clusters, cs_ordered, Y_att, Y_ci = data_generator.generate(None, batch_size=1, crp=None) # replace None with manual_clusters

                # 3. Infer on the example
                y_hat_cs, y_hat_co = model.infer(x, n_samples=args.inference_n_samples)

                # 4. Custom visualizations
                log.info('Creating ordering plot...')
                show_all(x, cs_ordered, y_hat_cs, title=f'Single inference at iter {it}', save_path=f'figures/Gauss2D/{args.model}_{args.run_id}_samples_2D_{it}_example',
                         s=5, marker='o', show_cardinalities=True,
                         fixed_scale=True)

                # 5. Metrics (main)
                # report the clustering and clustering score always, and max-cardinality adherence score if max cardinality specified.
                m = get_metrics(model, data_generator, log, m_examples=args.examples_per_metrics, max_cardinality=args.max_cardinality)
                cs_score = m['cs_score']
                co_score = m['co_score']

                # report accordingly
                if args.max_cardinality:
                    mc_score = m['mc_score']
                    log.info(f'Metrics: Clustering Score: {cs_score:.5f}  |  Ordering Score: {co_score:.5f} | Max-Cardinality Adherence ({args.max_cardinality}): {mc_score:.5f}')
                else:
                    log.info('Metrics: Clustering Score: {:.5f}  |  Ordering Score: {:.5f}'.format(cs_score, co_score))

                # 6. Save the models if their performance is better than previous ones
                if (co_score + cs_score) >= (best_co_score + best_cs_score):
                    # have to guard against first time the metrics are obtained and there's no previous model to remove
                    if 'best_over_all_path' in vars():
                        os.remove(best_over_all_path)
                    best_over_all_path = 'saved_models/' + 'Gauss2D' + '/' + end_name + '_' + str(it) + '_' \
                                         + 'best_over_all' + '_cl_' + '{:.5f}'.format(cs_score) + '_or_' + '{:.5f}'.format(co_score) + '.pt'
                    log.info(f'Saving best overall model as {best_over_all_path}')
                    torch.save(model, best_over_all_path)

                if cs_score > best_cs_score:
                    if 'best_clusters_path' in vars():
                        os.remove(best_clusters_path)
                    best_clusters_path = 'saved_models/' + 'Gauss2D' + '/' + end_name + '_' + str(it) + '_'\
                                         + 'best_clusters' + '_cl_' + '{:.5f}'.format(cs_score) + '_or_' + '{:.5f}'.format(co_score) + '.pt'
                    log.info(f'Saving best clusters model as {best_clusters_path}')
                    torch.save(model, best_clusters_path)
                    # update new best
                    best_cs_score = cs_score

                if co_score > best_co_score:
                    if 'best_ordering_path' in vars():
                        os.remove(best_ordering_path)
                    best_ordering_path = 'saved_models/' + 'Gauss2D' + '/' + end_name + '_' + str(it) + '_' \
                                         + 'best_ordering' + '_cl_' + '{:.5f}'.format(cs_score) + '_or_' + '{:.5f}'.format(co_score) + '.pt'
                    log.info(f'Saving best ordering model as {best_ordering_path}')
                    torch.save(model, best_ordering_path)
                    # update new best
                    best_co_score = co_score

                # 7. put model back into training mode
                model.train()

        if it % args.save_every == 0 and it > 0:
            if 'fname' in vars():
                os.remove(fname)
            model.params['it'] = it
            fname = 'saved_models/' + 'Gauss2D' + '/' + end_name + '_' + str(it) + '.pt'
            log.info(f'Saving model at args.save_every, {fname}')
            torch.save(model, fname)

        if it in args.learning_rates_consecutive:
            log.info(f'Adjusting consecutive learning rate at iteration {it}, from {optimizer.defaults["lr"]} to {args.learning_rates_consecutive[it]} ...')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rates_consecutive[it], weight_decay=args.weight_decay)


if __name__ == '__main__':
    # run id
    run_id = get_run_id()
    parser = argparse.ArgumentParser(description='Neural Ordered Clusters')
    parser.add_argument('--model', type=str, default='NOC', metavar='S')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--iterations', type=int, default=30000, metavar='N',
                        help='number of iterations to train (default: 30000)')
    parser.add_argument('--learning-rate-initial', type=float, default=1e-4, metavar='N',
                        help='The starting learning rate, which can then be adjusted at specified iterations via separate argument (default: 1e-4)')
    parser.add_argument('--learning-rates-consecutive', type=dict, default={15000: 5e-5, 20000: 1e-5}, metavar='N',
                        help='A dictionary of iterations as keys and consecutive learning rates as values. (default: {1200: 5e-5, 2200: 1e-5})')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='N',
                        help='Weight decay for the optimizer (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 5)')
    parser.add_argument('--run-id', type=int, default=run_id, metavar='N',
                        help='auto-generated run-id')
    parser.add_argument('--logs-path', type=str, default=f'logs/{run_id}', metavar='S',
                        help='path to the log file created during run')
    parser.add_argument('--max-cardinality', type=int, default=30, metavar='N',
                        help='Whether to apply a max cardinality threshold to CRP clusters, and what threshold it should be. (default: 30)')
    parser.add_argument('--save-every', type=int, default=100,
                        help="Save model every n-th iteration, (default: 100)")
    parser.add_argument('--validate-every', type=int, default=100, metavar='N',
                        help='Validate model every n-th iteration (default: 100)')
    parser.add_argument('--examples-per-metrics', type=int, default=50, metavar='N',
                        help='How many test/validation examples to generate for the metric function (default: 50)')
    parser.add_argument('--inference-n-samples', type=int, default=100, metavar='N',
                        help='How many samples to generate during inference (default: 100)')
    parser.add_argument('--ordering-loss-multiplier', type=float, default=4.0, metavar='N',
                        help='A weighing factor by which to multiply the cluster ordering loss. (default: 1 for ENOC, 4 for CNOC)')
    parser.add_argument('--cardinality-loss-multiplier', type=float, default=0.003, metavar='N',
                        help='A weighing factor by which to multiply the cluster cardinality loss. (default: 0.01, only relevant for FNOC)')
    parser.add_argument('--cardinality-conditioned-on-assigned', type=bool, default=True,
                        help='Whether to use the G joint representation of all '
                             'unassigned points when predicting the current cluster cardinality. (default: True, only relevant for FNOC')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
