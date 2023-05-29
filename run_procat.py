import argparse
import torch
from data_procat import get_procat_tokenizer, get_procat_dataloader, get_parameters_procat, get_batch, \
    decode_procat_batch, get_model_basic_procat_metrics
from model_noc import NOC, get_NOC_procat_encoder
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

    # data paths
    train_dloader_path = os.path.join(
        os.getcwd(), 'data', 'PROCAT', '{}_setlen_{}_batch_{}_train.pb'.format(args.language_model, args.max_sentence_length, args.batch_size))
    cv_dloader_path = os.path.join(
        os.getcwd(), 'data', 'PROCAT', '{}_setlen_{}_batch_{}_cv.pb'.format(args.language_model, args.max_sentence_length, args.batch_size))
    test_dloader_path = os.path.join(
        os.getcwd(), 'data', 'PROCAT', '{}_setlen_{}_batch_{}_test.pb'.format(args.language_model, args.max_sentence_length, args.batch_size))

    # prepare tokenizer
    log.info('Loading the proper tokenizer for language model: {}...'.
             format(args.language_model))
    tokenizer = get_procat_tokenizer(args.language_model)

    # get data
    if args.generate_new_data:
        # new dataloaders
        log.info('Obtaining test data ...')
        test_dataloader = get_procat_dataloader('test', log, args, tokenizer)

        log.info('Obtaining cv data ...')
        cv_dataloader = get_procat_dataloader('validation', log, args, tokenizer)

        log.info('Obtaining train data ...')
        train_dataloader = get_procat_dataloader('train', log, args, tokenizer)

        # save dataloaders
        log.info('Data persistance (saving dataloaders)')
        torch.save(train_dataloader, train_dloader_path)
        torch.save(test_dataloader, cv_dloader_path)
        torch.save(cv_dataloader, test_dloader_path)

        # reload dataloaders
        log.info('Data (re-)loading')
        train_dataloader = torch.load(train_dloader_path)
        test_dataloader = torch.load(cv_dloader_path)
        cv_dataloader = torch.load(test_dloader_path)

    # reload
    log.info('Reloading dataloaders...')
    train_dataloader = torch.load(train_dloader_path)
    test_dataloader = torch.load(cv_dloader_path)
    cv_dataloader = torch.load(test_dloader_path)

    # check batch
    log.info('Checking test dataloader ...')
    a_batch = next(iter(test_dataloader))
    decode_procat_batch(a_batch, tokenizer, log)

    # unpack params
    max_it = args.iterations
    params = get_parameters_procat('PROCAT', vars(args))
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")
    log.info('Device: {}'.format(str(params['device'])))
    end_name = args.model + '_' + str(args.run_id)

    model = None
    if args.model == 'NOC':
        encoder = get_NOC_procat_encoder(params)
        model = NOC(params, encoder).to(params['device'])

    # log params
    n_params_report = count_params(model, return_string=True)
    log.info(n_params_report[1])

    # add directiories for saved models ...
    if not os.path.isdir(f'saved_models/PROCAT'):
        os.makedirs(f'saved_models/PROCAT')
    # ... and figures
    if not os.path.isdir(f'figures/PROCAT'):
        os.makedirs(f'figures/PROCAT')

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
    best_co_score = 0
    best_cs_score = 0

    # main loop
    while True:
        t_start = time.time()
        it += 1

        if it == max_it:
            break

        data, labels, cardinalities, K, labels_ordered, _, Y_ci = get_batch(train_dataloader, single_example_batch=False)
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
            fname = 'saved_models/' + 'PROCAT' + '/' + end_name + '_' + str(it) + '.pt'
            log.info(f'Saving model at args.save_every, {fname}')
            torch.save(model, fname)

        if it % args.validate_every == 0 and it > 0:

            torch.cuda.empty_cache()
            model.eval()

            # metrics
            log.info(f'Obtaining basic metrics for {args.examples_per_metrics} examples,'
                     f' with {args.inference_n_samples} samples each...')
            metrics_basic = get_model_basic_procat_metrics(a_model=model,
                                                           dataloaders=cv_dataloader,
                                                           n_examples=args.examples_per_metrics,
                                                           n_samples=args.inference_n_samples,
                                                           logger=log)
            cs_score = metrics_basic['CLUSTERING_SCORE_AVG']
            co_score = metrics_basic['ORDERING_SCORE_AVG']

            log.info('Metrics: Clustering Score: {:.5f}  |  Ordering Score: {:.5f}'.format(cs_score, co_score))

            # save based on scores
            if (co_score + cs_score) >= (best_co_score + best_cs_score):
                # have to guard against first time the metrics are obtained and there's no previous model to remove
                if 'best_over_all_path' in vars():
                    os.remove(best_over_all_path)
                best_over_all_path = 'saved_models/' + 'PROCAT' + '/' + end_name + '_' + str(it) + '_' \
                                     + 'best_over_all' + '_cl_' + '{:.5f}'.format(cs_score) + '_or_' + '{:.5f}'.format(co_score) + '.pt'
                log.info(f'Saving best overall model as {best_over_all_path}')
                torch.save(model, best_over_all_path)

            if cs_score > best_cs_score:
                if 'best_clusters_path' in vars():
                    os.remove(best_clusters_path)
                best_clusters_path = 'saved_models/' + 'PROCAT' + '/' + end_name + '_' + str(it) + '_' \
                                     + 'best_clusters' + '_cl_' + '{:.5f}'.format(cs_score) + '_or_' + '{:.5f}'.format(co_score) + '.pt'
                log.info(f'Saving best clusters model as {best_clusters_path}')
                torch.save(model, best_clusters_path)
                # update new best
                best_cs_score = cs_score

            if co_score > best_co_score:
                if 'best_ordering_path' in vars():
                    os.remove(best_ordering_path)
                best_ordering_path = 'saved_models/' + 'PROCAT' + '/' + end_name + '_' + str(it) + '_' \
                                     + 'best_ordering' + '_cl_' + '{:.5f}'.format(cs_score) + '_or_' + '{:.5f}'.format(co_score) + '.pt'
                log.info(f'Saving best ordering model as {best_ordering_path}')
                torch.save(model, best_ordering_path)
                # update new best
                best_co_score = co_score

            # return model to train mode
            model.train()

        if it in args.learning_rates_consecutive:
            log.info(f'Adjusting consecutive learning rate at iteration {it}, from {optimizer.defaults["lr"]} to {args.learning_rates_consecutive[it]} ...')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rates_consecutive[it], weight_decay=args.weight_decay)


if __name__ == '__main__':
    run_id = get_run_id()
    parser = argparse.ArgumentParser(description='Neural Ordered Clusters')
    parser.add_argument('--model', type=str, default='NOC', metavar='S')
    parser.add_argument('--train_set', type=str, default='PROCAT', choices=['PROCAT', 'PROCAT_mini'])
    parser.add_argument('--test_set', type=str, default='PROCAT', choices=['PROCAT', 'PROCAT_mini'])
    parser.add_argument('--language_model', type=str, default='danish-bert-botxo')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Examples per batch, when creating new dataloaders (default: 64)')
    parser.add_argument('--n-examples-per-train-dataloaders', type=int, default=8000,
                        help='Examples per train dataloader (each of different cardinality) (default: 8000)')
    parser.add_argument('--n-examples-per-test-dataloaders', type=int, default=2000,
                        help='Examples per test dataloader (each of different cardinality) (default: 2000)')
    parser.add_argument('--section-padding-int', type=int, default=-999,
                        help='What to pad Y_ci and Y_att with due to varying number of sections in examples. (default: -999)')
    parser.add_argument('--max-sentence-length', type=int, default=512)
    parser.add_argument('--num-sentences', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers. (default: 0)')
    parser.add_argument('--iterations', type=int, default=12500, metavar='N',
                        help='number of iterations (examples) to train (default: 12500)')
    parser.add_argument('--learning-rate-initial', type=float, default=1e-4, metavar='N',
                        help='The starting learning rate, which can then be adjusted at specified iterations via separate argument (default: 1e-4)')
    parser.add_argument('--learning-rates-consecutive', type=dict, default={5000: 5e-5, 10000: 1e-5}, metavar='N',
                        help='A dictionary of iterations as keys and consecutive learning rates as values. (default: {5000: 5e-5, 10000: 1e-5})')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='N',
                        help='Weight decay for the optimizer (default: 0.001)')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--run-id', type=int, default=run_id, metavar='N',
                        help='auto-generated run-id')
    parser.add_argument('--logs-path', type=str, default=f'logs/procat_{run_id}', metavar='S',
                        help='path to the log file created during run')
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
                        help='A weighing factor by which to multiply the cluster ordering loss. (default: 10.0)')
    parser.add_argument('--cardinality-loss-multiplier', type=float, default=0.1, metavar='N',
                        help='A weighing factor by which to multiply the cluster cardinality loss. (default: 0.5)')
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    main(args)
