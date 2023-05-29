import math
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
from matplotlib import rcParams
from matplotlib import gridspec


from model_enoc import NCP_Sampler
from torch import Tensor


def plot_avgs(losses, accs, rot_vars, w, save_name=None):
    up = -1  # 3500

    # cast to numpy if torch
    if type(accs) == Tensor:
        accs = accs.numpy()

    avg_loss = []
    for i in range(w, len(losses)):
        avg_loss.append(np.mean(losses[i - w:i]))

    avg_acc = []
    for i in range(w, len(accs)):
        avg_acc.append(np.mean(accs[i - w:i]))

    avg_var = []
    for i in range(w, len(rot_vars)):
        avg_var.append(np.mean(rot_vars[i - w:i]))

    plt.figure(22, figsize=(13, 10))
    plt.clf()

    plt.subplot(312)
    plt.semilogy(avg_loss[:up])
    plt.ylabel('Mean NLL')
    plt.grid()

    plt.subplot(311)
    plt.plot(avg_acc[:up])
    plt.ylabel('Mean Accuracy')
    plt.grid()

    plt.subplot(313)
    plt.semilogy(avg_var[:up])
    plt.ylabel('Permutation Variance')
    plt.xlabel('Iteration')
    plt.grid()

    if save_name:
        plt.savefig(save_name)


def plot_samples_2d(dpmm, data, cs, N=50, seed=None, save_name=None):
    if seed:
        np.random.seed(seed=seed)

    plt.figure(1, figsize=(30, 5))
    plt.clf()

    fig, ax = plt.subplots(ncols=6, nrows=1, num=1)
    ax = ax.reshape(6)

    # plt.clf()
    N = data.shape[1]
    s = 26  # size for scatter
    fontsize = 15

    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)

    ax[0].scatter(data[0, :, 0], data[0, :, 1], color='gray', s=s)

    K = len(set(cs))

    ax[0].set_title(str(N) + ' Points', fontsize=fontsize)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(2)

    ncp_sampler = NCP_Sampler(dpmm, data)
    S = 100  # number of samples (originally 5000)
    css, probs, _, _ = ncp_sampler.sample(S)

    for i in range(5):
        ax[i + 1].cla()
        cs = css[i, :]

        for j in range(N):
            xs = data[0, j, 0]
            ys = data[0, j, 1]
            ax[i + 1].scatter(xs, ys, color='C' + str(cs[j] + 1), s=s)

        K = len(set(cs))

        ax[i + 1].set_title(str(K) + ' Clusters    Prob: ' + '{0:.2f}'.format(probs[i]), fontsize=fontsize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[i + 1].spines[axis].set_linewidth(0.8)

    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    return K, probs


def show_all(example, target, prediction,
             save_path=None, fixed_scale=True, legend=True,
             title=None, width=12, height=6, limits=[-30, 30], s=15, marker='o', show_origin=True, show_cardinalities=False):
    """
    Take an example, target cluster assignments (in order) and prediction of cluster assignments (also in predicted, relabelled order)
    and show them side by side.
    """

    # need color map / list
    max_vc = 10  # visualizable
    max_pc = 100  # predictable
    cmap = plt.cm.get_cmap('hsv')  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    norm = Normalize(vmin=0, vmax=max_vc)
    cmap_rest = plt.cm.get_cmap('gray')
    norm_rest = Normalize(vmin=0, vmax=max_pc - max_vc)
    colours = [cmap(norm(n)) for n in range(max_vc)]
    colours.extend([cmap_rest(norm_rest(n)) for n in range(max_pc - max_vc)])

    # unpack, turn to list
    x = example[0].tolist()
    target_cs_ordered = target.tolist()
    predicted_cs_ordered = prediction.tolist()

    # this makes the colormap be of the same size as the first 2 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    cax1 = divider1.append_axes('right', size='5%', pad=0.00)
    cax2 = divider2.append_axes('right', size='5%', pad=0.00)
    cax2.axis('off')
    cax1.axis('off')
    cax3 = divider3.append_axes('right', size='3%', pad=0.00)

    # handle size of saved figure
    fig.set_size_inches(width, height)

    if fixed_scale:
        ax1.set_xlim(limits)
        ax1.set_ylim(limits)
        ax2.set_xlim(limits)
        ax2.set_ylim(limits)
        ax3.set_xlim(limits)
        ax3.set_ylim(limits)

    # x and y scales equal
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')

    # titles
    ax1.set_title('Target')
    ax2.set_title('Prediction')
    ax3.set_title('Distance from Origin')
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle('Target, Prediction and Heatmap')

    # control axis tick font sizes
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)

    # if show origin
    if show_origin:
        ax1.plot([0.0], [0.0], marker="x", markersize=3, color="gray")
        ax2.plot([0.0], [0.0], marker="x", markersize=3, color="gray")
        ax3.plot([0.0], [0.0], marker="x", markersize=3, color="gray")

    # plot each axis
    plot_in_axis(x, target_cs_ordered, ax1, colours=colours, s=s, marker=marker, show_cardinalities=show_cardinalities)
    plot_in_axis(x, predicted_cs_ordered, ax2, colours=colours, s=s, marker=marker, show_cardinalities=show_cardinalities)
    im = plot_heatmap_in_axis(x, ax3, s=s)

    # colorbar
    fig.colorbar(im, cax=cax3, orientation='vertical')

    # legends
    if legend:
        ax1.legend(prop={'size': 8})
        ax2.legend(prop={'size': 8})
        # no legend for heatmap
        # ax3.legend(prop={'size': 6})

    # save
    if save_path:
        plt.savefig(save_path, dpi=399)

    # close all figures
    # added to prevent "RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory"
    plt.close('all')


def prepare_figure(example, pred0, pred1, pred2, pred3,
             save_path=None, fixed_scale=True, legend=True,
             title=None, width=12, height=6, limits=[-30, 30], s=15, marker='o', show_origin=True, show_cardinalities=False):
    """
    Special version of show_all() with 4 dividers for final figures.
    """

    # need color map / list
    max_vc = 10  # visualizable
    max_pc = 100  # predictable
    cmap = plt.cm.get_cmap('hsv')  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    norm = Normalize(vmin=0, vmax=max_vc)
    cmap_rest = plt.cm.get_cmap('gray')
    norm_rest = Normalize(vmin=0, vmax=max_pc - max_vc)
    colours = [cmap(norm(n)) for n in range(max_vc)]
    colours.extend([cmap_rest(norm_rest(n)) for n in range(max_pc - max_vc)])

    # unpack, turn to list
    x = example[0].tolist()
    pred0 = pred0.tolist()
    pred1 = pred1.tolist()
    pred2 = pred2.tolist()
    pred3 = pred3.tolist()

    # this makes the colormap be of the same size as the first 2 subplots
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5,
                                                  sharey=False)

    # set the spacing between subplots
    plt.subplots_adjust(wspace=0.05)

    divider0 = make_axes_locatable(ax0)
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    divider4 = make_axes_locatable(ax4)
    cax0 = divider0.append_axes('right', size='0%', pad=0.00)
    cax1 = divider1.append_axes('right', size='0%', pad=0.00)
    cax2 = divider2.append_axes('right', size='0%', pad=0.00)
    cax3 = divider3.append_axes('right', size='0%', pad=0.00)
    cax0.axis('off')
    cax1.axis('off')
    cax2.axis('off')
    cax3.axis('off')
    cax4 = divider4.append_axes('right', size='3%', pad=0.00)

    # handle size of saved figure
    fig.set_size_inches(width, height)

    if fixed_scale:
        ax0.set_xlim(limits)
        ax0.set_ylim(limits)
        ax1.set_xlim(limits)
        ax1.set_ylim(limits)
        ax2.set_xlim(limits)
        ax2.set_ylim(limits)
        ax3.set_xlim(limits)
        ax3.set_ylim(limits)
        ax4.set_xlim(limits)
        ax4.set_ylim(limits)

    # x and y scales equal
    ax0.set_aspect('equal', adjustable='box')
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax4.set_aspect('equal', adjustable='box')

    # titles
    ax0.set_title('Target (supervision)', fontsize='9')
    ax1.set_title('ACP', fontsize='9')
    ax2.set_title('S2S-B', fontsize='9')
    ax3.set_title('NOC (ours)', fontsize='9')
    ax4.set_title('Distance from origin', fontsize='9')

    if title:
        fig.suptitle(title)

    # control axis tick font sizes
    # # ax1.xaxis.set_ticklabels([]) # remove ticks from x scale to save space
    # # ax1.yaxis.set_ticklabels([]) # remove ticks from y scale to save space
    ax0.xaxis.set_tick_params(labelbottom=True)
    ax0.tick_params(axis='y', labelsize=3)

    ax0.tick_params(axis='x', labelsize=3)
    ax1.tick_params(axis='x', labelsize=3)
    ax2.tick_params(axis='x', labelsize=3)
    ax3.tick_params(axis='x', labelsize=3)
    ax4.tick_params(axis='x', labelsize=3)

    ax1.yaxis.set_ticks([])  # remove ticks from y scale to save space
    ax2.yaxis.set_ticks([])  # remove ticks from y scale to save space
    ax3.yaxis.set_ticks([])  # remove ticks from y scale to save space
    ax4.yaxis.set_ticks([])  # remove ticks from y scale to save space

    # ax1.tick_params(axis='y', labelsize=8)
    # ax2.tick_params(axis='y', labelsize=8)
    # ax3.tick_params(axis='y', labelsize=8)
    # ax4.tick_params(axis='y', labelsize=8)

    # if show origin
    if show_origin:
        ax0.plot([0.0], [0.0], marker="x", markersize=3, color="gray")
        ax1.plot([0.0], [0.0], marker="x", markersize=3, color="gray")
        ax2.plot([0.0], [0.0], marker="x", markersize=3, color="gray")
        ax3.plot([0.0], [0.0], marker="x", markersize=3, color="gray")
        ax4.plot([0.0], [0.0], marker="x", markersize=3, color="gray")

    # plot each axis
    plot_in_axis(x, pred0, ax0, colours=colours, s=s, marker=marker, show_cardinalities=show_cardinalities)
    plot_in_axis(x, pred1, ax1, colours=colours, s=s, marker=marker, show_cardinalities=show_cardinalities)
    plot_in_axis(x, pred2, ax2, colours=colours, s=s, marker=marker, show_cardinalities=show_cardinalities)
    plot_in_axis(x, pred3, ax3, colours=colours, s=s, marker=marker, show_cardinalities=show_cardinalities)
    im = plot_heatmap_in_axis(x, ax4, s=s)

    # colorbar
    cbar = fig.colorbar(im, cax=cax4, orientation='vertical')
    cbar.ax.tick_params(labelsize=3)

    # legends
    ax0.legend(prop={'size': 3}) # target always has legend
    if legend:
        ax1.legend(prop={'size': 6})
        ax2.legend(prop={'size': 6})
        ax3.legend(prop={'size': 6})
        # no legend for heatmap
        # ax3.legend(prop={'size': 6})

    # save
    if save_path:
        plt.savefig(save_path, dpi=399)

    # close all figures
    # added to prevent "RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory"
    plt.close('all')


def plot_in_axis(data, cluster_assignments_ordered, an_axis, colours, s=15, marker='o', show_cardinalities=False):
    # zip and sort by cluster index
    ci_and_x = list(zip(cluster_assignments_ordered, data))
    ci_and_x.sort()

    # remove unassigned points (assigned to cluster 999)
    unassigned_ci_and_x = [cx for cx in ci_and_x if cx[0] == 999]
    if len(unassigned_ci_and_x) > 0:
        ci_and_x = ci_and_x[:-len(unassigned_ci_and_x)]

    # split into clusters, in order
    clusters = [[p for ci, p in g] for k, g in groupby(ci_and_x, key=itemgetter(0))]

    # plot every cluster sequentially
    for i, c in enumerate(clusters):
        # pick colours in canonical order
        colour = colours[i]

        # join Xs and Ys (coordinates)
        Xs, Ys = zip(*c)

        # choose proper labels
        if show_cardinalities:
            label = '{} ({})'.format(i + 1, len(Xs))
        else:
            label = str(i+1)

        # plot cluster
        an_axis.scatter(Xs, Ys, color=colour, s=s, marker=marker, edgecolors='gray', linewidth=0.3, label=label)


def plot_heatmap_in_axis(data, an_axis, s=15):
    colourmap = 'rainbow_r'  # change to rainbow to reverse colours

    origin = [0.0, 0.0]

    # obtain distances from origin for all points
    distances = [math.hypot(p[0] - origin[0], p[1] - origin[1]) for p in data]

    # split Xs and Ys
    Xs, Ys = zip(*data)

    # plot cluster
    im = an_axis.scatter(Xs, Ys, c=distances, cmap=colourmap, s=s, vmin=0,
                         vmax=40)  # vmin and vmax make the color scale become fixed, for cross-plot consistency
    # an_axis.colorbar()

    # show all plotted clusters
    # an_axis.colorbar()
    return im


def plot_predicted_synthetic_catalog(catalog_as_indices, img_map, figsizes=(21, 2), save_path=None, title=None):
    """Take a prediction of a catalog as indices, plot it"""

    # get catalog sequence
    seq = catalog_as_indices

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