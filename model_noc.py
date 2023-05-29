"""
All credit for implementations of original neural clustering methods  goes to Ari Pakman et al. and Juho Lee et al., as per their cited papers:

From juho-lee:

Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, ICML 2019
Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh
https://arxiv.org/abs/1810.00825

and

Deep Amortized Clustering
Juho Lee, Yoonho Lee, Yee Whye Teh
https://arxiv.org/abs/1909.13433
"""

import math

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from utils import relabel, get_ordered_cluster_assignments


class CCC_Sampler():
    """
    Clusterwise Constrained Cardinality Sampler for NOC
    """

    def __init__(self, model, data, device=None):

        if not torch.cuda.is_available():
            print('Warning: CUDA is not available')

        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.device = device

        with torch.no_grad():
            data = torch.from_numpy(data).float().to(self.device)
            self.enc_data = model.encoder(data).squeeze(0)  # [N,e_dim]
            # print(self.enc_data.shape)
            self.hs = self.model.h(self.enc_data)  # [N,h_dim]

            if not self.model.use_attn:
                self.us = self.model.u(self.enc_data)  # [N,u_dim]

    def _sample_Z(self, A, U, G, nZ):

        mu, log_sigma = self.model.get_pz(A, U, G)  # [t,z_dim]
        std = log_sigma.exp().unsqueeze(0)
        eps = torch.randn([nZ, *mu.shape]).to(self.device)
        return mu.unsqueeze(0) + eps * std

    def sample(self, S, sample_Z=False, sample_B=False, prob_nZ=10, prob_nA=None):
        """
        S: number of parallel sampling threads
        nZ = number of samples of the latent variable z when estimating the probability of each sample
        Output:
            cs: tensor of samples, shape = [Ss,N], where Ss <= S, because duplicates are eliminated
            probs: estimated probabilities of the Ss samples
        """
        ccs = self._sample(S=S, sample_Z=sample_Z, sample_B=sample_B)
        cs, probs = self._estimate_probs(ccs, nZ=prob_nZ, nA=prob_nA)

        return cs, probs

    def _sample(self, S, sample_Z=False, sample_B=False):

        device = self.device
        N = self.enc_data.shape[0]

        G = torch.zeros([S, self.model.g_dim]).to(device)
        cs = -torch.ones([S, N]).long().to(device)

        if self.model.use_attn:
            big_enc_data = self.enc_data.view(
                [1, N, self.model.e_dim]).expand(S, N, self.model.e_dim)

        big_hs = self.hs.view([1, N, self.model.h_dim]
                              ).expand(S, N, self.model.h_dim)

        with torch.no_grad():

            # this matrix keeps track of available unassigned indices in each thread
            mask = torch.ones([S, N]).to(device)

            k = -1
            t = S  # t counts how many threads have not completed their sampling
            while t > 0:

                k += 1

                # sample the anchor element in a new cluster for each thread
                anchs = torch.multinomial(mask[:t, :], 1)  # [t,1]
                # assign label k to anchor elements
                cs[:t, :].scatter_(1, anchs, k)

                if self.model.use_attn:

                    HX = self.model.ISAB1(
                        big_enc_data[:t, :, :], mask[:t, :])  # [t,N,e_dim]
                    # [t,e_dim]
                    A = HX[torch.arange(t), anchs[torch.arange(t), 0], :]
                    us_pma_input = self.model.MAB(
                        HX, A.unsqueeze(1))  # [t,N,e_dim]
                    # mask[:t, :].scatter_(1, anchs, 0)

                    U = self.model.pma_u(us_pma_input, mask[:t, :]).squeeze(1)

                    # eliminate selected anchors from the mask
                    mask[:t, :].scatter_(1, anchs, 0)
                    Dr = HX

                else:
                    mask[:t, :].scatter_(1, anchs, 0)
                    # this is used when U is agregated from us with the mean
                    normalized_mask = mask[:t, :] / \
                                      mask[:t, :].sum(1, keepdims=True)
                    U = torch.mm(normalized_mask, self.us)  # [t, u_dim]
                    A = self.enc_data[anchs[:, 0], :]  # [t, u_dim]
                    Dr = self.enc_data.view([1, N, self.model.e_dim]).expand(
                        t, N, self.model.e_dim)

                if sample_Z:
                    Z = self._sample_Z(A, U, G[:t, :], 1)  # [1,t,z_dim]
                else:
                    # or use the mu without sampling
                    Z, _ = self.model.get_pz(A, U, G[:t, :])
                    Z = Z.unsqueeze(0)

                Ur = U.view([t, 1, self.model.u_dim]).expand(
                    t, N, self.model.u_dim)
                Ar = A.view([t, 1, self.model.e_dim]).expand(
                    t, N, self.model.e_dim)
                Zr = Z.view([t, 1, self.model.z_dim]).expand(
                    t, N, self.model.z_dim)
                Gr = G[:t, :].view([t, 1, self.model.g_dim]).expand(
                    t, N, self.model.g_dim)

                phi_arg = torch.cat([Dr, Zr, Ar, Ur, Gr], dim=2).view(
                    [t * N, self.model.phi_input_dim])

                logits = self.model.phi(phi_arg).view([t, N])
                prob_one = 1 / (1 + torch.exp(-logits))

                if sample_B:
                    inds = torch.rand([t, N]).to(device) < prob_one[:, :]
                else:

                    # predict current cluster cardinality for all samples, preventing jagged arrays,
                    predicted_cluster_cardinalities = self.model.predict_cardinality_batched(big_enc_data, anchs, mask, G, is_inference=True)

                    # model might technically predict cardinalities greater than the available number of points, so we clamp
                    predicted_cluster_cardinalities = torch.clamp(predicted_cluster_cardinalities, 0, N)

                    # select top PCC (predicted k-th cluster cardinality) probabilities, have only those be 1s in the inds tensor
                    # inds can technically have 1s in places of already used elements, but this gets cleared when sampled_new is created
                    # and mask_update doesn't care, cause these will get caught by the mask
                    inds = self.get_pcc_inds(mask, prob_one, predicted_cluster_cardinalities).to(self.device)

                sampled = inds.long()

                # these are the points that were available (1 in mask) AND sampled (1 in sampled)
                sampled_new = mask[:t, :].long() * sampled

                # find the flattened indices of new points for cluster k
                new_points = torch.nonzero(
                    sampled_new.view(t * N), as_tuple=True)
                # assign points to cluster k
                cs[:t, :].view(t * N)[new_points] = k

                mask_update = 1 - sampled  # this just flips 1s to 0s and vice versa
                # the matrix mask_update has a 1 on those points that survived the last sampling
                # so if a point was available before sampling and survived, it should be available in mask
                mask[:t, :] = mask[:t, :] * mask_update

                new_cluster = (cs[:t, :] == k).float()  # new cluster is just 1s and 0s over N, where 1 is at indices of points in the current cluster
                if self.model.use_attn:
                    new_Hs = self.model.pma_h(
                        big_hs[:t], new_cluster).squeeze(1)
                else:
                    # this is used when H is agregated from hs with the mean
                    new_cluster = new_cluster / new_cluster.sum(1, keepdims=True)
                    new_Hs = torch.mm(new_cluster, self.hs)

                G[:t] = G[:t] + self.model.g(new_Hs)

                msum = mask[:t, :].sum(dim=1)  # msum is the number of still available points (ones that have a 1 at their index in the mask)
                if (msum == 0).any():  # if any thread was fully sampled
                    msumfull = mask.sum(dim=1)

                    # reorder the threads so that those already completed are at the end
                    mm = torch.argsort(msumfull, descending=True)
                    mask = mask[mm, :]
                    cs = cs[mm, :]
                    G = G[mm, :]
                    # recompute the number of threads where there are still points to assign
                    t = (mask.sum(dim=1) > 0).sum()

        cs = cs.cpu().numpy()
        for i in range(S):
            cs[i, :] = relabel(cs[i, :])  # relabelling here? Doesn't this mess with later cluster order prediction?

        # eliminate duplicates
        lcs = list(set([tuple(cs[i, :]) for i in range(S)]))
        Ss = len(lcs)
        ccs = np.zeros([Ss, N], dtype=np.int32)
        for s in range(Ss):
            ccs[s, :] = lcs[s]

        return ccs

    def _estimate_probs(self, cs, nZ, nA=None):

        with torch.no_grad():

            S = cs.shape[0]
            N = cs.shape[1]
            probs = np.ones(S)

            for s in range(S):

                K = cs[s, :].max() + 1
                G = torch.zeros(self.model.g_dim).to(self.device)

                # array of available indices before sampling cluster k
                Ik = np.arange(N)

                for k in range(K):

                    # all these points are possible anchors
                    Sk = cs[s, :] == k

                    nk = len(Ik)

                    ind_in = np.where(cs[s, Ik] == k)[0]
                    ind_out = np.where(cs[s, Ik] != k)[0]

                    if nA is None or Sk.sum() < nA:
                        sk = Sk.sum()
                        anchors = ind_in
                    else:
                        sk = nA
                        anchors = np.random.choice(ind_in, sk, replace=False)

                    d1 = list(range(sk))

                    if self.model.use_attn:

                        HX = self.model.ISAB1(
                            self.enc_data[Ik, :])  # [nk,e_dim]
                        bigHx = HX.view([1, nk, self.model.e_dim]).expand(
                            [sk, nk, self.model.e_dim])
                        A = HX[anchors, :]  # [sk,e_dim]
                        us_pma_input = self.model.MAB(
                            bigHx, A.unsqueeze(1))  # [sk,nk,e_dim]
                        U = self.model.pma_u(
                            us_pma_input).squeeze(1)  # [sk,u_dim]
                        Dr = HX

                    else:

                        A = self.enc_data[Ik, :][anchors, :]
                        U = self.us[Ik, :].sum(0)
                        U = U.view([1, self.model.u_dim]).expand(
                            sk, self.model.u_dim)
                        U = U - self.us[Ik, :][anchors, :]
                        U /= nk - 1
                        Dr = self.enc_data[Ik, :]

                    Ge = G.view([1, self.model.g_dim]).expand(
                        sk, self.model.g_dim)
                    Z = self._sample_Z(A, U, Ge, nZ)  # [nZ,sk,z_dim]
                    Ar = A.view([1, 1, sk, self.model.e_dim]).expand(
                        [nZ, nk, sk, self.model.e_dim])
                    Dr = Dr.view([1, nk, 1, self.model.e_dim]).expand(
                        [nZ, nk, sk, self.model.e_dim])
                    Ur = U.view([1, 1, sk, self.model.u_dim]).expand(
                        [nZ, nk, sk, self.model.u_dim])
                    Gr = G.view([1, 1, 1, self.model.g_dim]).expand(
                        [nZ, nk, sk, self.model.g_dim])
                    Zr = Z.view([nZ, 1, sk, self.model.z_dim]).expand(
                        [nZ, nk, sk, self.model.z_dim])

                    phi_arg = torch.cat([Dr, Zr, Ar, Ur, Gr], dim=3).view(
                        [nZ * nk * sk, self.model.phi_input_dim])

                    logits = self.model.phi(phi_arg).view([nZ, nk, sk])
                    prob_one = 1 / (1 + torch.exp(-logits))
                    prob_one = prob_one.cpu().detach().numpy()

                    prob_one[:, anchors, d1] = 1

                    pp = prob_one[:, ind_in, :].prod(1)
                    if len(ind_out) > 0:
                        pp *= (1 - prob_one)[:, ind_out, :].prod(1)

                    pp = pp.mean(0).sum() * (Sk.sum() / sk) / nk

                    probs[s] *= pp

                    # prepare for next iteration
                    Hs = self.hs[Sk, :]
                    if self.model.use_attn:
                        Hs = Hs.unsqueeze(0)
                        Hs = self.model.pma_h(Hs).view(self.model.h_dim)
                    else:
                        Hs = Hs.mean(dim=0)
                    G += self.model.g(Hs)

                    # update the set of available indices
                    Ik = np.setdiff1d(Ik, np.where(Sk)[0], assume_unique=True)

        # sort in decreasing order of probability
        inds = np.argsort(-probs)
        probs = probs[inds]
        cs = cs[inds, :]

        return cs, probs

    @staticmethod
    def get_pcc_inds(mask, prob_one, pccs):
        """Take the current mask, predicted probabilities for each element to belong in current k-th clusters (prob_one),
        and the predicted cluster cardinalities (pccs) for each of the current t samples, return a tensor called inds,
        which is of size (t, N), where only the top pcc indices are set to 1 and the rest are 0s.
        The anchor is removed (marked as 0 in the mask tensor) beforehand, otherwise it would have to be removed here too.
        """
        # get current number of open samples (t) and the cardinality of the input set (N)
        t, N = prob_one.size()

        # apply mask (mask includes anchor already) to only have predicted probabilities for available elements
        prob_one = prob_one * mask[:t, :].long()

        # placeholder
        inds = torch.zeros(t, N)

        # fill inds with ones in proper places dictated by pccs
        for i, prob_one_row in enumerate(prob_one):
            # find predicted cluster cardinality for this sample
            pcc = pccs[i].item()

            # find topk elems and indices
            _, top_pcc_inds = torch.topk(prob_one_row, pcc)

            # update this row of inds (1s at indices of top_pcc_inds)
            inds[i, top_pcc_inds] = 1

        return inds


class NOC(nn.Module):
    """
    Flexible-cardinality clusterwise Neural Ordered Clusters (with attention, based on ACP)
    """

    def __init__(self, params, encoder):

        super().__init__()

        self.params = params
        self.encoder = encoder
        self.previous_k = -1
        self.device = params['device']

        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']
        self.u_dim = params['h_dim']
        self.z_dim = params['z_dim']
        self.e_dim = params['e_dim']
        self.s2s_c_dim = params['s2s_c_dim']

        H = params['H_dim']

        self.use_attn = self.params['use_attn']
        self.condition_cardinality_on_assigned = self.params['condition_cardinality_on_assigned']

        if self.use_attn:
            self.n_heads = params['n_heads']
            self.n_inds = params['n_inds']

            self.pma_h = PMA(dim_X=self.h_dim, dim=self.h_dim,
                             num_inds=1, num_heads=params['n_heads'])
            self.pma_u = PMA(dim_X=self.u_dim, dim=self.u_dim,
                             num_inds=1, num_heads=params['n_heads'])
            self.pma_u_in = PMA(dim_X=self.u_dim, dim=self.u_dim,
                                num_inds=1, num_heads=params['n_heads'])
            self.pma_u_out = PMA(
                dim_X=self.u_dim, dim=self.u_dim, num_inds=1, num_heads=params['n_heads'])

            self.ISAB1 = ISAB(dim_X=self.e_dim,
                              dim=self.e_dim, num_inds=self.n_inds)
            self.MAB = MAB(dim_X=self.e_dim, dim_Y=self.e_dim,
                           dim=self.e_dim, num_heads=params['n_heads'])
            # cardinality prediction
            self.ISAB_c = ISAB(dim_X=self.e_dim,
                               dim=self.e_dim, num_inds=self.n_inds)
            self.MAB_c = MAB(dim_X=self.e_dim, dim_Y=self.e_dim,
                             dim=self.e_dim, num_heads=params['n_heads'])
            self.pma_c = PMA(dim_X=self.u_dim, dim=self.u_dim,
                             num_inds=1, num_heads=params['n_heads'])
            if self.condition_cardinality_on_assigned:
                self.c = torch.nn.Linear(self.u_dim + self.g_dim, 1)
            else:
                self.c = torch.nn.Linear(self.u_dim, 1)

            # since with varying number of clusters we're not guaranteed batches of exual size at each k, averaging might not be best
            self.c_loss = torch.nn.MSELoss(reduction='mean')

        else:
            self.u = torch.nn.Sequential(
                torch.nn.Linear(self.e_dim, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.u_dim),
            )

        self.h = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.h_dim),
        )

        self.g = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.g_dim),
        )

        self.phi_input_dim = self.e_dim + self.z_dim + \
                             self.e_dim + self.u_dim + self.g_dim

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.phi_input_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 1, bias=False),
        )

        self.pz_mu_log_sigma = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim + self.u_dim + self.g_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 2 * self.z_dim),
        )

        self.qz_mu_log_sigma = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim + 2 * self.u_dim + self.g_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 2 * self.z_dim),
        )

        # set2seq params
        self.elem_embedder = nn.Linear(self.h_dim * 2, self.s2s_c_dim)  # doubling for CNOC

        self.set_encoder_dim = self.s2s_c_dim * 2
        self.set_embedding = SetEncoderSetTransformer(
            elem_embed_dim=self.s2s_c_dim,  # doubling for CNOC
            elem_embed_n_layers=1,
            set_embed_num_heads=4,
            set_embed_num_seeds=1,
            set_embed_dim=self.set_encoder_dim,
            set_embed_n_layers=1)

        self.context_rnn_used = True
        self.permute_module_hidden_dim = self.s2s_c_dim * 2

        # Initialize decoder_input0
        self.decoder_input0 = Parameter(torch.FloatTensor(
            self.permute_module_hidden_dim), requires_grad=False)
        nn.init.uniform_(self.decoder_input0, -1, 1)

        self.context_rnn = SetEncoderRNN(
            elem_embed_dims=self.s2s_c_dim,  # doubling for CNOC
            set_embed_dim=self.permute_module_hidden_dim,
            set_embed_rnn_layers=2,
            set_embed_rnn_bidir=False,
            set_embed_dropout=0.01)

        self.decoder = PointerDecoder(
            elem_embed_dim=self.s2s_c_dim,  # doubling for NOC
            set_embed_dim=self.set_encoder_dim,
            hidden_dim=self.s2s_c_dim * 2,  # quadrupling for NOC
            masking=True)

    def train_clustering_loss(self, batch_x, batch_cluster_labels, vae_samples=40):
        """Train on a batch, return loss and additional metrics if need be."""
        batch_x = torch.from_numpy(batch_x).float().to(self.device)
        batch_cluster_labels = torch.from_numpy(batch_cluster_labels).to(self.device)
        losses, elbo = self.forward_train(batch_x, batch_cluster_labels, w=vae_samples)

        return losses, elbo

    def train_clustering_loss_masked(self, batch_x, batch_cluster_labels, vae_samples=40):
        """
        Masked version can handle targets where each example shouldn't have the same exact
        target labels. Batches still need to be square (not jagged).
        """
        batch_x = torch.from_numpy(batch_x).float().to(self.device)
        batch_cluster_labels = torch.from_numpy(batch_cluster_labels).to(self.device)
        losses, elbo = self.forward_train_mask(batch_x, batch_cluster_labels, w=vae_samples)

        return losses, elbo

    def train_ordering_loss_masked(self, batch_x, batch_cluster_labels, batch_Y_ci, ordering_criterion):
        """
        Take a batch of data with varying number of clusters (k), the cluster labels and cluster order,
        return a clustering loss after splitting into minibatches for same k.
        """
        # at this point we want the data to be tensors
        data = torch.from_numpy(batch_x).float().to(self.device)
        labels = torch.from_numpy(batch_cluster_labels).to(self.device)
        batch_Y_ci = batch_Y_ci.to(self.device)

        # get minibatches and row tracker per k
        minibatches, k_clusters_to_example_row = self.get_same_k_minibatches(data, labels)

        # track batch ordering loss
        minibatch_losses = []

        for minibatch in minibatches:
            # unpack
            minibatch_data = minibatch[0].to(self.device)
            minibatch_labels = minibatch[1].to(self.device)

            # current k (assume all in minibatch have same k)
            current_k = len(minibatch_labels[0].unique())

            # now use the model to obtain cluster encoding
            minibatch_clusters_encoded = self.forward_encode_clusters(minibatch_labels, minibatch_data)

            # now use set-2-seq to predict cluster order
            minibatch_o, minibatch_p = self.forward_set2seq(minibatch_clusters_encoded)

            # obtain minibatch loss & weigh it by number of examples (we can then sum)
            minibatch_Y_hat = minibatch_o.contiguous().view(-1, minibatch_o.size()[-1])

            # first, we need the right rows
            minibatch_Y_ci = torch.index_select(batch_Y_ci, 0, k_clusters_to_example_row[current_k])

            # these may have padding that will need to be removed
            minibatch_Y_ci = minibatch_Y_ci[:, :current_k]

            # and have proper type
            minibatch_Y_ci = minibatch_Y_ci.long().view(-1)
            minibatch_Y_ci = minibatch_Y_ci.to(self.device)

            # obtain minibatch loss
            minibatch_loss = ordering_criterion(minibatch_Y_hat, minibatch_Y_ci)

            # update batch loss
            minibatch_losses.append(minibatch_loss)

        # sum
        loss_ordering = torch.sum(torch.stack(minibatch_losses))

        return loss_ordering

    def get_same_k_minibatches(self, data: torch.Tensor, labels: torch.Tensor) -> (dict, list):
        """Take a batch of data and labels, with examples with different number
        of clusters k. Return a list of tensor minibatches with same number of k and
        a dictionary tracking how to combine them back into the original order"""

        # find indices of examples with same number of clusters k
        k_clusters_to_example_row = dict()
        for i, example in enumerate(labels):
            k = len(example.unique())
            if k in k_clusters_to_example_row:
                k_clusters_to_example_row[k].append(i)
            else:
                k_clusters_to_example_row[k] = [i, ]

        # need list of indices as tensor
        for k, indices in k_clusters_to_example_row.items():
            k_clusters_to_example_row[k] = torch.Tensor(indices).long().to(self.device)

        # turn those groups into mini-batches
        minibatches = []
        for k, example_rows in k_clusters_to_example_row.items():
            group_rows = k_clusters_to_example_row[k]

            # data minibatch
            group_data = torch.index_select(data, 0, group_rows)

            # label minibatch
            group_labels = torch.index_select(labels, 0, group_rows)

            minibatches.append((group_data, group_labels))

        return minibatches, k_clusters_to_example_row

    # the conditional prior
    def get_pz(self, anchor, U, G):

        mu_logstd = self.pz_mu_log_sigma(torch.cat((anchor, U, G), dim=1))
        mu = mu_logstd[:, :self.z_dim]
        log_sigma = mu_logstd[:, self.z_dim:]

        return mu, log_sigma

    # the conditional posterior
    def get_qz(self, anchor, U_in, U_out, G):

        mu_logstd = self.qz_mu_log_sigma(
            torch.cat((anchor, U_in, U_out, G), dim=1))

        mu = mu_logstd[:, :self.z_dim]
        log_sigma = mu_logstd[:, self.z_dim:]

        return mu, log_sigma

    # this function is not used during training, it is auxiliary for samplers
    def sample_z(self, anchor, U, G):

        mu, log_sigma = self.get_pz(anchor, U, G)
        std = log_sigma.exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_train(self, data, labels, w):
        """"Forward training iteration. Labels are shared across the mini-batch.
            Note: data and labels do not need to be pre-sorted. They will be sorted after encoder.
        Args:
            Data: torch.Tensor | torch_geometric.data.Batch | batched dgl.DGLGraph
                The data format needs to be consistent with what the encoder takes as input.
                The encoder output will always be a torch.Tensor of shape (batch_size, N, e_dim)
            labels: torch.Tensor of shape (N,)
            w: the number of VAE samples
        Return:
            loss, elbo
        """
        device = data.device
        N = len(labels)

        _, cluster_counts = torch.unique(
            labels, sorted=True, return_counts=True)  # Cluster counts is the number of elements per cluster, in order
        K = len(cluster_counts)
        all_anchors, all_targets, all_unassigned, all_last_assigned = get_ATUS(
            cluster_counts,
            device=device)  # these are teacher-forced indices of all initial cluster points, all available points, all points currently assigned and the target binary vector at each k-th step
        loss_clustering = 0
        loss_cardinality = 0
        elbo = 0

        # iterate k times and compute final loss
        enc_data = self.encoder(data).view([-1, N, self.e_dim])

        # sort the nodes by labels for CCP
        sorted_ind = torch.argsort(labels)
        labels = labels[sorted_ind]  # not clear what this achieves
        enc_data = enc_data[:, sorted_ind]

        G = None
        for k in range(K):
            # when the last cluster has only one element
            if k == K - 1 and len(all_unassigned[k]) == 0:
                break

            ind_anchor = all_anchors[k]
            ind_unassigned = all_unassigned[k]
            ind_last_assigned = all_last_assigned[k - 1] if k > 0 else None
            targets = all_targets[k].to(device)

            kl_loss, elbo_term, G, _ = self.forward_k(
                k, enc_data, G, ind_anchor, ind_unassigned, ind_last_assigned, w, targets)
            loss_clustering += kl_loss
            elbo += elbo_term

            # cardinality loss
            # current cluster cardinality prediction and loss
            if self.condition_cardinality_on_assigned:
                predicted_cardinality = self.predict_cardinality(enc_data, ind_anchor, ind_unassigned, G)
            else:
                predicted_cardinality = self.predict_cardinality(enc_data, ind_anchor, ind_unassigned)

            cc_loss = self.cardinality_loss(ind_anchor, predicted_cardinality, labels)
            loss_cardinality += cc_loss

        return {'loss_clustering': loss_clustering, 'loss_cardinality': loss_cardinality}, elbo

    def forward_train_mask(self, data, batch_labels, w):
        """"Forward training iteration using masks. Labels are different across the mini-batch.
            Note: data and labels do not need to be pre-sorted. They will be sorted after encoder.
        Args:
            Data: torch.Tensor | torch_geometric.data.Batch | batched dgl.DGLGraph
                The data format needs to be consistent with what the encoder takes as input.
                The encoder output will always be a torch.Tensor of shape (batch_size, N, e_dim)
            labels: List[torch.Tensor]
            w: the number of VAE samples
        Return:
            loss, elbo
        """
        device = data.device
        batch_size = len(batch_labels)

        Ns = []
        Ks = []
        max_N = 0
        max_K = 0
        batch_cluster_counts = []

        sorted_ind_batch = []
        sorted_batch_labels = []
        for i, labels in enumerate(batch_labels):
            sorted_ind = torch.argsort(labels)
            labels = labels[sorted_ind]
            sorted_ind_batch.append(sorted_ind)
            sorted_batch_labels.append(labels)
        batch_labels = sorted_batch_labels

        for labels in batch_labels:
            Ns.append(len(labels))
            max_N = max(len(labels), max_N)

            _, cluster_counts = torch.unique(
                labels, sorted=True, return_counts=True)
            Ks.append(len(cluster_counts))
            max_K = max(len(cluster_counts), max_K)
            batch_cluster_counts.append(cluster_counts)

        # print(max_K, max_N)
        batch_label_tensor = torch.zeros(batch_size, max_N) - 1
        # -1 is used for padding
        for i, labels in enumerate(batch_labels):
            batch_label_tensor[i, :Ns[i]] = labels

        batch_anchors, batch_targets, batch_unassigned, batch_assigned = \
            get_ATUS_batch(batch_cluster_counts, device=device)
        # (batch_size, max_K, max_N)

        enc_data_raw = self.encoder(data)  # [total_N, e_dim]

        # had to flatten manually here, not sure why
        enc_data_raw = torch.flatten(enc_data_raw, 0, 1)

        enc_data_raw = enc_data_raw.split(Ns, dim=0)  # split the batch

        # stack the encoded data into the same batch tensor with padding
        enc_data = torch.zeros(batch_size, max_N, self.e_dim).to(device)
        # the mask of indices corresponding to actual data in each batch
        mask = torch.zeros([batch_size, max_N]).to(device)
        for i in range(batch_size):
            sorted_ind = sorted_ind_batch[i]
            enc_data[i, :Ns[i]] = enc_data_raw[i][sorted_ind]
            mask[i, :Ns[i]] = 1

        Ks = torch.tensor(Ks)
        Ns = torch.tensor(Ns)

        # the number of unfinished training examples in each mini-batch
        t = batch_size
        G = None

        loss_clustering = 0
        loss_cardinality = 0
        elbo = 0
        # iterate k times and compute final loss
        for k in range(max_K):

            i = 0
            while i < t:
                # when no cluster left or the last cluster has only one element
                finished = (k == Ks[i]) \
                           or (k == Ks[i] - 1 and len(batch_unassigned[i][k]) == 0)
                if finished:
                    new_order = torch.cat(
                        [torch.arange(i), torch.arange(i + 1, t)])
                    enc_data = enc_data[new_order, :]
                    mask = mask[new_order, :]
                    G = G[new_order, :]
                    Ks = Ks[new_order]
                    Ns = Ns[new_order]
                    batch_anchors = batch_anchors[new_order]
                    batch_targets = batch_targets[new_order]
                    batch_unassigned = batch_unassigned[new_order]
                    batch_assigned = batch_assigned[new_order]
                    t -= 1  # i stays the same
                else:
                    i += 1
                # all finished
                if t == 0:
                    break

            anchor = batch_anchors[:, k]
            unassigned = batch_unassigned[:, k]
            last_assigned = batch_assigned[:, k - 1] if k > 0 else None
            targets = batch_targets[:, k].to(data.device)

            loss_term, elbo_term, G, _ = self.forward_k_mask(
                k, enc_data, G, anchor, unassigned, last_assigned, w, mask, targets)

            loss_clustering += loss_term
            elbo += elbo_term

            if self.condition_cardinality_on_assigned:
                predicted_cardinality = self.predict_cardinality_batched(enc_data, anchor, unassigned, G)
            else:
                predicted_cardinality = self.predict_cardinality_batched(enc_data, anchor, unassigned)

            cc_loss = self.cardinality_loss(predicted_cardinality, targets)
            loss_cardinality += cc_loss

        return {'loss_clustering': loss_clustering, 'loss_cardinality': loss_cardinality}, elbo

    def forward_k(self, k, enc_data, G, ind_anchor, ind_unassigned, ind_last_assigned,
                  w, targets=None):
        """
        k: the cluster number to sample in this call
        data: [batch_size, N, ...]
        anchor: the anchor point
        ind_unassigned: numpy integer array with the indices of the unassigned points
        ind_in_clusters: list, where ind_in_clusters[k] is a numpy integer array with the indices of the points in cluster k.
        w: number of VAE samples

        Output:
            logits for sampling the binary variables to join cluster k
        """
        # print(k)

        G = self.update_global(enc_data, ind_last_assigned, G, k)

        anch, data_unassigned, us_unassigned, U = self.encode_unassigned(
            enc_data, ind_anchor, ind_unassigned)

        pz_mu, pz_log_sigma = self.get_pz(anch, U, G)

        if targets is None:
            z = pz_mu  # + torch.randn_like(pz_mu)*torch.exp(pz_log_sigma)
            z = z.unsqueeze(0)
            logits = self.vae_likelihood(z, U, G, anch, data_unassigned)
            return logits.squeeze(-1)

        qz_mu, qz_log_sigma, z = self.conditional_posterior(
            us_unassigned, G, anch, targets, w)

        logits = self.vae_likelihood(z, U, G, anch, data_unassigned)

        loss_kl, elbo = self.kl_loss(
            qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets)

        return loss_kl, elbo, G, logits

    def forward_k_mask(self, k, enc_data, G, anchor, unassigned, last_assigned,
                       w, mask, targets=None):
        """
        k: the cluster number to sample in this call
        data: [batch_size, N, ...]
        anchor: the anchor point
        unassigned: numpy integer array with the indices of the unassigned points
        assigned: list, where assigned[k] is a numpy integer array with the indices of the points in cluster k.
        w: number of VAE samples

        Output:
            logits for sampling the binary variables to join cluster k
        """
        # print(k)

        G = self.update_global_mask(enc_data, last_assigned, G, k)

        anch, data_unassigned, us_unassigned, U = self.encode_unassigned_mask(
            enc_data, anchor, unassigned)

        pz_mu, pz_log_sigma = self.get_pz(anch, U, G)

        if targets is None:
            z = pz_mu  # + torch.randn_like(pz_mu)*torch.exp(pz_log_sigma)
            z = z.unsqueeze(0)
            logits = self.vae_likelihood(z, U, G, anch, data_unassigned)
            return logits.squeeze(-1)

        qz_mu, qz_log_sigma, z = self.conditional_posterior_mask(
            us_unassigned, G, anch, targets, unassigned, w)

        logits = self.vae_likelihood(z, U, G, anch, data_unassigned)

        loss, elbo = self.kl_loss_mask(
            qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets, unassigned)

        return loss, elbo, G, logits

    def update_global(self, enc_data, ind_last_assigned, G, k):

        if k == 0:
            G = torch.zeros([enc_data.shape[0], self.g_dim]
                            ).to(enc_data.device)
        else:
            hs_last_cluster = self.h(enc_data[:, ind_last_assigned, :])
            if self.use_attn:
                G += self.g(self.pma_h(hs_last_cluster).squeeze(dim=1))
            else:
                G += self.g(hs_last_cluster.mean(dim=1))
        return G

    def update_global_mask(self, enc_data, last_assigned, G, k):

        if k == 0:
            G = torch.zeros([enc_data.shape[0], self.g_dim]
                            ).to(enc_data.device)
        else:
            hs = self.h(enc_data)
            if self.use_attn:
                hs_mean = self.pma_h(hs, last_assigned).squeeze(dim=1)
            else:
                hs_mean = self.masked_mean(hs, last_assigned)
            G += self.g(hs_mean)
        return G

    def encode_unassigned(self, enc_data, ind_anchor, ind_unassigned):
        if self.use_attn:
            enc_all = enc_data[:, torch.cat(
                [torch.LongTensor([ind_anchor]).to(ind_unassigned.device), ind_unassigned]), :]  # Added by Pallab
            HX = self.ISAB1(enc_all)
            anch = HX[:, 0, :]
            us_unassigned = HX[:, 1:, :]
            # us_pma_input = self.MAB(us_unassigned,anch.unsqueeze(1))
            us_pma_input = self.MAB(HX, anch.unsqueeze(1))
            U = self.pma_u(us_pma_input).squeeze(dim=1)
            data_unassigned = us_unassigned
        else:
            anch = enc_data[:, ind_anchor, :]
            data_unassigned = enc_data[:, ind_unassigned, :]
            us_unassigned = self.u(enc_data[:, ind_unassigned, :])
            U = us_unassigned.mean(dim=1)  # [batch_size, u_dim]
        return anch, data_unassigned, us_unassigned, U

    def encode_unassigned_mask(self, enc_data, anchor, unassigned):
        if self.use_attn:
            anchor_and_unassigned = anchor | unassigned
            HX = self.ISAB1(enc_data, anchor_and_unassigned)
            anch = HX[anchor]
            us_unassigned = HX
            # us_pma_input = self.MAB(us_unassigned,anch.unsqueeze(1))
            us_pma_input = self.MAB(HX, anch.unsqueeze(1))  # no mask on anchor
            U = self.pma_u(us_pma_input, anchor_and_unassigned).squeeze(dim=1)
            data_unassigned = us_unassigned
        else:
            anch = enc_data[anchor]
            data_unassigned = enc_data
            us_unassigned = self.u(enc_data)
            U = self.masked_mean(us_unassigned, unassigned)
        return anch, data_unassigned, us_unassigned, U

    def masked_mean(self, tensor, mask):
        mask_sum = mask.sum(dim=1, keepdim=True)
        positives = mask_sum > 0
        mask_mean = torch.zeros_like(mask_sum).float()
        mask_mean[positives] = 1 / mask_sum[positives].float()
        return (tensor * mask.unsqueeze(-1) * mask_mean.unsqueeze(-1)).sum(dim=1)

    def conditional_posterior(self, us_unassigned, G, anch, targets, w):
        device = G.device
        t_in = targets.type(torch.BoolTensor)
        reduced_shape = (us_unassigned.shape[0], us_unassigned.shape[2])

        if torch.all(~t_in):  # all False, U_in should be zero
            U_in = torch.zeros(reduced_shape).to(device)
        else:
            if self.use_attn:
                U_in = self.pma_u_in(us_unassigned[:, t_in, :]).squeeze(1)
            else:
                U_in = us_unassigned[:, t_in, :].mean(dim=1)

        if torch.all(t_in):  # all True, U_out should be zero
            U_out = torch.zeros(reduced_shape).to(device)
        else:
            if self.use_attn:
                U_out = self.pma_u_out(
                    us_unassigned[:, ~t_in, :]).squeeze(1)
            else:
                U_out = us_unassigned[:, ~t_in, :].mean(dim=1)

        # [batch_size, z_dim], [batch_size, z_dim]
        qz_mu, qz_log_sigma = self.get_qz(anch, U_in, U_out, G)

        qz_b = dist.Normal(qz_mu, qz_log_sigma.exp())
        z = qz_b.rsample(torch.Size([w]))  # [w,batch_size, z_dim]
        return qz_mu, qz_log_sigma, z

    def conditional_posterior_mask(self, us_unassigned, G, anch, targets, unassigned, w):
        device = G.device
        reduced_shape = (us_unassigned.shape[0], us_unassigned.shape[2])
        invert_targets = unassigned & ~targets

        U_in = torch.zeros(reduced_shape).to(device)
        zero_mask = torch.all(~(targets & unassigned), dim=1)
        if self.use_attn:
            U_in[~zero_mask] = self.pma_u_in(
                us_unassigned[~zero_mask], targets[~zero_mask]).squeeze(1)
        else:
            U_in[~zero_mask] = self.masked_mean(
                us_unassigned[~zero_mask], targets[~zero_mask])

        U_out = torch.zeros(reduced_shape).to(device)
        zero_mask = torch.all(targets == unassigned, dim=1)
        if self.use_attn:
            U_out[~zero_mask] = self.pma_u_out(
                us_unassigned[~zero_mask], invert_targets[~zero_mask]).squeeze(1)
        else:
            U_out[~zero_mask] = self.masked_mean(
                us_unassigned[~zero_mask], invert_targets[~zero_mask])

        # [batch_size, z_dim], [batch_size, z_dim]
        qz_mu, qz_log_sigma = self.get_qz(anch, U_in, U_out, G)

        qz_b = dist.Normal(qz_mu, qz_log_sigma.exp())
        z = qz_b.rsample(torch.Size([w]))  # [w,batch_size, z_dim]
        return qz_mu, qz_log_sigma, z

    def vae_likelihood(self, z, U, G, anch, data_unassigned):
        w, batch_size = z.shape[0], z.shape[1]
        Lk = data_unassigned.shape[1]
        expand_shape = (-1, Lk, w, -1)

        zz = z.transpose(0, 1)  # [batch_size, w, z_dim]
        zz = zz.view(batch_size, 1, w, -1).expand(expand_shape)
        dd = data_unassigned.view(batch_size, Lk, 1, -1).expand(expand_shape)
        aa = anch.view(batch_size, 1, 1, -1).expand(expand_shape)
        UU = U.view(batch_size, 1, 1, -1).expand(expand_shape)
        GG = G.view(batch_size, 1, 1, -1).expand(expand_shape)

        ddzz = torch.cat([dd, zz, aa, UU, GG], dim=3).view(
            [batch_size * Lk * w, self.phi_input_dim])
        logits = self.phi(ddzz).view([batch_size, Lk, w])
        return logits

    def kl_loss(self, qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets):
        # For the loss function we use below the doubly-reparametrized gradient estimator from https://arxiv.org/abs/1810.04152
        # as implemented in https://github.com/iffsid/DReG-PyTorch

        pb_z = dist.Bernoulli(logits=logits)
        batch_size, Lk, w = logits.shape
        targets = targets.view(1, Lk, 1).expand(batch_size, Lk, w)
        lpb_z = pb_z.log_prob(targets).sum(dim=1)  # [batch_size,w]
        lpb_z.transpose_(0, 1)

        # in dreg qz_b is not differentiated
        qz_b_ = dist.Normal(qz_mu.detach(), qz_log_sigma.detach().exp())
        lqz_b = qz_b_.log_prob(z).sum(-1)  # [w,batch_size]

        lpz = dist.Normal(pz_mu, pz_log_sigma.exp()).log_prob(
            z).sum(-1)  # [w,batch_size]

        lw = lpz + lpb_z - lqz_b  # [w,batch_size]

        with torch.no_grad():
            reweight = torch.exp(lw - torch.logsumexp(lw + 1e-30, 0))
            # reweight = F.softmax(lw, dim=0)
            if self.training:
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

        loss = -(reweight * lw).sum(0).mean(0)

        le = torch.exp(lw).mean(dim=0) + 1e-30  # mean over w terms
        elbo = torch.log(le).mean()  # mean over minibatch

        return loss, elbo

    def kl_loss_mask(self, qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets, unassigned):
        # For the loss function we use below the doubly-reparametrized gradient estimator from https://arxiv.org/abs/1810.04152
        # as implemented in https://github.com/iffsid/DReG-PyTorch

        pb_z = dist.Bernoulli(logits=logits)
        batch_size, Lk, w = logits.shape
        targets = targets.unsqueeze(-1).expand(batch_size, Lk, w)
        lpb_z = (pb_z.log_prob(targets.float()) *
                 unassigned.unsqueeze(-1)).sum(dim=1)  # [batch_size,w]
        lpb_z.transpose_(0, 1)

        # in dreg qz_b is not differentiated
        qz_b_ = dist.Normal(qz_mu.detach(), qz_log_sigma.detach().exp())
        lqz_b = qz_b_.log_prob(z).sum(-1)  # [w,batch_size]

        lpz = dist.Normal(pz_mu, pz_log_sigma.exp()).log_prob(z).sum(-1)
        # [w,batch_size]

        lw = lpz + lpb_z - lqz_b  # [w,batch_size]

        with torch.no_grad():
            reweight = torch.exp(lw - torch.logsumexp(lw + 1e-30, 0))
            # reweight = F.softmax(lw, dim=0)
            if self.training:
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

        loss = -(reweight * lw).sum(0).mean(0)

        le = torch.exp(lw).mean(dim=0) + 1e-30  # mean over w terms
        elbo = torch.log(le).mean()  # mean over minibatch

        return loss, elbo

    def predict_cardinality(self, enc_anchor, enc_unassigned, G=None):
        """Take the encoded anchor and encoded unassigned points, optionally representation
        of previous clusters, predict the cardinality of the cluster with the anchor and return it"""

        # concatenate into enc_anch_and_unassigned, starting with anchor
        enc_anch_and_unassigned = torch.cat([enc_anchor, enc_unassigned], dim=1)

        HX = self.ISAB_c(enc_anch_and_unassigned)
        anch = HX[:, 0, :]
        cc_pma_input = self.MAB_c(HX, anch.unsqueeze(1))
        CC = self.pma_c(cc_pma_input).squeeze(dim=1)
        # linear regression
        # depending on whether we're also conditioning on G
        if self.condition_cardinality_on_assigned:
            cc = self.c(torch.cat([CC, G], dim=1))
        else:
            cc = self.c(CC)
        return cc

    def predict_cardinality_batched(self, enc_data, anchors, unassigned, G=None, is_inference=False):
        """Version of cardinality prediction that should work with forward_masked, to speed up
        training by running on batches in parallel. Has to account for each example in batch having
        a different amount of opened clusters"""

        if is_inference:
            # t obtained from anchors
            t = anchors.size(0)
            N = enc_data.size(1)

            # enc and mask need to be cut down to t, which we know from anchors
            # and big_enc_data is the same for each of the samples
            enc_data = enc_data[:t, :, :].to(self.device)
            G = G[:t, :]

            # mask is ordered according to which ones are still open (closed are pushed to the end)
            # 1s in the masks are available indices
            unassigned = unassigned[:t, :].to(self.device)

            # turn to expected format
            unassigned = unassigned.bool()
            a = torch.zeros(t, N).bool()
            for i, e in enumerate(anchors):
                a[i][e] = True
            anchors = a.to(self.device)

        # loop over each scampled enc data row, accesing mask to get ind_unassigned, call self.predict_cardinality and stack
        emb_dim = enc_data.size(2)
        batch_size = enc_data.size(0)

        # group examples by number of unassigned points
        num_unassigned_to_row_number = dict()
        for i, ex_un in enumerate(unassigned):
            n_true = ex_un.nonzero(as_tuple=True)[0].size(0)
            if n_true in num_unassigned_to_row_number:
                num_unassigned_to_row_number[n_true].append(i)
            else:
                num_unassigned_to_row_number[n_true] = [i, ]

        # perform the calculation for each group, then concatenate according to the indices
        # need a placeholder for predicted cardinalities
        predicted_cardinalities = torch.zeros(batch_size).to(self.device)

        # select the right rows of unassigned (with same n_true)
        for n_true, example_inds in num_unassigned_to_row_number.items():
            example_inds = torch.Tensor(example_inds).long().to(self.device)

            # extract shapes
            n_examples = example_inds.size(0)

            # get the examples, anchors and unassigned with same n_true
            enc_n_true_group = enc_data[example_inds, :, :]
            anch_n_true_group = anchors[example_inds, :]
            unassigned_n_true_group = unassigned[example_inds, :]

            # masks overs entire embedded data are needed
            unassigned_mask = unassigned_n_true_group.unsqueeze(2)
            anchor_mask = anch_n_true_group.unsqueeze(2)

            # repeat into the embedding dim
            unassigned_mask = torch.tile(unassigned_mask, (1, 1, emb_dim))
            anchor_mask = torch.tile(anchor_mask, (1, 1, emb_dim))

            # flatten everything
            flat_enc_data = torch.flatten(enc_n_true_group, 0, 2)
            flat_unassigned_mask = torch.flatten(unassigned_mask, 0, 2)
            flat_anchor_mask = torch.flatten(anchor_mask, 0, 2)

            # use mask select to get batched anchors and unassigned elements
            flat_unassigned_data = torch.masked_select(flat_enc_data, flat_unassigned_mask)
            flat_anchor_data = torch.masked_select(flat_enc_data, flat_anchor_mask)

            # reshape
            enc_anchor = torch.reshape(flat_anchor_data, (n_examples, 1, emb_dim))
            enc_unassigned = torch.reshape(flat_unassigned_data, (n_examples, n_true, emb_dim))

            # actually predict cardinality
            if self.condition_cardinality_on_assigned:
                G_same_n_true = G[example_inds, :]
                cc_n_true = self.predict_cardinality(enc_anchor, enc_unassigned, G=G_same_n_true)
            else:
                cc_n_true = self.predict_cardinality(enc_anchor, enc_unassigned)

            # update the tracker
            predicted_cardinalities[example_inds] = cc_n_true.squeeze(1)

        # if inference, slightly reshape and round to the nearest integer
        if is_inference:
            predicted_cardinalities = predicted_cardinalities.unsqueeze(1)

            # round to the nearest integer
            predicted_cardinalities = predicted_cardinalities.round().long()

        return predicted_cardinalities

    def cardinality_loss(self, predicted_cardinalities, targets):
        # sum bools to get target cardinalities per example
        target_cardinalities = torch.sum(targets, dim=1).float()

        # calculate loss
        current_cardinality_loss = self.c_loss(predicted_cardinalities, target_cardinalities)

        return current_cardinality_loss

    def forward_encode_clusters(self, cs, data):
        """
        Obtain cluster encodings for batches with examples that have a varying number of target clusters.
        :param cs: target or predicted cluster assignments (cs)
        :param data - raw data for the examples
        :return: aggregated clusters (batch, K, c_dim)
        """
        # get the number of clusters for this minibatch (assume same number of clusters per mini-batch for all examples)
        K = len(cs[0].unique())

        # encode the raw data the CNOC way
        enc_data = self.encoder(data)

        # placeholder
        minibatch_clusters_aggregated = []

        # we are not guaranteed the same number of elements per cluster in each example,
        # so we have to go example by example
        for i, example in enumerate(enc_data):
            example_clusters_aggregated = []

            # aggregate each cluster, using given cluster assignments
            for k in range(K):
                # find indices of the points belonging to current cluster k
                k_indices = torch.nonzero(cs[i] == k).squeeze(1)

                # transform and aggregate using PMA
                k_points = self.h(enc_data[i, k_indices, :]).unsqueeze(0)

                # assigned path
                k_cluster_assigned = self.pma_h(k_points)

                # unassigned path
                HX = self.ISAB1(k_points)
                anch = HX[:, 0, :]
                # us_unassigned = HX[:, 1:, :]
                us_pma_input = self.MAB(HX, anch.unsqueeze(1))
                k_cluster_unassigned = self.pma_u(us_pma_input)

                # join the assigned and unassigned representations
                k_cluster = torch.cat([k_cluster_assigned, k_cluster_unassigned], dim=2)

                # join with unassigned before aggregating
                example_clusters_aggregated.append(k_cluster)

            # turn into one tensor
            example_clusters_aggregated = torch.cat(example_clusters_aggregated, dim=1)
            minibatch_clusters_aggregated.append(example_clusters_aggregated)

        # aggregate
        minibatch_clusters_aggregated = torch.cat(minibatch_clusters_aggregated, dim=0)

        return minibatch_clusters_aggregated

    def forward_set2seq(self, inputs, target=None):
        """
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :param Tensor target: (optional, only for training future+history permutation module)
        :return: Pointers probabilities and indices [or losses if future+history and in training mode)
        """
        # inputs: (batch x seq_len x elem_dim)
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # embed / resize inputs
        if self.device == torch.device('cuda:0'):
            inputs = inputs.to(torch.device('cuda:0'))
        embedded_inputs = self.elem_embedder(inputs)

        # set encoding
        set_encoding = self.set_embedding(embedded_inputs)
        embedded_set = set_encoding['embedded_set']

        # (optional) context
        # get encoder outputs as context via rnn
        if self.context_rnn_used:
            set_context = self.context_rnn(embedded_inputs)
            context = set_context['encoder_outputs']
        else:
            context = None

        # obtain first decoder hidden state
        decoder_hidden0 = torch.split(embedded_set, self.set_encoder_dim // 2,
                                      dim=1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # pass to a permutation module

        # have to handle training and pure inference differently
        # due to the future+history permutation module possibility
        # if no y (target) has been passed, proceed as normal

        # if self.permute_module_type == 'futurehistory' and target is not None:
        #     loss, (point_loss, rela_loss, left1_loss, left2_loss) = self.decoder.forward_train(
        #         inputs,
        #         decoder_hidden0,
        #         target)
        #     return loss, (point_loss, rela_loss, left1_loss, left2_loss)

        # otherwise, proceed normally
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs,
            decoder_input0,
            decoder_hidden0,
            context)
        return outputs, pointers

    def infer(self, an_example, n_samples=10):
        """
        Function for taking a trained model and making a prediction
        on a single-example batch
        :param an_example: single example from a data generator
        :param n_samples: number of samples to generate (originally 5K)
        :return: cluster assignments and cluster order.
        """
        # initiate sampler, generate cluster predictions
        acp_sampler = CCC_Sampler(self, an_example, device=self.device)
        css, probs = acp_sampler.sample(n_samples,
                                        sample_Z=False, sample_B=False,
                                        prob_nZ=1, prob_nA=10)

        # use only the best prediction going forward
        cluster_assignments = css[np.argmax(probs)]

        # embed the clusters according to the prediction
        an_example = torch.from_numpy(an_example).float().to(self.device)
        cluster_assignments = torch.from_numpy(cluster_assignments).to(self.device).unsqueeze(0)
        clusters_encoded = self.forward_encode_clusters(cluster_assignments, an_example)

        # # debug (remove)
        # for c in clusters_encoded[0]:
        #     print(c[:6].detach().numpy())

        # get the cluster order
        attentions, cluster_order = self.forward_set2seq(clusters_encoded)

        # relabel, according to predicted cluster order (for ordering metrics)
        cluster_assignments = cluster_assignments.squeeze(0)
        cluster_assignments = cluster_assignments.cpu().numpy()
        cluster_assignments_relabelled = get_ordered_cluster_assignments(cluster_assignments, cluster_order[0])

        # return
        return cluster_assignments_relabelled, cluster_order


def get_noc_mog_encoder(params):
    return NOC_MOG_Encoder(
        in_dim=params['x_dim'],
        out_dim=params['e_dim'],
        H_dim=params['H_dim'])


class NOC_MOG_Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, H_dim):
        super().__init__()

        H = H_dim
        self.h = torch.nn.Sequential(
            torch.nn.Linear(in_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, out_dim),
        )

    def forward(self, x):
        return self.h(x)


def get_NOC_synth_encoder(params):
    return NOC_Synthetic_Structures_encoder(params)


def get_NOC_procat_encoder(params):
    return NOC_PROCAT_encoder(params)


class NOC_Synthetic_Structures_encoder(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.out_dim = params['e_dim']
        self.h_dim = params['H_dim']

        if params['add_set_repr_to_elems']:
            # if we are to add a perm-invar representation of the entire set to individual elements
            # we first need to obtain it
            self.h = NOC_Synthetic_Structures_Inner_Encoder_With_Set_Repr_Inlcusion(params)
        else:
            # otherwise, we don't add the perm-invar representation of elements to the per-elem representations
            self.h = torch.nn.Sequential(
                torch.nn.Embedding(params['vocab_size'], self.h_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(self.h_dim, self.h_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(self.h_dim, self.h_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(self.h_dim, self.h_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(self.h_dim, self.out_dim),
            )

    def forward(self, x):
        x = x.long()
        return self.h(x)


class NOC_PROCAT_encoder(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.out_dim = params['e_dim']
        self.h_dim = params['H_dim']

        self.h = OfferEmbedderBertCased(params)

    def forward(self, x):
        x = x.long()
        return self.h(x)


class OfferEmbedderBertCased(nn.Module):
    """
    An offer embedding module using pretrained danish-bert-botxo.
    Outputs an embedding per sentence passed.
    """

    def __init__(self):
        super(OfferEmbedderBertCased, self).__init__()
        self.language_model = AutoModelForPreTraining. \
            from_pretrained('danish-bert-botxo',
                            from_tf=True,
                            output_hidden_states=True)

    def forward(self, tokenized_bert_batch):
        z = self.language_model(**tokenized_bert_batch)
        z = [z[2][i] for i in (-1, -2, -3, -4)]  # last 4 layers, concat, mean
        z = torch.cat(tuple(z), dim=-1)
        z = torch.mean(z, dim=1)
        return z

    @staticmethod
    def prepare_procat_batch(batch):
        """
        Take a 200-offer PROCAT split batch, return 200 separate batches that the language
        model will understand.
        """
        per_offer_batches = [
            {'input_ids': batch['s{}_input_ids'.format(i + 1)],
             'token_type_ids': batch['s{}_token_type_ids'.format(i + 1)],
             'attention_mask': batch['s{}_attention_mask'.format(i + 1)]}
            for i in range(200)]
        return per_offer_batches


class NOC_Synthetic_Structures_Inner_Encoder_With_Set_Repr_Inlcusion(nn.Module):
    """Module for adding the representation of the entire set to per-elem representations"""

    def __init__(self, params):
        super().__init__()
        H = params['H_dim']
        self.h_dim = params['h_dim']
        self.out_dim = params['e_dim']

        self.initial_embedding = nn.Embedding(params['vocab_size'], self.h_dim)
        self.set_repr_layers = nn.Sequential(SABs2s(self.h_dim, self.h_dim, num_heads=4, ln=False))
        self.set_pooling = PMAs2s(self.h_dim, num_heads=4, num_seeds=1)
        self.h_remaining = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(self.h_dim * 2, self.h_dim),  # 2 because we concatenate
            torch.nn.PReLU(),
            torch.nn.Linear(self.h_dim, self.h_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(self.h_dim, self.h_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(self.h_dim, self.out_dim),
        )

    def forward(self, x):
        z = self.initial_embedding(x)
        s = self.set_repr_layers(z)
        s = self.set_pooling(s)
        # repeat for concatenation
        s = torch.tile(s, (1, z.size(1), 1))
        zs = torch.cat([z, s], dim=2)
        r = self.h_remaining(zs)  # if addition just add
        return r


def get_ATUS(clusters, device=None):
    """Get the anchors, targets, unsassigned and assigned indices at each of the k steps during training.

    Args:
        clusters: list[int] -- sizes of all clusters

    Returns:
        anchors: Tensor(k) -- indices of anchors at each k.
        targets: list[Tensor(N-len(assigned))] -- each tensor is the prediction target at each
            training step k, i.e. a binary mask of points in cluster k except
            the anchor point. Shortened to start at the anchor and excluded already
            assigned points.
        unassigned: list[Tensor(N)] -- each tensor is the indices of unassigned
            points at step k.
        assigned: list[Tensor(N)] -- each tensor is the indices of assigned points
            in cluster k. assigned[k-1] is the indices of the last assigned points
            at training step k.
    """
    assert (torch.all(clusters > 0))
    N = torch.sum(clusters)
    K = len(clusters)
    clusters = clusters.to(device)

    anchors = torch.zeros(K, dtype=torch.int32, device=device)
    targets = []
    unassigned = []
    assigned = []

    cumsum = torch.cumsum(torch.cat([torch.LongTensor([0]).to(device), clusters]), dim=0)

    for k in range(K):
        anchors[k] = cumsum[k]
        available = torch.arange(cumsum[k] + 1, N).to(device)
        unassigned.append(available)
        # indices of the elements in cluster k
        assigned_in_k = torch.arange(cumsum[k], cumsum[k + 1]).to(device)
        assigned.append(assigned_in_k)
        # (shortened) binary masks of elements in cluster k except the anchor
        target_k = torch.zeros(len(available)).to(device)
        target_k[:len(assigned_in_k) - 1] = 1
        targets.append(target_k)

    return anchors, targets, unassigned, assigned


def get_ATUS_batch(batch_clusters, device=None):
    """Get the anchors, targets, unsassigned and assigned indices at each of the k steps during training.
        This is the mini-batch version of get_ATUS

    Args:
        clusters: list[int] -- sizes of all clusters

    Returns:
        anchors: Tensor(b, max_K, max_N) -- binary mask of anchor indices
        targets: Tensor(b, max_K, max_N) -- each tensor along dim=1 is the prediction target at each
            training step k, i.e. a binary mask of points in cluster k except
            the anchor point.
        unassigned: Tensor(b, max_K, max_N) -- each tensor is the binary mask of unassigned
            points at step k.
        assigned: Tensor(b, max_K, max_N) -- each tensor is the binary mask of assigned points
            in cluster k. assigned[k-1] is the binary mask of the last assigned points
            at training step k.
    """
    batch_size = len(batch_clusters)
    max_N = max([torch.sum(clusters) for clusters in batch_clusters])
    max_K = max([len(clusters) for clusters in batch_clusters])
    batch_anchors, batch_targets, batch_unassigned, batch_assigned = \
        [torch.zeros(batch_size, max_K, max_N, dtype=torch.bool, device=device)
         for _ in range(4)]

    for b, clusters in enumerate(batch_clusters):
        N = torch.sum(clusters)
        K = len(clusters)
        cumsum = torch.cumsum(
            torch.cat([torch.LongTensor([0]).to(device), clusters.to(device)]), dim=0)
        for k in range(K):
            batch_anchors[b, k, cumsum[k]] = 1
            batch_targets[b, k, cumsum[k] + 1:cumsum[k + 1]] = 1
            batch_unassigned[b, k, cumsum[k] + 1:N] = 1
            batch_assigned[b, k, cumsum[k]:cumsum[k + 1]] = 1
    return batch_anchors, batch_targets, batch_unassigned, batch_assigned


class MAB(nn.Module):
    def __init__(self, dim_X, dim_Y, dim, num_heads=4, ln=False, p=None, residual=True):
        super().__init__()

        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_X, dim)
        self.fc_k = nn.Linear(dim_Y, dim)
        self.fc_v = nn.Linear(dim_Y, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()

        self.residual = residual

    def forward(self, X, Y, mask=None):

        # X.shape = [batch,nx, dim_X]
        # Y.shape = [batch,ny, dim_Y]
        # mask.shape = [batch,ny]

        Q, K, V = self.fc_q(X), self.fc_k(Y), self.fc_v(Y)
        # Q.shape = [batch,nx,dim]
        # K.shape = [batch,ny,dim]
        # V.shape = [batch,ny,dim]

        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)
        # Q_.shape = [batch*nh,nx,dim/nh]    nh=num_heads
        # K_.shape = [batch*nh,ny,dim/nh]    nh=num_heads
        # V_.shape = [batch*nh,ny,dim/nh]    nh=num_heads

        A_logits = (Q_ @ K_.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        # A_logits.shape = [batch*nh,nx,ny]
        if mask is not None:
            mask = torch.stack([mask] * Q.shape[-2], -2)  # [batch,nx,ny]
            mask = torch.cat([mask] * self.num_heads, 0)  # [batch*nh,nx,ny]
            A_logits.masked_fill_(mask == 0, -float('inf'))
            A = torch.softmax(A_logits, -1)
            # to prevent underflow due to no attention
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)

        attn = torch.cat((A @ V_).chunk(
            self.num_heads, 0), -1)  # [batch, nx, dim]
        if self.residual:
            O = self.ln1(Q + self.dropout1(attn))
            O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        else:
            O = self.ln1(self.dropout1(attn))
            O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        return O


class SAB(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.mab = MAB(dim_X, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)


class SABs2s(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SABs2s, self).__init__()
        self.mab = MABs2s(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class MABs2s(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MABs2s, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class StackedSAB(nn.Module):
    def __init__(self, dim_X, dim, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SAB(dim_X, dim, **kwargs)] +
            [SAB(dim, dim, **kwargs)] * (num_blocks - 1))

    def forward(self, X, mask=None):
        for sab in self.blocks:
            X = sab(X, mask=mask)
        return X


class PMA(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(num_inds, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        I = self.I if X.dim() == 2 else self.I.repeat(X.shape[0], 1, 1)
        return self.mab(I, X, mask=mask)


class PMAs2s(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMAs2s, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class ISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.pma = PMA(dim_X, dim, num_inds, **kwargs)
        self.mab = MAB(dim_X, dim, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, self.pma(X, mask=mask))


class StackedISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ISAB(dim_X, dim, num_inds, **kwargs)] +
            [ISAB(dim, dim, num_inds, **kwargs)] * (num_blocks - 1))

    def forward(self, X, mask=None):
        for isab in self.blocks:
            X = isab(X, mask=mask)
        return X


class aPMA(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.I0 = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.I0)
        self.pma = PMA(dim, dim, 1, **kwargs)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, num_iters):
        I = self.I0
        for i in range(1, num_iters):
            I = torch.cat([I, self.pma(I)], 1)
        return self.mab(I.repeat(X.shape[0], 1, 1), X)


class SetEncoderSetTransformer(nn.Module):
    def __init__(self,
                 elem_embed_dim,
                 elem_embed_n_layers,
                 set_embed_num_heads,
                 set_embed_num_seeds,
                 set_embed_dim,
                 set_embed_n_layers
                 ):
        super(SetEncoderSetTransformer, self).__init__()

        # element embedding
        self.elem_embed_dim = elem_embed_dim
        self.elem_embed_n_layers = elem_embed_n_layers

        # set embedding
        self.set_embed_dim = set_embed_dim
        self.set_embed_n_layers = set_embed_n_layers
        self.set_embed_num_heads = set_embed_num_heads
        self.set_embed_num_seeds = set_embed_num_seeds

        # # first layer must be aware of input element dimensionality
        # # and type (e.g. are inputs dictionary indices)
        # self.first_layer = SetEncoderFirstLayer(self.elem_dim,
        #                                         self.elem_embed_dim,
        #                                         self.embedding_by_dict,
        #                                         self.embedding_by_dict_size)

        # remaining per-element layers
        self.remaining_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.elem_embed_dim,
                               self.elem_embed_dim,
                               num_heads=self.set_embed_num_heads,
                               ln=False)) for _ in
             range(self.elem_embed_n_layers)])

        # set embedding
        self.set_pooling = PMAs2s(self.elem_embed_dim, self.set_embed_num_heads,
                                  self.set_embed_num_seeds)
        self.set_embed_first_layer = SAB(self.elem_embed_dim,
                                         self.set_embed_dim,
                                         num_heads=self.set_embed_num_heads)
        self.set_embed_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.set_embed_dim,
                               self.set_embed_dim,
                               num_heads=self.set_embed_num_heads)) for _ in
             range(self.set_embed_n_layers)])

    def forward(self, X):

        # # per-elem embedding
        # Z = self.first_layer(X)

        for layer in self.remaining_layers:
            Z = layer(X)

        # set embedding
        Z = self.set_pooling(Z)
        Z = self.set_embed_first_layer(Z)
        for layer in self.set_embed_layers:
            Z = layer(Z)

        # in set transformer, squeeze here
        Z = Z.squeeze(1)
        return {'embedded_set': Z}


class SetEncoderRNN(nn.Module):
    def __init__(self,
                 elem_embed_dims,
                 set_embed_dim,
                 set_embed_rnn_layers,
                 set_embed_rnn_bidir,
                 set_embed_dropout):
        super(SetEncoderRNN, self).__init__()

        # element embedding
        self.elem_embed_dims = elem_embed_dims
        # set embedding
        self.set_embed_dims = set_embed_dim // 2 if set_embed_rnn_bidir else set_embed_dim
        self.set_embed_rnn_layers = set_embed_rnn_layers * 2 if set_embed_rnn_bidir else set_embed_rnn_layers
        self.set_embed_rnn_bidir = set_embed_rnn_bidir
        self.set_embed_dropout = set_embed_dropout

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

        # # first layer must be aware of input element dimensionality
        # # and type (e.g. are inputs dictionary indices)
        # self.first_layer = SetEncoderFirstLayer(self.elem_dims,
        #                                         self.elem_embed_dims,
        #                                         self.embedding_by_dict,
        #                                         self.embedding_by_dict_size)

        # actual set encoder
        self.fc1 = nn.Linear(self.elem_embed_dims, self.set_embed_dims)
        self.rnn = nn.LSTM(
            self.set_embed_dims,
            self.set_embed_dims,
            self.set_embed_rnn_layers,
            dropout=self.set_embed_dropout,
            bidirectional=self.set_embed_rnn_bidir,
            batch_first=True)
        self.fc2 = nn.Linear(self.set_embed_dims * 2, self.set_embed_dims)

    def forward(self, X):
        # # per-elem embedding
        # Z = self.first_layer(X)

        # adjust sizes
        Z = self.fc1(X)

        # get first hidden state and cell state
        hidden0 = self.init_hidden(Z)

        encoder_outputs, Z = self.rnn(Z, hidden0)

        if self.set_embed_rnn_bidir:
            # last layer's h and c only, concatenated
            Z = (torch.cat((Z[0][-2:][0], Z[0][-2:][1]), dim=-1),
                 torch.cat((Z[1][-2:][0], Z[1][-2:][1]), dim=-1))
        else:
            Z = (Z[0][-1], Z[1][-1])

        # concatenate (we're splitting them in the decoder)
        Z = torch.cat((Z[0], Z[1]), 1)

        # adjust size again
        Z = self.fc2(Z)

        return {'embedded_set': Z, 'encoder_outputs': encoder_outputs}

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.set_embed_rnn_layers,
                                                      batch_size,
                                                      self.set_embed_dims)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.set_embed_rnn_layers,
                                                      batch_size,
                                                      self.set_embed_dims)

        return h0, c0


class SetToSequencePointerAttention(nn.Module):
    """
    Attention mechanism for a Pointer Net. Implementation follows:
    https://github.com/shirgur/PointerNet/blob/master/PointerNet.py
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(SetToSequencePointerAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]),
                              requires_grad=False)
        self.soft = torch.nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # input: (batch, hidden)
        # context: (batch, seq_len, hidden)

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1,
                                                           context.size(1))

        # context: (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)

        # ctx: (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V: (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # att: (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # alpha: (batch, seq_len)
        alpha = self.soft(att)

        # hidden_state: (batch, hidden)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class PointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self,
                 elem_embed_dim,
                 set_embed_dim,
                 hidden_dim,
                 masking=True):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PointerDecoder, self).__init__()
        self.elem_embed_dim = elem_embed_dim
        self.set_embed_dim = set_embed_dim
        self.hidden_dim = hidden_dim
        self.masking = masking

        self.emb_inputs_resizer = nn.Linear(elem_embed_dim, hidden_dim)
        self.set_resizer_hidden = nn.Linear(set_embed_dim // 2, hidden_dim)
        self.set_resizer_cellstate = nn.Linear(set_embed_dim // 2, hidden_dim)
        self.input_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = SetToSequencePointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs or sth else
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # resize context and set
        embedded_inputs = self.emb_inputs_resizer(embedded_inputs)
        hidden = self.set_resizer_hidden(hidden[0]), self.set_resizer_cellstate(
            hidden[1])

        def step(x, hidden, context):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """
            # Regular LSTM
            # x: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h, c = hidden

            # gates: (batch, hidden * 4)
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            # h_t: (batch, hidden)
            # context: (batch, seq_len, hidden)
            # mask: (batch, seq_len)
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(
                self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            # decoder_input: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h_t, c_t, outs = step(decoder_input, hidden, context)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,
                                                                      outs.size()[
                                                                          1])).float()

            # Update mask to ignore seen indices, if masking is enabled
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,
                                                                  self.hidden_dim).byte()

            # Below line aims to fix:
            # UserWarning: indexing with dtype torch.uint8 is now deprecated,
            # please use a dtype torch.bool instead.
            embedding_mask = embedding_mask.bool()

            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.hidden_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden
