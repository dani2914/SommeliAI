""" Hierarchical Supervised LDA """

import logging

import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints
import pyro
from pyro.infer import TraceEnum_ELBO
import pyro.distributions as dist

from scipy.special import digamma
from scipy.optimize import minimize
import util

class MultiClassSupervisedLda:
    def __init__(self, num_classes, num_topics, num_vocabs):
        # hyperparameters for the priors

        self.C = num_classes
        self.K = num_topics
        self.V = num_vocabs
    
        # unfit hyperparameters
        self.alpha = torch.ones(self.K) / self.K

        # variational parameters
        self.gamma = []
        self.phi = []

        # other parameters
        self.pi = torch.ones((self.K, self.V)) / (self.K * self.V)
        self.eta = torch.ones((self.C, self.K)) / (self.C * self.K)
        self.ss = {}
        self.log_prob_w = torch.zeros((self.K, self.V))

    def v_em(self, docs_words, docs_counts, doc_labels):
        max_length = max([len(d) for d in docs_words])
        num_docs = len(docs_words)
        # words_per_document = (~np.isnan(docs_words)).sum(-1)

        # initializing with corpus level information
        self.gamma = torch.ones((num_docs, self.K))
        self.phi = torch.ones((max_length, self.K))
        self.ss["zbar_m"] = torch.zeros((num_docs, self.K), dtype=torch.float64)
        self.ss["zbar_var"] = torch.zeros(
            (num_docs, int(self.K*(self.K+1)/2)), dtype=torch.float64)
        self.ss["word"] = torch.zeros((self.K, self.V), dtype=torch.float64)
        self.ss["num_docs"] = 0
        self.ss["doc_labels"] = doc_labels

        likelihood = 0.
        for d in range(num_docs):
            if ((d % 1) == 0):
                print(f"document {d}")
            likelihood += self.doc_e_step(d, 
                docs_words[d, :],
                docs_counts[d, :],
                docs_labels[d],
                self.gamma[d, :],
                self.phi
            )

        print("mle!")
        self.mle()

        return likelihood

    def mle(self):
        word_total = np.nansum(self.ss["word"], axis=0)
        for k in range(self.K):
            mask = self.ss["word"][k, :] > 0
            self.log_prob_w[k, mask] = np.log(self.ss["word"][k, mask]) - np.log(word_total[mask])
            self.log_prob_w[k, ~mask] = -100.
        
        x0 = self.eta.flatten()
        
        x1 = minimize(self.softmax_f, x0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
        self.eta = x1.reshape(self.eta.shape)
        

    def doc_e_step(self, doc_ix, doc_words, doc_counts, doc_label, gamma, phi):
        likelihood = self.infer_one_doc(doc_words, doc_counts, doc_label, gamma, phi)
        
        # update sufficient statistics -- do this per document so we don't need to keep phi around

        mask = ~np.isnan(doc_words)
        doc_counts = torch.tensor(doc_counts)
        self.ss["word"][:, doc_words[mask]] += (
            doc_counts[mask] * phi[mask, :].T
        )
        self.ss["zbar_m"][doc_ix, :] += (
            doc_counts[mask] * phi[mask, :].T
        ).sum(1)
        self.ss["zbar_m"][doc_ix, :] /= (~np.isnan(doc_words)).sum()
        for k in range(self.K):
            for i in range(k, self.K):
                idx = util.map_idx(i, k, self.K)
                if i == k:
                    self.ss["zbar_var"][doc_ix, idx] += (
                        doc_counts[mask] * phi[mask, k].T * doc_counts[mask]
                    ).sum(axis=0)
                self.ss["zbar_var"][doc_ix, idx] -= (    
                    doc_counts[mask] * phi[mask, k].T * doc_counts[mask] * phi[mask, i]
                ).sum(axis=0)
        self.ss["zbar_var"][doc_ix, :] /= (~np.isnan(doc_words)).sum() ** 2.
        self.ss["num_docs"] += 1

        return likelihood

    def infer_one_doc(self, doc_words, doc_counts, doc_label, gamma, phi):
        # compute posterior dirichlet
        # K vec, sum out words
        FP_MAX_ITER = 10
        MAX_VAR_ITERS = 1

        gamma[:] = self.alpha + torch.from_numpy(np.nansum(self.phi, 0))
        digam = digamma(gamma)
        phi[:] = 1. / self.K
        doc_length = (~np.isnan(doc_words)).sum()


        sf_aux = torch.ones((self.C,), dtype=torch.float64)
        for l in range(self.C):
            for n in range(doc_length):
                t = (phi[n, :] * torch.exp(self.eta[l, :] * doc_counts[n]/np.nansum(doc_counts))).sum()
                sf_aux[l] *= t

        var_iter = 0
        converged = False
        while (not converged) and (var_iter < MAX_VAR_ITERS):
            var_iter += 1
            for n in range(doc_length):
                sf_params = torch.zeros((self.K,))
                for l in range(self.C):
                    t = sum([
                        phi[n, k] * torch.exp(self.eta[l, k] * 
                            doc_counts[n]/np.nansum(doc_counts)
                        ) for k in range(self.K)
                    ])
                    sf_aux[l] /= t  # take out word n

                    # h in the paper
 
                sf_params = torch.tensor([
                    (sf_aux * torch.exp(
                        self.eta[:, k] * doc_counts[n]/np.nansum(doc_counts)
                    )).sum() for k in range(self.K)
                ])

                oldphi = phi[n, :]
                for _ in range(FP_MAX_ITER):  # fixed point update
                    sf_val = 1.0  # the base class, in log space
                    for k in range(self.K):
                        sf_val += sf_params[k]*phi[n, k]

                    phisum = 0.                    
                    phi[n, :] = digam + self.log_prob_w[:, int(doc_words[n])]

                    # added softmax parts
                    if (doc_label < self.C-1):
                        phi[n, :] += self.eta[doc_label, :]/(np.nansum(doc_counts))
                    phi[n, :] -= sf_params/(sf_val*doc_counts[n])

                    # note, phi is in log space
                    phisum = sum([util.log_sum(
                                phisum, 
                                phi[n, k]
                            ) if k > 0 else
                            phi[n, k] 
                            for k in range(self.K)
                    ])
                    
                    phi[n, :] = np.exp(phi[n, :] - phisum)  # normalize

                # back to sf_aux value
                for l in range(self.C):
                    t = 0.0
                    for k in range(self.K):
                        t += phi[n, k] * np.exp(self.eta[l, k] * doc_counts[n]/(np.nansum(doc_counts)))
                    sf_aux[l] *= t

                gamma[:] = gamma + doc_counts[n]*(phi[n, :] - oldphi)
                digam = digamma(gamma)


        # likelihood = slda_compute_likelihood(doc, phi, var_gamma)
        # assert(!isnan(likelihood))
        # converged = fabs((likelihood_old - likelihood) / likelihood_old)
        # likelihood_old = likelihood
        return 0.

    def softmax_f(self, x):
        PENALTY = 1.

        f = 0.
        f_reg = 0.

        f_reg -= np.sum(np.power(self.eta[:self.K - 1, self.K], 2.) * PENALTY/2.0)

        f = 0.0  # log likelihood
        f += np.sum(self.eta[self.ss["doc_labels"], :] * self.ss["z_bar_m"])

        t = 0.0  # in log space,  1+exp()+exp()...
        for l in range(self.K - 1):
            a1 = 0.0  # \eta_k^T * \bar{\phi}_d
            a2 = 0.0  # 1 + 0.5 * \eta_k^T * Var(z_bar)\eta_k
            for k in range(self.K):
                a1 += self.eta[l, k] * self.ss["z_bar_m"][k].sum(axis=0)
                for j in range(self.K):
                    idx = util.map_idx(k, j, self.K)
                    a2 += self.eta[l, k] * self.ss["z_bar_var"][idx].sum(axis=0) * self.eta[l, j]
            a2 = 1.0 + 0.5 * a2
            t = util.log_sum(t, a1 + np.log(a2))
        f -= t

        return -(f + f_reg)

if __name__ == "__main__":
    TESTING_SUBSIZE = 50
    num_classes = 5
    learning_rate = 0.01
    num_steps = 1000
    logging.info('Generating data')

    fname = "./data/winemag_dataset_0.csv"

    data, docs_words, docs_counts, K, V, min_acceptable_words = util.get_all_supervised_requirements(
        fname,
        max_classes=5,
        max_annotations=100
    )

    docs_labels = data["class"].astype(np.int64).values
    lda = MultiClassSupervisedLda(num_classes, K, V)
    lda.v_em(docs_words, docs_counts, docs_labels)
