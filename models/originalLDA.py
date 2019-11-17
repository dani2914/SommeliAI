import pyro.distributions as dist
from pyro.infer import TraceEnum_ELBO
import argparse
import functools
import numpy as np
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import util
import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import ClippedAdam

class originalLDA:

    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample, jitter):
        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(False)

        self.D = num_docs
        self.N = num_words_per_doc
        self.K = num_topics
        self.V = num_vocabs
        self.S = num_subsample
        self.jitter = jitter

    @property
    def loss(self):
        return TraceEnum_ELBO(max_plate_nesting=1)

    def model(self, doc_list=None):
        """pyro model for lda"""
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V) / self.V

        # returns topic x vocab matrix
        # with pyro.plate("topics", self.K):
        #     # beta => per topic word vec
        #     beta = pyro.sample(f"beta", dist.Dirichlet(eta))
        Beta = torch.zeros((self.K, self.V))
        for k in pyro.plate("topics", self.K):
            # beta => per topic word vec
            Beta[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))

        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        X, Theta = [], []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if doc_list is None else doc_list[d]
            for t, y in pyro.markov(enumerate(doc_list[d])):
                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}_{t}",
                                            dist.Categorical(theta),
                                            infer={"enumerate": "parallel"})
                # from that topic vec, select a word
                w = pyro.sample(f"w_{d}_{t}", dist.Categorical(Beta[z_assignment]), obs=y)

            X.append(w)
            Theta.append(theta)

        Theta = torch.stack(Theta)

        return X, Beta, Theta

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        Beta_q = torch.zeros((self.K, self.V))
        for k in pyro.plate("topics", self.K):
            # eta_q => q for the per-topic word distribution
            eta_q = pyro.param(f"eta_q_{k}", (1 + 0.01*(2*torch.rand(self.V)-1)), constraint=constraints.positive)# #torch.ones(self.V) / self.K, constraint=constraints.positive)#/ self.V, torch.rand(self.V),
            # beta_q => posterior per topic word vec
            # eta_q += self.jitter
            Beta_q[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(eta_q))

        Theta_q = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # alpha_q => q for the per-doc topic vector
            alpha_q = pyro.param(f"alpha_q_{d}", torch.ones(self.K) / self.K, constraint=constraints.positive)#/ / self.K, torch.rand(self.K),
            # alpha_q += self.jitter
            # theta_q => posterior per-doc topic vector
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha_q))
            # theta_q += self.jitter

            # with pyro.plate(f"words_{d}", self.N[d]):
            #     # assign a topic
            #     pyro.sample(f"z_assignment_{d}", dist.Categorical(theta_q))

            assert not any(np.isnan(alpha_q.detach().numpy()))
            assert not any(np.isnan(theta_q.detach().numpy()))

            Theta_q.append(theta_q)

        Theta_q = torch.stack(Theta_q)

        return Beta_q, Theta_q
