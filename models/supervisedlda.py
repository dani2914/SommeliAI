
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
from pyro.infer.autoguide import AutoDiagonalNormal

import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


class supervisedLDA():
    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample):
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)

        self.D = num_docs
        self.N = num_words_per_doc
        self.K = num_topics
        self.V = num_vocabs
        self.S = num_subsample

    @property
    def loss(self):
        return Trace_ELBO(max_plate_nesting=2)

    def model(self, data=None, label=None):
        """pyro model for lda"""
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V) / self.V

        # returns t x w matrix
        with pyro.plate("topics", self.K):
            beta = pyro.sample("beta", dist.Dirichlet(eta))

        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        # eta => prior for regression coefficient
        weights_loc = torch.randn(self.K)
        weights_scale = torch.eye(self.K)
        eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        # sigma => prior for regression variance
        sigma_loc = torch.tensor(1.)
        sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

        # returns d x t matrix
        Theta = []
        y_list = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if data is None else data[d]
            z_bar = torch.zeros(self.K)

            with pyro.plate(f"words_{d}", self.N[d]):
                z_assignment = pyro.sample(f"z_assignment_{d}",
                                           dist.Categorical(theta),
                                           infer={"enumerate": "parallel"})

                w = pyro.sample(f"w_{d}", dist.Categorical(beta[z_assignment]), obs=doc)

            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= self.N[d]

            mean = torch.dot(eta, z_bar)
            y_label = pyro.sample(f"doc_label_{d}", dist.Normal(mean, sigma), obs=torch.tensor([label[d]]))
        y_list.append(y_label)

        return Theta, beta, y_list

    def guide(self, data=None, label=None):

        with pyro.plate("topics", self.K):
            # eta_q => q for the per-topic word distribution
            eta_q = pyro.param("eta_q", torch.rand(self.V),
                               constraint=constraints.positive)
            beta = pyro.sample("beta", dist.Dirichlet(eta_q))

        # eta => prior for regression coefficient
        weights_loc = pyro.param('weights_loc', torch.randn(self.K))
        weights_scale = pyro.param('weights_scale', torch.eye(self.K),
                                   constraint=constraints.positive)
        eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        # sigma => prior for regression variance
        sigma_loc = pyro.param('bias', torch.tensor(1.), constraint=constraints.positive)
        sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            alpha_q = pyro.param(f"alpha_q_{d}", torch.ones(self.K),
                                 constraint=constraints.positive)
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha_q))
            z_bar = torch.zeros(self.K)
            with pyro.plate(f"words_{d}", self.N[d]):
                z_assignment = pyro.sample(f"z_assignment_{d}", dist.Categorical(theta_q))

            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= self.N[d]

            mean = torch.dot(eta, z_bar)

        return beta, mean, sigma
