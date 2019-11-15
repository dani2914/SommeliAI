
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

from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
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

    def model(self, data=None, label=None, record=False):
        """pyro model for lda"""

        Beta = []
        # eta => prior for the per-topic word distributions
        # eta = 1 + 0.01 * (2 * torch.rand(self.V) - 1)
        eta = torch.ones(self.V) / 5
        # returns t x w matrix
        for k in pyro.plate("topics", self.D, subsample_size=self.S):
            beta = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))
            Beta.append(beta.data.numpy())

        Beta = torch.tensor(Beta)
        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / 5

        # eta => prior for regression coefficient
        weights_loc = torch.randn(self.K)
        weights_scale = torch.eye(self.K)
        # eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        eta = torch.randn(self.K) * 2 - 1
        # sigma => prior for regression variance
        sigma_loc = torch.tensor(1.)
        sigma = torch.tensor(1.)
        # sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

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

                w = pyro.sample(f"w_{d}", dist.Categorical(Beta[z_assignment]), obs=doc)

            # for z in z_assignment:
            #     z_bar[z] += 1
            # z_bar /= self.N[d]
            # mean = torch.dot(eta, z_bar)
            # y_label = pyro.sample(f"doc_label_{d}", dist.Normal(mean, sigma), obs=torch.tensor([label[d]]))
            # y_list.append(y_label)

        return Theta, Beta, y_list

    def guide(self, data=None, label=None):

        Beta = []
        Theta = []
        docs = []
        for k in pyro.plate("topics", self.D, subsample_size=self.S):
            # eta_q => q for the per-topic word distribution
            eta_q = pyro.param(f"eta_q_{k}", torch.ones(self.V) / 5,
                               constraint=constraints.positive)
            beta = pyro.sample(f"beta_{k}", dist.Dirichlet(eta_q))
            Beta.append(beta.data.numpy())

        # eta => prior for regression coefficient
        weights_loc = pyro.param('weights_loc', torch.randn(self.K))
        weights_scale = pyro.param('weights_scale', torch.eye(self.K),
                                   constraint=constraints.positive)
        # eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        # # sigma => prior for regression variance
        #
        # # eta = pyro.param('coef', torch.randn(self.K) * 2 - 1)
        # sigma_loc = pyro.param('bias', torch.randn(1), constraint=constraints.positive)
        # sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
        # #sigma = pyro.param('bias', torch.tensor(1.), constraint=constraints.positive)
        eta = pyro.param('coef', torch.randn(self.K) * 2 - 1)
        sigma = pyro.param('bias', torch.tensor(1.), constraint=constraints.positive)

        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            alpha_q = pyro.param(f"alpha_q_{d}", torch.ones(self.K) / 5,
                                 constraint=constraints.positive)
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha_q))
            z_bar = torch.zeros(self.K)
            with pyro.plate(f"words_{d}", self.N[d]):
                z_assignment = pyro.sample(f"z_assignment_{d}", dist.Categorical(theta_q))

            # for z in z_assignment:
            #     z_bar[z] += 1
            # z_bar /= self.N[d]
            #
            # mean = torch.dot(eta, z_bar)
            # docs.append(label[d])
            # Theta.append(theta_q)

        return Beta, Theta, eta, sigma, docs
