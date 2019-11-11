
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
    def __init__(self, num_docs, num_words_per_doc_vec, num_topics, num_vocabs, subsample_size):
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)

        self.num_docs = num_docs
        self.num_words_per_doc_vec = num_words_per_doc_vec
        self.num_topics = num_topics
        self.num_vocabs = num_vocabs
        self.num_subsample = subsample_size

    @property
    def loss(self):
        return Trace_ELBO(max_plate_nesting=2)

    def model(self, data=None, label=None):
        """pyro model for lda"""
        # beta => prior for the per-topic word distributions
        beta_0 = torch.ones(self.num_vocabs) / self.num_vocabs

        # returns t x w matrix
        with pyro.plate("topics", self.num_topics):
            phi = pyro.sample("phi", dist.Dirichlet(beta_0))

        # alpha => prior for the per-document topic distribution
        alpha_0 = torch.ones(self.num_topics) / self.num_topics

        # eta => prior for regression coefficient
        weights_loc = torch.randn(self.num_topics)
        weights_scale = torch.eye(self.num_topics)
        eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        # sigma => prior for regression variance
        sigma_loc = torch.tensor(1.)
        sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

        # returns d x t matrix
        Theta = []
        y_list = []
        for i in pyro.plate("documents", self.num_docs, subsample_size=self.num_subsample):
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(alpha_0))

            doc = None if data is None else data[i]
            z_bar = torch.zeros(self.num_topics)

            with pyro.plate(f"words_{i}", self.num_words_per_doc_vec[i]):
                z_assignment = pyro.sample(f"z_assignment_{i}",
                                           dist.Categorical(theta),
                                           infer={"enumerate": "parallel"})

                w = pyro.sample(f"w_{i}", dist.Categorical(phi[z_assignment]), obs=doc)

            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= z_bar.shape[0]

            mean = torch.dot(eta, z_bar)
            y_label = pyro.sample(f"doc_label_{i}", dist.Normal(mean, sigma), obs=torch.tensor([label[i]]))
        y_list.append(y_label)

        return Theta, phi, y_list

    def guide(self, data=None, label=None):

        with pyro.plate("topics", self.num_topics):
            # beta_q => q for the per-topic word distribution
            beta_q = pyro.param("beta_q", torch.randn(self.num_vocabs), constraint=constraints.positive)
            phi = pyro.sample("phi", dist.Dirichlet(beta_q))

        # eta => prior for regression coefficient
        weights_loc = pyro.param('weights_loc', torch.randn(self.num_topics))
        weights_scale = pyro.param('weights_scale', torch.randn(self.num_topics, self.num_topics),
                                   constraint=constraints.positive)
        eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        # sigma => prior for regression variance
        sigma_loc = pyro.param('bias', torch.randn(1.), constraint=constraints.positive)
        sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

        for i in pyro.plate("documents", self.num_docs, subsample_size=self.num_subsample):
            alpha_q = pyro.param(f"alpha_q_{i}", torch.randn(self.num_topics),
                                 constraint=constraints.positive)
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(alpha_q))
            z_bar = torch.zeros(self.num_topics)
            with pyro.plate("words_{}".format(i), self.num_words_per_doc_vec[i]):
                z_assignment = pyro.sample("z_assignment_{}".format(i), dist.Categorical(theta))

            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= z_bar.shape[0]

            mean = torch.dot(eta, z_bar)

        return phi
