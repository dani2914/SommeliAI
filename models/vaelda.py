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


class vaeLDA:
    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample):
        # hyperparameters for the priors

        # pyro settings
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
        return TraceEnum_ELBO(max_plate_nesting=2)

    def model(self, data=None):
        # Globals.
        Beta = []
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V)

        # returns t x w matrix
        for k in pyro.plate("topics", self.D):
            beta = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))
            Beta.append(beta.cpu().data.numpy())

        Beta = torch.tensor(Beta)
        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K)

        # returns d x t matrix
        Theta = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))
            doc = None if data is None else data[d]

            with pyro.plate(f"words_{d}", self.N[d]):
                z_assignment = pyro.sample(f"z_assignment_{d}",
                                           dist.Categorical(theta),
                                           infer={"enumerate": "parallel"})

                w = pyro.sample(f"w_{d}", dist.Categorical(Beta[z_assignment]), obs=doc)

        return Theta, Beta

    def make_predictor(self, args):
        layer_sizes = ([self.V] +
                       [int(s) for s in args.layer_sizes.split('-')] +
                       [self.K])
        logging.info('Creating MLP with sizes {}'.format(layer_sizes))
        layers = []
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layer = nn.Linear(in_size, out_size)
            layer.weight.data.normal_(0, 0.001)
            layer.bias.data.normal_(0, 0.001)
            layers.append(layer)
            layers.append(nn.Sigmoid())
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    def parametrized_guide(self, predictor, vocab_count, data=None):
        Beta = []
        for k in pyro.plate("topics", self.D, subsample_size=self.S):
            # eta_q => q for the per-topic word distribution
            eta_q = pyro.param(f"eta_q_{k}", torch.ones(self.V),
                               constraint=constraints.positive)
            beta = pyro.sample(f"beta_{k}", dist.Dirichlet(eta_q))
            Beta.append(beta.cpu().data.numpy())

        # eta => prior for regression coefficient
        # weights_loc = pyro.param('weights_loc', torch.randn(self.K))
        # weights_scale = pyro.param('weights_scale', torch.eye(self.K),
        #                            constraint=constraints.positive)
        # eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
        # sigma => prior for regression variance

        pyro.module("predictor", predictor)
        count_words = 0

        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            words = list(set([int(word) for word in data[d]]))
            # The neural network will operate on histograms rather than word
            # index vectors, so we'll convert the raw data to a histogram.
            counts = np.zeros((1, self.V))
            for j in range(len(words)):
                counts[0, words[j]] = vocab_count[count_words + j]
            counts = torch.tensor(counts, dtype=torch.float)
            doc_topics = predictor(counts)
            alpha = pyro.sample(f"alpha_{d}", dist.Delta(doc_topics, event_dim=1))

        return Beta
