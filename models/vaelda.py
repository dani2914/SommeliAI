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

    def model(self, doc_list=None):
        """pyro model for lda"""
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V) / self.V

        with pyro.plate("topics", self.K):
            alpha_q = pyro.sample(f"alpha", dist.Gamma(1. / self.K, 1.))
            # beta => per topic word vec
            Beta = pyro.sample(f"beta", dist.Dirichlet(eta))

        X, Theta = [], []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha_q))

            doc = None if doc_list is None else doc_list[d]
            for t, y in pyro.markov(enumerate(doc_list[d])):
                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}_{t}",
                                           dist.Categorical(theta),
                                           infer={"enumerate": "parallel"})
                # from that topic vec, select a word
                w = pyro.sample(f"w_{d}_{t}", dist.Categorical(Beta[z_assignment]), obs=doc)

            X.append(w)
            Theta.append(theta)

        return X, Beta, Theta

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
        """pyro guide for lda inference"""
        Alpha = []
        topic_weights_posterior = pyro.param(
            "topic_weights_posterior",
            lambda: torch.ones(self.K),
            constraint=constraints.positive)
        topic_words_posterior = pyro.param(
            "topic_words_posterior",
            lambda: torch.ones((self.K, self.V)),
            constraint=constraints.greater_than(0.5))
        Beta_q = torch.zeros((self.K, self.V))
        with pyro.plate("topics", self.K):
            # eta_q => q for the per-topic word distribution
            alpha_q = pyro.sample("alpha", dist.Gamma(topic_weights_posterior, 1.))
            Beta_q = pyro.sample("beta", dist.Dirichlet(topic_words_posterior))

        pyro.module("predictor", predictor)
        count_words = 0

        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            # The neural network will operate on histograms rather than word
            # index vectors, so we'll convert the raw data to a histogram.
            counts = [[0 for i in range(self.V)]]
            for j in data[d]:
                counts[0][int(j)] += 1
            counts = torch.tensor(counts, dtype=torch.float)
            doc_topics = predictor(counts)
            alpha = pyro.sample(f"theta_{d}", dist.Delta(doc_topics, event_dim=1))
            Alpha.append(alpha)

        return Beta_q, Alpha
