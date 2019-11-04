
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

from pyro.infer import SVI, TraceGraph_ELBO
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
        return TraceGraph_ELBO(max_plate_nesting=2)

    def model(self, data=None, label=None):
        # Globals.
        with pyro.plate("topics", self.num_topics):
            topic_words = pyro.sample("beta",
                                      dist.Dirichlet(torch.ones(self.num_vocabs) / self.num_vocabs))

        # returns d x t matrix
        doc_x_words = []
        # Locals.
        with pyro.plate("labels", self.num_topics):
            label_topics = pyro.sample("theta",
                                      dist.Dirichlet(torch.ones(self.num_topics) / self.num_topics))

        for i in pyro.plate("documents", self.num_docs, subsample_size=self.num_subsample):
            words = data[i]
            doc_topics = label_topics[label[i]]
            with pyro.plate("words_{}".format(i), self.num_words_per_doc_vec[i]):
                word_topics = pyro.sample("z_{}".format(i), dist.Categorical(doc_topics),
                                              infer={"enumerate": "parallel"})
                words = pyro.sample("docs_x_words_{}".format(i), dist.Categorical(topic_words[word_topics]),
                                       obs=words)

            doc_x_words.append(words)

        return doc_x_words, topic_words

    def make_predictor(num_words, num_topics, args):
        layer_sizes = ([num_words] +
                       [int(s) for s in args.layer_sizes.split('-')] +
                       [num_topics])
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

    def guide(self, data=None, label=None):
        # beta_q => q for the per-topic word distribution
        beta_q = pyro.param("beta_q", torch.ones(self.num_vocabs), constraint=constraints.positive)
        eta_q = pyro.param("eta_q", torch.ones(self.num_topics), constraint=constraints.positive)

        with pyro.plate("topics", self.num_topics):
            pyro.sample("beta", dist.Dirichlet(beta_q))

        with pyro.plate("labels", self.num_topics):
            label_topics = pyro.sample("theta", dist.Dirichlet(eta_q))

        for i in pyro.plate("documents", self.num_docs, subsample_size=self.num_subsample):
            doc_topics = label_topics[label[i]]

            with pyro.plate("words_{}".format(i), self.num_words_per_doc_vec[i]):
                word_topics = pyro.sample("z_{}".format(i), dist.Categorical(doc_topics))

