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
    def __init__(self, num_docs, num_words_per_doc_vec, num_topics, num_vocabs, subsample_size):
        # hyperparameters for the priors

        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)

        self.num_docs = num_docs
        self.num_words_per_doc_vec = num_words_per_doc_vec
        self.num_topics = num_topics
        self.num_vocabs = num_vocabs
        self.subsample_size = subsample_size

    @property
    def loss(self):
        return TraceEnum_ELBO(max_plate_nesting=2)

    def model(self, data=None):
        # Globals.
        topic_weights = torch.ones(self.num_topics) / self.num_topics
        vocab_weights = torch.ones(self.num_vocabs) / self.num_vocabs
        with pyro.plate("topics", self.num_topics):
            topic_x_words = pyro.sample("topic_words", dist.Dirichlet(vocab_weights))

        doc_words = []
        # Locals
        for i in pyro.plate("documents", self.num_docs, subsample_size=self.subsample_size):
            doc = data[i]
            doc_x_topics = pyro.sample("doc_topics_{}".format(i), dist.Dirichlet(topic_weights))
            with pyro.plate("words_{}".format(i), self.num_words_per_doc_vec[i]):
                word_x_topics = pyro.sample("word_topics_{}".format(i), dist.Categorical(doc_x_topics)
                                            , infer={"enumerate": "parallel"})
                words = pyro.sample("doc_words_{}".format(i), dist.Categorical(topic_x_words[word_x_topics]),
                                       obs=doc)
                doc_words.append(words)

        return doc_words, topic_x_words

    # We will use amortized inference of the local topic variables, achieved by a
    # multi-layer perceptron. We'll wrap the guide in an nn.Module.

    def make_predictor(self, args):
        layer_sizes = ([self.num_vocabs] +
                       [int(s) for s in args.layer_sizes.split('-')] +
                       [self.num_topics])
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
        topic_words_posterior = pyro.param(
                "topic_words_posterior",
                lambda: torch.ones(self.num_vocabs),
                constraint=constraints.positive)
        with pyro.plate("topics", self.num_topics):
            topic_words = pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

        pyro.module("predictor", predictor)
        count_words = 0
        for i in pyro.plate("documents", self.num_docs, subsample_size=self.subsample_size):
            words = list(set([int(word)for word in data[i]]))
            # The neural network will operate on histograms rather than word
            # index vectors, so we'll convert the raw data to a histogram.
            counts = np.zeros((1, self.num_vocabs))
            for j in range(len(words)):
                counts[0, words[j]] = vocab_count[count_words + j]
            counts = torch.tensor(counts, dtype=torch.float)
            doc_topics = predictor(counts)
            doc_x_topics = pyro.sample("doc_topics_{}".format(i), dist.Delta(doc_topics, event_dim=1))
            count_words += len(words)
            with pyro.plate("words_{}".format(i), self.num_words_per_doc_vec[i]):
                word_x_topics = pyro.sample("word_topics_{}".format(i), dist.Categorical(doc_x_topics)
                                            , infer={"enumerate": "parallel"})

        return topic_words
