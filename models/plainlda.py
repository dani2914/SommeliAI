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
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam

class plainLDA:

    def __init__(self, num_docs, num_words_per_doc_vec,
                 num_topics, num_vocabs, num_subsample):
        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(False)

        self.num_docs = num_docs
        self.num_words_per_doc_vec = num_words_per_doc_vec
        self.num_topics = num_topics
        self.num_vocabs = num_vocabs
        self.num_subsample = num_subsample

    @property
    def loss(self):
        return TraceMeanField_ELBO(max_plate_nesting=2)

    def model(self, doc_list=None):
        """pyro model for lda"""
        # beta => prior for the per-topic word distributions
        beta_0 = torch.ones(self.num_vocabs) / self.num_vocabs

        # returns t x w matrix
        with pyro.plate("topics", self.num_topics):
            phi = pyro.sample("phi", dist.Dirichlet(beta_0))

            # alpha => prior for the per-document topic distribution
            alpha_0 = torch.ones(self.num_topics) / self.num_topics

        # returns d x t matrix
        Theta = []
        for i in pyro.plate("documents", self.num_docs, subsample_size=self.num_subsample):

            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(alpha_0))

            data = None if doc_list is None else doc_list[i]

            with pyro.plate(f"words_{i}", self.num_words_per_doc_vec[i]):
                z_assignment = pyro.sample(f"z_assignment_{i}",
                                            dist.Categorical(theta),
                                            infer={"enumerate": "parallel"})

                w = pyro.sample(f"w_{i}", dist.Categorical(phi[z_assignment]), obs=data)

            Theta.append(w)

        return Theta, phi

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        # beta_q => q for the per-topic word distribution
        beta_q = pyro.param("beta_q", torch.ones(self.num_vocabs),
                            constraint=constraints.positive)

        with pyro.plate("topics", self.num_topics):
            topic_words = pyro.sample("phi", dist.Dirichlet(beta_q))

            # alpha_q => q for the per-document topic distribution
            alpha_q = pyro.param("alpha_q", torch.ones(self.num_topics),
                                 constraint=constraints.positive)

        for i in pyro.plate("documents", self.num_docs,
                            subsample_size=self.num_subsample):
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(alpha_q))

            with pyro.plate(f"words_{i}", self.num_words_per_doc_vec[i]):
                z_assignment = pyro.sample(f"z_assignment_{i}",
                                            dist.Categorical(theta))

        return topic_words
