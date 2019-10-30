""" LDA Class """
import util

import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import TraceEnum_ELBO

class origLDA:

    def __init__(self, num_docs, num_words_per_doc_vec, num_topics, num_vocabs):
        # hyperparameters for the priors

        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)

        self.num_docs = num_docs
        self.num_words_per_doc_vec = num_words_per_doc_vec
        self.num_topics = num_topics
        self.num_vocabs = num_vocabs

    @property
    def loss(self):
        return TraceEnum_ELBO(max_plate_nesting=2)

    def model(self, doc_list=None):
        """pyro model for lda"""

        # beta => prior for the per-topic word distributions
        beta_0 = torch.ones(self.num_vocabs) / self.num_vocabs

        # returns t x w matrix
        with pyro.plate("topics", self.num_topics):
            topics_x_words = pyro.sample("topics_x_words", dist.Dirichlet(beta_0))

            # alpha => prior for the per-document topic distribution
            alpha_0 = torch.ones(self.num_topics) / self.num_topics

        # returns d x t matrix 
        doc_x_words = []
        for i in pyro.plate("documents", self.num_docs):

            docs_x_topics = pyro.sample(f"docs_x_topics_{i}",
                                        dist.Dirichlet(alpha_0))

            data = None if doc_list is None else doc_list[i]

            with pyro.plate(f"words_{i}", self.num_words_per_doc_vec[i]):
                word_x_topics = pyro.sample(f"word_x_topics_{i}",
                                            dist.Categorical(docs_x_topics),
                                            infer={"enumerate": "sequential"})

                words = pyro.sample(f"docs_x_words_{i}",
                                    dist.Categorical(topics_x_words[word_x_topics]),
                                    obs=data)

            doc_x_words.append(words)

        return doc_x_words, topics_x_words


    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        # beta_q => q for the per-topic word distribution
        beta_q = pyro.param("beta_q", torch.ones(self.num_vocabs), constraint=constraints.positive)

        with pyro.plate("topics", self.num_topics):
            pyro.sample("topics_x_words", dist.Dirichlet(beta_q))

            # alpha_q => q for the per-document topic distribution
            alpha_q = pyro.param("alpha_q", torch.ones(self.num_topics), constraint=constraints.positive)

        for i in pyro.plate("documents", self.num_docs):
            pyro.sample(f"docs_x_topics_{i}", dist.Dirichlet(alpha_q))
