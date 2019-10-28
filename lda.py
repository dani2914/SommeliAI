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

    def __init__(self):
        # hyperparameters for the priors

        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)


    def model(self, doc_list=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
        """pyro model for lda"""

        # beta => prior for the per-topic word distributions
        beta_0 = torch.ones(num_vocabs) / num_vocabs

        # returns t x w matrix
        with pyro.plate("topics", num_topics):
            topics_x_words = pyro.sample("topics_x_words", dist.Dirichlet(beta_0))

            # alpha => prior for the per-document topic distribution
            alpha_0 = torch.ones(num_topics) / num_topics

        # returns d x t matrix 
        doc_x_words = []
        for i in pyro.plate("documents", num_docs):

            docs_x_topics = pyro.sample("docs_x_topics_{}".format(i),
                                        dist.Dirichlet(alpha_0))

            data = None if doc_list is None else doc_list[i]

            with pyro.plate("words_{}".format(i), num_words_per_doc_vec[i]):
                word_x_topics = pyro.sample("word_x_topics_{}".format(i),
                                            dist.Categorical(docs_x_topics),
                                            infer={"enumerate": "parallel"})

                words = pyro.sample("docs_x_words_{}".format(i),
                                    dist.Categorical(topics_x_words[word_x_topics]),
                                    obs=data)

            doc_x_words.append(words)

        return doc_x_words, topics_x_words


    def guide(self, doc_list=None, num_docs=None, num_words_per_doc_vec=None,
                           num_topics=None, num_vocabs=None):
        """pyro guide for lda inference"""

        # beta_q => q for the per-topic word distribution
        beta_q = pyro.param("beta_q", torch.ones(num_vocabs), constraint=constraints.positive)

        with pyro.plate("topics", num_topics):
            pyro.sample("topics_x_words", dist.Dirichlet(beta_q))

            # alpha_q => q for the per-document topic distribution
            alpha_q = pyro.param("alpha_q", torch.ones(num_topics), constraint=constraints.positive)

        for i in pyro.plate("documents", num_docs):
            pyro.sample("docs_x_topics_{}".format(i), dist.Dirichlet(alpha_q))
