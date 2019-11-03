""" Vanilla LDA """

import util

import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints
from pyro.infer import TraceEnum_ELBO

import pyro
import pyro.distributions as dist

class vaniLDA:
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

        # topic distributions stacked into K x V matrix
        with pyro.plate("topics", self.num_topics):
            phi = pyro.sample("phi", dist.Dirichlet(beta_0))
            assert phi.shape == (self.num_topics, self.num_vocabs)

        # alpha => prior for the per-document topic distribution
        alpha_0 = torch.ones(self.num_topics) / self.num_topics
        Theta = [] # topic distributions stacked into D x K matrix
        for i in pyro.plate("documents", self.num_docs):
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(alpha_0))
    
            data = None if doc_list is None else doc_list[i]
            with pyro.plate(f"words_{i}", self.num_words_per_doc_vec[i]):
                z_assignment = pyro.sample(f"z_{i}", dist.Categorical(theta), infer={"enumerate": "parallel"})
                w = pyro.sample(f"w_{i}", dist.Categorical(phi[z_assignment]), obs=data)
                Theta.append(w)
        return Theta, phi


    def guide(self, doc_list=None):
        """pyro guide for lda inference"""        
        # topic distributions stacked into K x V matrix
        with pyro.plate("topics", self.num_topics):
            lambda_q = pyro.param("lambda_q", torch.ones(self.num_vocabs), constraint=constraints.positive)
            phi = pyro.sample("phi", dist.Dirichlet(lambda_q))

        for i in pyro.plate("documents", self.num_docs):
            # alpha_q => q for the per-document topic distribution
            gamma_q = pyro.param("gamma_q", torch.ones(self.num_topics), constraint=constraints.positive)
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(gamma_q))

            with pyro.plate(f"words_{i}", self.num_words_per_doc_vec[i]):
                pi = pyro.param("pi", torch.randn(self.num_topics).exp(), constraint=constraints.simplex)
                z = pyro.sample(f"z_{i}", dist.Categorical(pi))
