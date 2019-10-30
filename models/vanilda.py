""" Vanilla LDA """

import util

import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

class vaniLDA:
    def __init__(self):
        # hyperparameters for the priors

        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)

    @property
    def loss(self):
        return Trace_ELBO(max_plate_nesting=2)

    def model(self, doc_list=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
        """pyro model for lda"""

        # beta => prior for the per-topic word distributions
        beta_0 = torch.ones(num_vocabs) / num_vocabs

        # topic distributions stacked into K x V matrix
        with pyro.plate("topics", num_topics):
            phi = pyro.sample("phi", dist.Dirichlet(beta_0))
            assert phi.shape == (num_topics, num_vocabs)

        # alpha => prior for the per-document topic distribution
        alpha_0 = torch.ones(num_topics) / num_topics
        Theta = [] # topic distributions stacked into D x K matrix
        for i in pyro.plate("documents", num_docs):
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(alpha_0))
    
            data = None if doc_list is None else doc_list[i]
            with pyro.plate(f"words_{i}", num_words_per_doc_vec[i]):
                z_assignment = pyro.sample(f"z_{i}", dist.Categorical(theta), infer={"enumerate": "parallel"})
                w = pyro.sample(f"w_{i}", dist.Categorical(phi[z_assignment]), obs=data)
                Theta.append(w)
        return Theta, phi


    def guide(self, doc_list=None, num_docs=None, num_words_per_doc_vec=None,
                           num_topics=None, num_vocabs=None):
        """pyro guide for lda inference"""        
        # topic distributions stacked into K x V matrix
        with pyro.plate("topics", num_topics):
            lambda_q = pyro.param("lambda_q", torch.ones(num_vocabs), constraint=constraints.positive)
            phi = pyro.sample("phi", dist.Dirichlet(lambda_q))

        for i in pyro.plate("documents", num_docs):
            # alpha_q => q for the per-document topic distribution
            gamma_q = pyro.param("gamma_q", torch.ones(num_topics), constraint=constraints.positive)
            theta = pyro.sample(f"theta_{i}", dist.Dirichlet(gamma_q))

            with pyro.plate(f"words_{i}", num_words_per_doc_vec[i]):
                pi = pyro.param("pi", torch.randn(num_topics).exp(), constraint=constraints.simplex)
                z = pyro.sample(f"z_{i}", dist.Categorical(pi))
