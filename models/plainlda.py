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
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro.optim import ClippedAdam

class plainLDA:

    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample):
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
        return Trace_ELBO(max_plate_nesting=2)
    #return TraceMeanField_ELBO(max_plate_nesting=2)
    #return Trace_ELBO(max_plate_nesting=2)


    def model(self, doc_list=None):
        """pyro model for lda"""
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V) / self.V

        # returns topic x vocab matrix
        with pyro.plate("topics", self.K):
            # beta => per topic word vec
            Beta = pyro.sample(f"beta", dist.Dirichlet(eta))

        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        X, Theta = [], []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if doc_list is None else doc_list[d]

            with pyro.plate(f"words_{d}", self.N[d]):

                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}",
                                           dist.Categorical(theta))#,
                #infer={"enumerate": "parallel"})
                # from that topic vec, select a word
                w = pyro.sample(f"w_{d}", dist.Categorical(Beta[z_assignment]), obs=doc)

            X.append(w)
            Theta.append(theta)

        Theta = torch.stack(Theta)

        return X, Beta, Theta

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        with pyro.plate("topics", self.K) as k_vec:
            # lambda => q for the per-topic word distribution
            lamda = torch.stack([pyro.param(f"lamda_q_{k}", 
            (1 + 0.01*(2*torch.rand(self.V)-1)), 
            constraint=constraints.positive) for k in k_vec])

            # beta_q => posterior per topic word vec
            Beta_q = pyro.sample(f"beta", dist.Dirichlet(lamda))

        Theta_q = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # gamma => q for the per-doc topic vector
            gamma = pyro.param(f"gamma_q_{d}",
                               (1+0.01*(2*torch.rand(self.K)-1))/self.K, 
                               constraint=constraints.positive)
    
            # theta_q => posterior per-doc topic vector
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma))

            with pyro.plate(f"words_{d}", self.N[d]) as w_vec:

                phi = torch.stack([pyro.param(f"phi_q_{d}_{w}",
                (1+0.01*(2*torch.rand(self.K)-1))/self.K, 
                constraint=constraints.positive) for w in w_vec])

                pyro.sample(f"z_assignment_{d}", dist.Categorical(phi))

            Theta_q.append(theta_q)

        Theta_q = torch.stack(Theta_q)

        return Beta_q, Theta_q












    # def model(self, doc_list=None):
        #     """pyro model for lda"""
        #     # eta => prior for the per-topic word distributions
        #     eta = torch.ones(self.V) / self.V

    #     # returns topic x vocab matrix
    #     # with pyro.plate("topics", self.K):
        #     #     # beta => per topic word vec
        #     #     beta = pyro.sample(f"beta", dist.Dirichlet(eta))
        #     Beta = torch.zeros((self.K, self.V))
        #     for k in pyro.plate("topics", self.K):
            #         # beta => per topic word vec
            #         Beta[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))

    #     # alpha => prior for the per-doc topic vector
    #     alpha = torch.ones(self.K) / self.K

    #     X, Theta = [], []
    #     for d, y in pyro.markov(enumerate(doc_list, subsample_size=self.S)):

    #         # theta => per-doc topic vector
    #         theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

    #         doc = None if doc_list is None else doc_list[d]
    #         with pyro.plate(f"words_{d}", self.N[d]):

    #             # assign a topic
    #             z_assignment = pyro.sample(f"z_assignment_{d}",
    #                                         dist.Categorical(theta),
    #                                         infer={"enumerate": "parallel"})
    #             # from that topic vec, select a word
    #             w = pyro.sample(f"w_{d}", dist.Categorical(Beta[z_assignment]), obs=doc)

    #         X.append(w)
    #         Theta.append(theta)

    #     Theta = torch.stack(Theta)

    #     return X, Beta, Theta

    # def model(self, doc_list=None):
        #     """pyro model for lda"""
        #     # eta => prior for the per-topic word distributions
        #     eta = torch.ones(self.V) / self.V

    #     # returns topic x vocab matrix
    #     # with pyro.plate("topics", self.K):
        #     #     # beta => per topic word vec
        #     #     beta = pyro.sample(f"beta", dist.Dirichlet(eta))
        #     Beta = torch.zeros((self.K, self.V))
        #     for k in pyro.plate("topics", self.K):
            #         # beta => per topic word vec
            #         Beta[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))

    #     # alpha => prior for the per-doc topic vector
    #     alpha = torch.ones(self.K) / self.K

    #     X_list, Theta = [], []
    #     for d in pyro.plate("documents", self.D, subsample_size=self.S):

    #         # theta => per-doc topic vector
    #         theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

    #         doc = None if doc_list is None else doc_list[d]

    #         X = torch.zeros(self.N[d])
    #         for t, w in pyro.markov(enumerate(doc)):
        #         #for t in pyro.markov(range(self.N[d])):

    #             # assign a topic
    #             z_assignment = pyro.sample(f"z_assignment_{d}_{t}",
    #                                         dist.Categorical(theta),
    #                                         infer={"enumerate": "parallel"})
    #             # from that topic vec, select a word
    #             X[t] = pyro.sample(f"w_{d}_{t}", dist.Categorical(Beta[z_assignment]), obs=w)

    #         X_list.append(X)
    #         Theta.append(theta)

    #     Theta = torch.stack(Theta)

    #     return X_list, Beta, Theta


# LATEST WORKING ### START 
    def model(self, doc_list=None):
        """pyro model for lda"""
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V) / self.V

        # returns topic x vocab matrix
        # with pyro.plate("topics", self.K):
            #     # beta => per topic word vec
            #     beta = pyro.sample(f"beta", dist.Dirichlet(eta))
        Beta = torch.zeros((self.K, self.V))
        for k in pyro.plate("topics", self.K):
            # beta => per topic word vec
            Beta[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))

        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        X, Theta = [], []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if doc_list is None else doc_list[d]

            words = torch.zeros(self.N[d])
            for w in pyro.plate(f"words_{d}", self.N[d]):

                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}_{w}",
                                           dist.Categorical(theta))#,
                #infer={"enumerate": "parallel"})
                # from that topic vec, select a word
                words[w] = pyro.sample(f"w_{d}_{w}", dist.Categorical(Beta[z_assignment]), obs=doc[w])

            X.append(words)
            Theta.append(theta)

        Theta = torch.stack(Theta)

        return X, Beta, Theta

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        Beta_q = torch.zeros((self.K, self.V))
        for k in pyro.plate("topics", self.K):
            # lambda => q for the per-topic word distribution
            lamda = pyro.param(f"lamda_q_{k}", (1 + 0.01*(2*torch.rand(self.V)-1)), constraint=constraints.positive)# #torch.ones(self.V) / self.K, constraint=constraints.positive)#/ self.V, torch.rand(self.V),
            # beta_q => posterior per topic word vec
            Beta_q[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(lamda))

        Theta_q = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # gamma => q for the per-doc topic vector
            gamma = pyro.param(f"gamma_q_{d}",
                               (1+0.01*(2*torch.rand(self.K)-1))/self.K, constraint=constraints.positive)
            #torch.ones(self.K) / self.K, constraint=constraints.positive)#/ / self.K, torch.rand(self.K),

            # theta_q => posterior per-doc topic vector
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma))

            for w in pyro.plate(f"words_{d}", self.N[d]):

                phi = pyro.param(f"phi_q_{d}_{w}",
                                 (1+0.01*(2*torch.rand(self.K)-1))/self.K, constraint=constraints.positive)
                pyro.sample(f"z_assignment_{d}_{w}", dist.Categorical(phi))

            Theta_q.append(theta_q)

        Theta_q = torch.stack(Theta_q)

        return Beta_q, Theta_q

# LATEST WORKING ### END

    # def guide(self, doc_list=None):
        #     """pyro guide for lda inference"""

    #     Beta_q = torch.zeros((self.K, self.V))
    #     for k in pyro.plate("topics", self.K):
        #         # lambda => q for the per-topic word distribution
        #         lamda = pyro.param(f"lamda_q_{k}", (1 + 0.01*(2*torch.rand(self.V)-1)), constraint=constraints.positive)# #torch.ones(self.V) / self.K, constraint=constraints.positive)#/ self.V, torch.rand(self.V),
        #         # beta_q => posterior per topic word vec
        #         Beta_q[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(lamda))

    #     Theta_q = []
    #     for d in pyro.plate("documents", self.D, subsample_size=self.S):

    #         # gamma => q for the per-doc topic vector
    #         gamma = pyro.param(f"gamma_q_{d}", torch.ones(self.K) / self.K, constraint=constraints.positive)#/ / self.K, torch.rand(self.K),

    #         # theta_q => posterior per-doc topic vector
    #         theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma).to_event())

    #         phi = pyro.param(f"phi_q_{d}", torch.ones(self.K) / self.K, constraint=constraints.positive)#/ / self.K, torch.rand(self.K),

    #         with pyro.plate(f"words_{d}", self.N[d]):
        #             # assign a topic
        #             pyro.sample(f"z_assignment_{d}", dist.Categorical(phi))

    #         Theta_q.append(theta_q)

    #     Theta_q = torch.stack(Theta_q)

    #     return Beta_q, Theta_q


