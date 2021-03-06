import pyro.distributions as dist
import torch
from torch.distributions import constraints

import pyro
from pyro.infer import TraceMeanField_ELBO


class plainLDA:

    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample):
        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(False)

        self.D = num_docs
        self.N = num_words_per_doc
        self.K = num_topics
        self.V = num_vocabs
        self.S = num_subsample

    @property
    def loss(self):
        return TraceMeanField_ELBO(max_plate_nesting=2)

    def model(self, doc_list=None):
        """pyro model for lda"""
        # eta => prior for the per-topic word distributions
        eta = torch.ones(self.V) / self.V

        # returns topic x vocab matrix
        with pyro.plate("topics", self.K):
            # beta => per topic word vec
            beta = pyro.sample("beta", dist.Dirichlet(eta))

        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        X, Theta = [], []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if doc_list is None else doc_list[d]
            with pyro.plate(f"words_{d}", self.N[d]):

                # assign a topic
                z_assignment = pyro.sample(
                    f"z_assignment_{d}",
                    dist.Categorical(theta),
                    infer={"enumerate": "parallel"}
                )
                # from that topic vec, select a word
                w = pyro.sample(
                    f"w_{d}",
                    dist.Categorical(beta[z_assignment]),
                    obs=doc
                )

            X.append(w)
            Theta.append(theta)
        return X, beta, Theta

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        with pyro.plate("topics", self.K):
            # eta_q => q for the per-topic word distribution
            eta_q = pyro.param("eta_q", torch.rand(self.V),
                               constraint=constraints.positive)
            # beta_q => posterior per topic word vec
            beta_q = pyro.sample("beta", dist.Dirichlet(eta_q))

        Theta_q = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # alpha_q => q for the per-doc topic vector
            alpha_q = pyro.param(f"alpha_q_{d}", torch.rand(self.K),
                                 constraint=constraints.positive)

            # theta_q => posterior per-doc topic vector
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha_q))

            with pyro.plate(f"words_{d}", self.N[d]):
                # assign a topic
                pyro.sample(f"z_assignment_{d}", dist.Categorical(theta_q))

            Theta_q.append(theta_q)

        return beta_q, Theta_q
