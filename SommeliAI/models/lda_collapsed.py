import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import TraceEnum_ELBO


class collapsedLDA:

    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsamples):
        # pyro settings
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(False)

        self.D = num_docs
        self.N = num_words_per_doc
        self.K = num_topics
        self.V = num_vocabs
        self.S = num_subsamples

    @property
    def loss(self):
        return TraceEnum_ELBO(max_plate_nesting=1)

    def model(self, doc_list=None):
        """pyro model for lda"""

        # eta => prior for the per-topic word distribution
        eta = torch.ones(self.V)

        with pyro.plate("topics", self.K):

            # Beta => per topic word distribution
            Beta = pyro.sample(f"beta", dist.Dirichlet(eta))

        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        X_List, Theta = [], []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            X = torch.zeros(self.N[d])
            for t in pyro.markov(range(self.N[d])):

                doc = None if doc_list is None else doc_list[d][t]

                # assign a topic
                z_assignment = pyro.sample(
                    f"z_assignment_{d}_{t}",
                    dist.Categorical(theta),
                    infer={"enumerate": "parallel"}
                )

                # from that topic vec, select a word
                X[t] = pyro.sample(
                    f"w_{d}_{t}",
                    dist.Categorical(Beta[z_assignment]),
                    obs=doc
                )

            X_List.append(X)
            Theta.append(theta)

        Theta = torch.stack(Theta)

        return X_List, Beta, Theta

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        with pyro.plate("topics", self.K) as k_vec:

            # Lambda => latent variable for the per-topic word q distribution
            Lamda = torch.stack([
                pyro.param(
                    f"lamda_q_{k}",
                    (1 + 0.01*(2*torch.rand(self.V)-1)),
                    constraint=constraints.positive)
                for k in k_vec
            ])

            # Beta_q => per-topic word q distribtion
            Beta_q = pyro.sample(f"beta", dist.Dirichlet(Lamda))

        Theta_q = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # gamma => q for the per-doc topic vector
            gamma = pyro.param(f"gamma_q_{d}",
                               (1+0.01*(2*torch.rand(self.K)-1))/self.K,
                               constraint=constraints.positive)

            # theta_q => per-doc topic q distribution
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma))

            Theta_q.append(theta_q)

        Theta_q = torch.stack(Theta_q)

        return Beta_q, Theta_q
