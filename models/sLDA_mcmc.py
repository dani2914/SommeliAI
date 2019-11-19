import logging

import torch

import data
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(__debug__)
pyro.set_rng_seed(0)

class sLDA_mcmc():
    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample):
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
        for d in pyro.plate("documents", self.D):

            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if doc_list is None else doc_list[d]
            for t, y in pyro.markov(enumerate(doc_list[d])):
                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}_{t}",
                                            dist.Categorical(theta),
                                            infer={"enumerate": "parallel"})
                # from that topic vec, select a word
                w = pyro.sample(f"w_{d}_{t}", dist.Categorical(Beta[z_assignment]), obs=y)

            X.append(w)
            Theta.append(theta)

        Theta = torch.stack(Theta)

        return X, Beta, Theta

    # def model(self, sigma):
    #     Beta = []
    #     # eta => prior for the per-topic word distributions
    #     eta = 1 + 0.01 * (2 * torch.rand(self.V) - 1)
    #
    #     # returns t x w matrix
    #     for k in pyro.plate("topics", self.D, subsample_size=self.S):
    #         beta = pyro.sample(f"beta_{k}", dist.Dirichlet(eta))
    #         Beta.append(beta.data.numpy())
    #
    #     Beta = torch.tensor(Beta)
    #     # alpha => prior for the per-doc topic vector
    #     alpha = torch.ones(self.K) / self.K + torch.rand(self.K) * 0.01
    #
    #     # eta => prior for regression coefficient
    #     weights_loc = torch.randn(self.K)
    #     weights_scale = torch.eye(self.K)
    #     eta = pyro.sample("eta", dist.MultivariateNormal(loc=weights_loc, covariance_matrix=weights_scale))
    #     # eta = torch.randn(self.K) * 2 - 1
    #     # sigma => prior for regression variance
    #     sigma_loc = torch.tensor(1.)
    #     sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    #
    #     # returns d x t matrix
    #     Theta = []
    #     y_list = []
    #     for d in pyro.plate("documents", self.D, subsample_size=self.S):
    #         # theta => per-doc topic vector
    #         theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))
    #         doc = None if data is None else data[d]
    #         z_bar = torch.zeros(self.K)
    #
    #         with pyro.plate(f"words_{d}", self.N[d]):
    #             z_assignment = pyro.sample(f"z_assignment_{d}",
    #                                        dist.Categorical(theta),
    #                                        infer={"enumerate": "parallel"})
    #
    #             w = pyro.sample(f"w_{d}", dist.Categorical(Beta[z_assignment]), obs=doc)
    #
    #         for z in z_assignment:
    #             z_bar[z] += 1
    #         z_bar /= self.N[d]
    #         mean = torch.dot(eta, z_bar)
    #         y_label = pyro.sample(f"doc_label_{d}", dist.Normal(mean, sigma), obs=torch.tensor([label[d]]))
    #         y_list.append(y_label)
    #
    #     return Theta, Beta, y_list
