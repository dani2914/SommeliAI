import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO

class meanfieldLDA:

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
        return Trace_ELBO(max_plate_nesting=2)

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

            doc = None if doc_list is None else doc_list[d]

            with pyro.plate(f"words_{d}", self.N[d]):

                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}", dist.Categorical(theta))               

                # from that topic vec, select a word
                X = pyro.sample(f"w_{d}", dist.Categorical(Beta[z_assignment]), obs=doc)

            X_List.append(X)
            Theta.append(theta)

        Theta = torch.stack(Theta)

        return X_List, Beta, Theta
        

    def guide(self, doc_list=None):
        """pyro guide for lda inference"""

        with pyro.plate("topics", self.K) as k_vec:

            # Lambda => latent variable for the per-topic word q distribution
            Lamda = torch.stack([pyro.param(f"lamda_q_{k}", 
            (1 + 0.01*(2*torch.rand(self.V)-1)), 
            constraint=constraints.positive) for k in k_vec])

            # Beta_q => per-topic word q distribtion
            Beta_q = pyro.sample(f"beta", dist.Dirichlet(Lamda))

        Theta_q = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):

            # gamma => q for the per-doc topic vector
            gamma = pyro.param(f"gamma_q_{d}",
                               (1+0.01*(2*torch.rand(self.K)-1))/self.K, 
                               constraint=constraints.positive)
                               
            # theta_q => posterior per-doc topic vector
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma))

            phi = pyro.param(f"phi_q_{d}",
                            (1+0.01*(2*torch.rand(self.K)-1))/self.K, 
                            constraint=constraints.positive)

            with pyro.plate(f"words_{d}", self.N[d]) as w_vec:

                phi = torch.stack([pyro.param(f"phi_q_{d}_{w}",
                (1+0.01*(2*torch.rand(self.K)-1))/self.K, 
                constraint=constraints.positive) for w in w_vec])

                # assign a topic
                pyro.sample(f"z_assignment_{d}", dist.Categorical(phi))

            Theta_q.append(theta_q)

        Theta_q = torch.stack(Theta_q)

        return Beta_q, Theta_q