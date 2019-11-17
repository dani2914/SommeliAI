import torch
from torch.distributions import constraints
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

class supervisedLDA():
    def __init__(self, num_docs, num_words_per_doc,
                 num_topics, num_vocabs, num_subsample):
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
        return Trace_ELBO(max_plate_nesting=2)

    def model(self, data=None, label=None):
        """pyro model for lda"""

        # eta => prior for the per-topic word distributions
        eta_prior = torch.ones(self.V) / self.V

        # returns topic x vocab matrix
        # with pyro.plate("topics", self.K):
        #     # beta => per topic word vec
        #     beta = pyro.sample(f"beta", dist.Dirichlet(eta))
        # beta = torch.zeros((self.K, self.V))
        with pyro.plate("topics", self.K):
            # eta = pyro.sample("eta",  dist.Gamma(1 / self.K, 1.))
            # beta => per topic word vec
            beta = pyro.sample(f"beta", dist.Dirichlet(eta_prior))
        # alpha => prior for the per-doc topic vector
        alpha = torch.ones(self.K) / self.K

        # eta => prior for regression coefficient
        weights_loc = torch.randn(self.K) * 2 - 1
        weights_scale = torch.eye(self.K)
        eta = pyro.sample(
            "eta",
            dist.MultivariateNormal(
                loc=weights_loc,
                covariance_matrix=weights_scale
            )
        )
        # eta = torch.randn(self.K)
        # sigma => prior for regression variance
        sigma_loc = torch.tensor(1.)
        # sigma = torch.tensor(1.)
        sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.001)))

        # returns d x t matrix
        y_list = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            # theta => per-doc topic vector
            theta = pyro.sample(f"theta_{d}", dist.Dirichlet(alpha))

            doc = None if data is None else data[d]
            with pyro.plate(f"words_{d}", self.N[d]):
                # assign a topic
                z_assignment = pyro.sample(
                    f"z_assignment_{d}",
                    dist.Categorical(theta)
                )
                # from that topic vec, select a word
                w = pyro.sample(
                    f"w_{d}",
                    dist.Categorical(beta[z_assignment]),
                    obs=doc
                )
            z_bar = torch.zeros(self.K)
            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= self.N[d]
            mean = torch.dot(eta, z_bar)

            y_label = pyro.sample(
                f"doc_label_{d}",
                dist.Normal(mean, sigma),
                obs=torch.tensor([label[d]])
            )
            y_list.append(y_label)

        return beta, y_list

    def guide(self, data=None, label=None):
        docs = []
        # Beta_q = torch.zeros((self.K, self.V))
        # Beta_q = torch.zeros((self.K, self.V))
        # eta_posterior = pyro.param(
        #     "eta_posterior",
        #     lambda: torch.ones(self.K),
        #     constraint=constraints.positive)
        lambda_q = pyro.param(
            "lambda_q",
            lambda: (1 + 0.01 * (2 * torch.rand(self.K, self.V) - 1)),
            constraint=constraints.greater_than(0.5))

        with pyro.plate("topics", self.K):
            # eta_q => q for the per-topic word distribution
            # eta_q = pyro.param(f"eta_q", (1 + 0.01 * (2 * torch.rand(self.K, self.V) - 1)),
            #        constraint=constraints.positive)
            # #torch.ones(self.V) / self.K, constraint=constraints.positive)
            # #/ self.V, torch.rand(self.V),
            # beta_q => posterior per topic word vec
            # eta_q += self.jitter
            beta_q = pyro.sample(f"beta", dist.Dirichlet(lambda_q))

            # eta = pyro.sample("eta", dist.Gamma(eta_posterior, 1.))

            # eta_q = pyro.param(f"eta_q_{k}", torch.ones(self.V) / self.V,
            #                    constraint=constraints.positive)
            # beta[k, :] = pyro.sample(f"beta_{k}", dist.Dirichlet(eta_q))

        # eta => prior for regression coefficient
        weights_loc = pyro.param('weights_loc', torch.randn(self.K) * 2 - 1)
        weights_scale = pyro.param('weights_scale', torch.eye(self.K),
                                   constraint=constraints.positive)
        eta = pyro.sample("eta",
                          dist.MultivariateNormal(
                              loc=weights_loc,
                              covariance_matrix=weights_scale
                          )
                        )
        # sigma => prior for regression variance

        # eta = pyro.param('coef', torch.randn(self.K) * 2 - 1)
        sigma_loc = pyro.param('bias', torch.tensor(1.), constraint=constraints.positive)
        sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.001)))
        # sigma = pyro.param('bias', torch.tensor(1.), constraint=constraints.positive)
        # eta = pyro.param('coef', torch.randn(self.K))
        # sigma = pyro.param('bias', torch.tensor(1.), constraint=constraints.positive)
        # assert not any(np.isnan(eta.detach().numpy()))
        # assert sigma
        z_assignments = []
        for d in pyro.plate("documents", self.D, subsample_size=self.S):
            # alpha_q => q for the per-doc topic vector
            gamma = pyro.param(
                f"alpha_q_{d}",
                torch.ones(self.K) / self.K,
                constraint=constraints.positive
            )  # / / self.K, torch.rand(self.K),
            # theta_q => posterior per-doc topic vector
            theta_q = pyro.sample(f"theta_{d}", dist.Dirichlet(gamma))

            phi_q = pyro.param(f"phi_q_{d}", torch.ones(self.K) / self.K,
                               constraint=constraints.simplex)  # / / self.K, torch.rand(self.K),
            with pyro.plate(f"words_{d}", self.N[d]):
                # assign a topic
                z_assignment = pyro.sample(f"z_assignment_{d}", dist.Categorical(phi_q))
            assert not any(np.isnan(gamma.detach().numpy()))
            assert not any(np.isnan(phi_q.detach().numpy()))
            assert not any(np.isnan(z_assignment.detach().numpy()))
            z_bar = torch.zeros(self.K)
            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= self.N[d]
            mean = torch.dot(eta, z_bar)
            docs.append(label[d])
            z_assignments.append(z_assignment)
        return beta_q, eta, sigma, docs, z_assignments
