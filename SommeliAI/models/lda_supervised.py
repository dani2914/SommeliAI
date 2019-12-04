import torch
from torch.distributions import constraints
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import TraceEnum_ELBO


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
        return TraceEnum_ELBO(max_plate_nesting=2)

    def model(self, data=None, label=None):
        """pyro model for lda"""

        # eta => prior for the per-topic word distributions
        eta_prior = torch.ones(self.V) / self.V

        # returns topic x vocab matrix
        with pyro.plate("topics", self.K):
            # eta = pyro.sample("eta",  dist.Gamma(1 / self.K, 1.))
            # beta => per topic word vec
            beta = pyro.sample("beta", dist.Dirichlet(eta_prior))
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

        # sigma => prior for regression variance
        sigma_loc = torch.tensor(0.1)
        sigma = pyro.sample(
            "sigma", dist.Normal(sigma_loc, torch.tensor(0.01)))

        # returns d x t matrix
        y_list = []
        for d in pyro.plate("documents", self.D):
            # theta => per-doc topic vector
            theta = pyro.sample("theta_" + str(d), dist.Dirichlet(alpha))

            doc = None if data is None else data[d]
            with pyro.plate("words_" + str(d), self.N[d]):
                # assign a topic
                z_assignment = pyro.sample(
                    "z_assignment_" + str(d),
                    dist.Categorical(theta)
                )
                # from that topic vec, select a word
                w = pyro.sample(
                    "w_" + str(d),
                    dist.Categorical(beta[z_assignment]),
                    obs=doc
                )
            z_bar = torch.zeros(self.K)
            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= self.N[d]
            mean = torch.dot(eta, z_bar)

            y_label = pyro.sample(
                "doc_label_" + str(d),
                dist.Normal(mean, sigma),
                obs=torch.tensor([label[d]])
            )
            y_list.append(y_label)

        return beta, y_list

    def guide(self, data=None, label=None):
        docs = []

        with pyro.plate("topics", self.K) as k_vec:
            # lambda_q => q for the per-topic word distribution

            lamda = torch.stack(
                [
                    pyro.param(
                        "lamda_q_" + str(k),
                        (1 + 0.01 * (2 * torch.rand(self.V) - 1)),
                        constraint=constraints.greater_than(0.5)
                    )
                    for k in k_vec.tolist()
                ]
            )

            # beta_q => posterior per topic word vec
            beta_q = pyro.sample("beta", dist.Dirichlet(lamda))

        # eta => prior for regression coefficient
        weights_loc = pyro.param('weights_loc', torch.randn(self.K) * 2 - 1)
        weights_scale = pyro.param('weights_scale', torch.eye(self.K),
                                   constraint=constraints.positive)
        eta = pyro.sample(
            "eta",
            dist.MultivariateNormal(
                loc=weights_loc,
                covariance_matrix=weights_scale
            )
        )
        # sigma => prior for regression variance

        # eta = pyro.param('eta', torch.randn(self.K) * 2 - 1)
        sigma_loc = pyro.param(
            'bias',
            torch.tensor(0.1),
            constraint=constraints.positive
        )
        sigma = pyro.sample(
            "sigma",
            dist.Normal(sigma_loc, torch.tensor(0.01))
        )

        z_assignments = []
        for d in pyro.plate("documents", self.D):
            # alpha_q => q for the per-doc topic vector
            gamma = pyro.param(
                "alpha_q_" + str(d),
                torch.ones(self.K) / self.K,
                constraint=constraints.positive
            )
            # theta_q => posterior per-doc topic vector

            theta_q = pyro.sample("theta_" + str(d), dist.Dirichlet(gamma))

            with pyro.plate("words_" + str(d), self.N[d]) as w_vec:
                # assign a topic
                phi_q = torch.stack(
                    [
                        pyro.param(
                            "phi_q_" + str(d) + "_" + str(w),
                            (1+0.01*(2*torch.rand(self.K)-1))/self.K,
                            constraint=constraints.simplex)
                        for w in w_vec.tolist()
                    ]
                )
                z_assignment = pyro.sample(
                    "z_assignment_" + str(d),
                    dist.Categorical(phi_q)
                )

            assert not any(np.isnan(gamma.detach().numpy()))
            # assert not any(np.isnan(phi_q.detach().numpy()))
            assert not any(np.isnan(z_assignment.detach().numpy()))
            z_bar = torch.zeros(self.K)
            for z in z_assignment:
                z_bar[z] += 1
            z_bar /= self.N[d]
            # mean = torch.dot(eta, z_bar)
            docs.append(label[d])
            z_assignments.append(z_assignment)
        return beta_q, eta, sigma, docs, z_assignments
