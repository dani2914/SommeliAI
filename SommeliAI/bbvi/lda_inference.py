import numpy as np

import lda_model
import read_data
from scipy.special import digamma
from math import lgamma
from math import isnan
from scipy.stats import dirichlet
from scipy.special import psi

np.random.seed(222)


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def lda_inference(doc, lda, adagrad=False):
    S = 10  # samples
    converged = 100.0
    rho = 1.e-3  # learning rate
    if adagrad:
        epsilon = 1.e-6  # fudge factor
        g_phi = np.zeros([doc.pdblength, lda.num_topics])
        g_var_gamma = np.zeros([lda.num_topics])

    # variational parameters

    phi = (
        np.ones(
            [doc.length, lda.num_topics]
        ) / lda.num_topics
    )  # N * k matrix
    var_gamma = lda.var_gamma[doc.index, :]

    likelihood_old = 0

    # integrate out phi at each step
    var_ite = 0
    # just take one step -- don't run until convergence
    while (converged > 1e-3 and var_ite < 1):
        var_ite += 1

        # sample S theta
        sample_theta = np.random.dirichlet(var_gamma, S)
        sample_theta = (sample_theta + 1e-2 / S) / (1 + 1e-2)

        # sample S z for each word n
        sample_zs = np.zeros([doc.length, S], dtype=np.int32)
        for n in range(doc.length):
            # sample S z for each word
            sample_z = np.random.multinomial(1, phi[n, :], S)  # S * k matrix
            which_j = np.argmax(sample_z, 1)  # S length vector
            sample_zs[n, :] = which_j

        # compute gamma gradient

        dig = digamma(var_gamma)
        var_gamma_sum = np.sum(var_gamma)
        digsum = digamma(var_gamma_sum)

        ln_theta = np.log(sample_theta)  # S * k matrix

        dqdg = ln_theta - dig + digsum  # S * k matrix

        # S length vectors
        ln_p_theta = dirichlet.logpdf(
            np.transpose(sample_theta),
            [lda.alpha] * lda.num_topics
        )
        ln_q_theta = dirichlet.logpdf(np.transpose(sample_theta), var_gamma)

        # explicitly evaluate expectation
        # E_p_z = np.sum(ln_theta * np.sum(phi, 0), 1) # S length vector

        # monte-carlo estimated expectation
        E_p_z = np.zeros(S)  # S length vector
        for sample_id in range(S):
            cur_ln_theta = ln_theta[sample_id, :]
            sampled_ln_theta = []
            for n in range(doc.length):
                which_j = sample_zs[n, :]
                # dim (doc.counts[n] * list(cur_ln_theta[which_j]))
                sampled_ln_theta += list(cur_ln_theta[which_j])
            E_p_z[sample_id] = np.average(sampled_ln_theta)

        grad_gamma = np.average(
            dqdg * np.reshape(ln_p_theta - ln_q_theta + E_p_z, (S, 1)),
            0
        )

        # update
        if adagrad:
            g_var_gamma += grad_gamma ** 2
            grad_gamma = grad_gamma / (np.sqrt(g_var_gamma) + epsilon)
        var_gamma = np.clip(var_gamma + rho * grad_gamma, 1e-2, np.inf)
        lda.var_gamma[doc.index, :] = var_gamma

        # for phi
        # for explicit evaluation of expectation
        # dig = digamma(var_gamma)
        # var_gamma_sum = np.sum(var_gamma)
        # digsum = digamma(var_gamma_sum)
        # resample from updated gamma
        sample_theta = np.random.dirichlet(var_gamma, S)
        ln_theta = np.log(sample_theta)  # S * k matrix

        for n in range(doc.length):
            # compute phi gradient
            which_j = sample_zs[n, :]

            dqdphi = 1 / phi[n][which_j]  # S length vector

            ln_p_w = lda.log_prob_w[which_j][:, doc.words[n]]  # S len vector

            ln_q_phi = np.log(phi[n][which_j])  # S length vector

            # explicitly evaluate expectation
            # E_p_z_theta = dig[which_j] - digsum # S length vector

            # monte-carlo estimated expectation
            E_p_z_theta = np.zeros(S)  # S length vector
            for sample_id in range(S):
                cur_ln_theta = ln_theta[sample_id, :]
                E_p_z_theta += cur_ln_theta[which_j]
            E_p_z_theta = E_p_z_theta / S

            grad_phi = (
                doc.counts[n] * dqdphi * (ln_p_w - ln_q_phi + E_p_z_theta)
            )

            # update phi

            for i, j in enumerate(which_j):
                if adagrad:
                    g_phi[n][j] += grad_phi[i] ** 2
                    grad_phi[i] = (
                        grad_phi[i] / (np.sqrt(g_phi[n][j]) + epsilon)
                    )
                # print grad_phi[i]
                phi[n][j] = phi[n][j] + rho * grad_phi[i]
                if phi[n][j] < 0:  # bound phi
                    phi[n][j] = 0
                phi[n] /= np.sum(phi[n])  # normalization
                if np.isnan(phi).any():
                    phi[np.isnan(phi).any(1), :] = 1. / lda.num_topics

        # assess convergence
        likelihood = compute_likelihood(doc, lda, phi, var_gamma)
        assert(not isnan(likelihood))
        converged = abs((likelihood_old - likelihood) / likelihood_old)
        likelihood_old = likelihood
        # print likelihood, converged

    # sample S theta from the updated gamma
    sample_theta = np.random.dirichlet(var_gamma, S)
    sample_theta = (sample_theta + 1e-2 / S) / (1 + 1e-2)

    # sample S z for each word n
    sample_zs = np.zeros([doc.length, S], dtype=np.int32)
    for n in range(doc.length):
        # sample S z for each word
        sample_z = np.random.multinomial(1, phi[n, :], S)  # S * k matrix
        which_j = np.argmax(sample_z, 1)  # S length vector
        sample_zs[n, :] = which_j

    # update betas
    for k in range(lda.num_topics):
        var_lambda = lda.var_lambda[k, :]
        sample_beta = np.random.dirichlet(var_lambda, S)

        dig = digamma(var_lambda)
        var_lambda_sum = np.sum(var_lambda)
        digsum = digamma(var_lambda_sum)

        ln_beta = np.log(sample_beta)  # S * k matrix

        dqdb = ln_beta - dig + digsum  # S * k matrix

        # ln_p_w = ln_beta[:, doc.words[n]] # S length vector

        # monte-carlo estimated expectation
        E_p_w_z_beta = np.zeros(S)  # S length vector
        for sample_id in range(S):
            sampled_ln_beta = 0.
            for n in range(doc.length):
                which_j = sample_zs[n, :]
                # (doc.counts[n] * list(cur_ln_theta[which_j]))
                sampled_ln_beta += ln_beta[which_j == k, doc.words[n]].sum()
            E_p_w_z_beta[sample_id] = sampled_ln_beta / doc.length

        # S length vectors
        ln_p_beta = (
            dirichlet.logpdf(
                np.transpose(sample_beta),
                [lda.eta] * lda.num_terms)
        )
        ln_q_beta = dirichlet.logpdf(np.transpose(sample_beta), var_lambda)

        grad_lambda = np.average(
            dqdb * np.reshape(ln_p_beta - ln_q_beta + ln_p_w, (S, 1)), 0)
        lda.var_lambda[k, :] = np.clip(
            var_lambda + rho * grad_lambda,
            1.e-1,
            np.inf
        )
        lda.log_prob_w[k, :] = dirichlet_expectation(lda.var_lambda[k, :])
    return likelihood


def compute_likelihood(doc, lda, phi, var_gamma):
    # an implementation reproducing lda-c code
    # var_gamma is a vector
    likelihood = 0
    digsum = 0
    var_gamma_sum = 0

    dig = digamma(var_gamma)
    var_gamma_sum = np.sum(var_gamma)
    digsum = digamma(var_gamma_sum)

    likelihood = (
        lgamma(lda.alpha * lda.num_topics)
        - lda.num_topics * lgamma(lda.alpha)
        - lgamma(var_gamma_sum)
    )

    for k in range(lda.num_topics):
        likelihood += (
            (lda.alpha - 1) * (dig[k] - digsum) +
            lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum)
        )

        for n in range(doc.length):
            if phi[n][k] > 0:
                likelihood += (
                    doc.counts[n] *
                    (
                        phi[n][k] * (
                            (dig[k] - digsum) - np.log(phi[n][k]) +
                            lda.log_prob_w[k, doc.words[n]]
                        )
                    )
                )

    return likelihood


def main():
    corpus = read_data.read_corpus("./data/tmp.data")
    model = lda_model.load_model("./model/final")
    niter = 200000
    docset = np.random.randint(0, len(corpus), niter)

    for i, e in enumerate(docset):
        print(i)
        _ = lda_inference(corpus[e], model, False)
        if (i % 10) == 0:
            lda_model.print_topics(model)
        if (i % 100) == 0:
            with open(f"./data/lambda_{i}", "w") as ofile:
                np.savetxt(ofile, model.var_lambda)
            with open(f"./data/gamma_{i}", "w") as ofile:
                np.savetxt(ofile, model.var_gamma)

    return


if __name__ == "__main__":
    main()
