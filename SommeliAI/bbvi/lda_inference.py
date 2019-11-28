import numpy as np
import numba as nb
import lda_model
import read_data
from numba.extending import get_cython_function_address
from numba import vectorize
from math import lgamma
import ctypes

np.random.seed(123)

addr = get_cython_function_address(
    "scipy.special.cython_special",
    "__pyx_fuse_1psi"
)
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
psi_fn = functype(addr)


@vectorize('float64(float64)')
def vec_psi(x):
    return psi_fn(x)


@nb.jit(nopython=True)
def psi(x):
    return vec_psi(x)


@nb.jit(nopython=True)
def dirichlet_logpdf(probs, concentrations):
    out = np.sum((concentrations - 1.) * np.log(probs))
    out += lgamma(np.sum(concentrations))
    for c in range(len(concentrations)):
        out -= lgamma(concentrations[c])
    return out


@nb.jit(nopython=True)
def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


@nb.jit(nopython=True)
def random_dirichlet(alpha):
    out = np.zeros(alpha.shape[0])
    for i, a in enumerate(alpha):
        y = np.random.gamma(a)
        out[i] = y
    out /= out.sum()
    return out


@nb.jit(nopython=True)
def jit_inference(
    doc_words,
    doc_counts,
    alpha,
    var_gamma,
    var_phi,
    log_prob_w
):
    S = 100  # samples
    rho = 1.e-3  # learning rate
    num_topics = log_prob_w.shape[0]
    doc_length = len(doc_words)

    # sample S theta
    sample_theta = np.zeros((S, len(var_gamma)))
    for s in range(S):
        sample_theta[s, :] = random_dirichlet(np.where(var_gamma > 1e-3, var_gamma, 1e-3))
    # sample_theta = (sample_theta + 1e-2 / S) / (1 + 1e-2)

    # sample S z for each word n
    sample_zs = np.zeros((doc_length, S), dtype=np.int32)
    for n in range(doc_length):
        # sample S z for each word
        sample_z = np.random.multinomial(1, var_phi[n], S)  # S * k matrix
        which_j = np.zeros(S, dtype=np.int32)
        for s in range(S):
            which_j[s] = np.argmax(sample_z[s, :])  # S length vector
            sample_zs[n, s] = which_j[s]

    # compute gamma gradient

    dig = psi(var_gamma)
    var_gamma_sum = np.sum(var_gamma)
    digsum = psi(var_gamma_sum)

    ln_theta = np.log(sample_theta)  # S * k matrix

    dqdg = ln_theta - dig + digsum  # S * k matrix

    # S length vectors
    ln_p_theta = dirichlet_logpdf(
        sample_theta,
        alpha * np.ones(num_topics)
    )
    ln_q_theta = dirichlet_logpdf(sample_theta, var_gamma)

    # explicitly evaluate expectation
    # E_p_z = np.sum(ln_theta * np.sum(phi, 0), 1) # S length vector

    # monte-carlo estimated expectation
    E_p_z = np.zeros(S)  # S length vector
    for sample_id in range(S):
        cur_ln_theta = ln_theta[sample_id, :]
        sampled_ln_theta = 0.
        for n in range(doc_length):
            which_j[:] = sample_zs[n, :]
            # dim (doc_counts[n] * list(cur_ln_theta[which_j]))
            sampled_ln_theta += np.sum(cur_ln_theta[which_j.astype(np.int64)])
        E_p_z[sample_id] = sampled_ln_theta / (S * doc_length)

    grad_gamma = np.zeros(num_topics)
    for s in range(S):
        grad_gamma += dqdg[s, :] * (ln_p_theta - ln_q_theta + E_p_z)[s]
    grad_gamma /= S

    # update
    var_gamma = var_gamma + rho * grad_gamma
    for k in range(len(var_gamma)):
        if var_gamma[k] < 1e-2:
            var_gamma[k] = 1.e-2
    # for phi
    # for explicit evaluation of expectation
    # dig = psi(var_gamma)
    # var_gamma_sum = np.sum(var_gamma)
    # digsum = psi(var_gamma_sum)
    # resample from updated gamma
    sample_theta = np.zeros((S, num_topics))
    for s in range(S):
        sample_theta[s, :] = random_dirichlet(var_gamma)
    ln_theta = np.log(sample_theta)  # S * k matrix

    for n in range(doc_length):
        # compute phi gradient
        which_j = sample_zs[n, :]

        dqdphi = 1 / var_phi[n][which_j]  # S length vector

        ln_p_w = log_prob_w[which_j][:, doc_words[n]]  # S len vector

        ln_q_phi = np.log(var_phi[n][which_j])  # S length vector

        # explicitly evaluate expectation
        # E_p_z_theta = dig[which_j] - digsum # S length vector

        # monte-carlo estimated expectation
        E_p_z_theta = np.zeros(S)  # S length vector
        for sample_id in range(S):
            cur_ln_theta = ln_theta[sample_id, :]
            E_p_z_theta += cur_ln_theta[which_j]
        E_p_z_theta = E_p_z_theta / S

        grad_phi = (
            doc_counts[n] * dqdphi * (ln_p_w - ln_q_phi + E_p_z_theta)
        )

        # update phi

        for i, j in enumerate(which_j):
            var_phi[n][j] = var_phi[n][j] + rho * grad_phi[i]
            if var_phi[n][j] < 0:  # bound phi
                var_phi[n][j] = 0
            var_phi[n] /= (np.sum(var_phi[n]) + 1e-7)  # normalization
        var_phi = np.where(~np.isnan(var_phi), var_phi, 1. / num_topics)

    return var_gamma, var_phi

@nb.jit(nopython=True)
def jit_m_step(doc_words, doc_counts, eta, var_gamma, var_phi, log_prob_w, var_lambda):
    # update betas
    S = 100  # samples
    rho = 1.e-4  # learning rate
    num_topics = log_prob_w.shape[0]
    num_terms = log_prob_w.shape[1]
    doc_length = len(doc_words)

    # sample S theta from the updated gamma
    sample_theta = random_dirichlet(var_gamma)
    sample_theta = (sample_theta + 1e-4 / S) / (1 + 1e-4)

    # sample S z for each word n
    sample_zs = np.zeros((doc_length, S), dtype=np.int32)
    for n in range(doc_length):
        # sample S z for each word
        # print("here?")
        # print(var_phi[n, :])
        sample_z = np.random.multinomial(1, var_phi[n, :], S)  # S * k matrix
        # print("here!")
        which_j = np.zeros(S, dtype=np.int32)
        for s in range(S):
            which_j[s] = np.argmax(sample_z[s, :])  # S length vector
            sample_zs[n, s] = which_j[s]

    for k in range(num_topics):
        lbound = -1.e-10

        sample_beta = np.zeros((S, num_terms))
        for s in range(S):
            sample_beta[s, :] = random_dirichlet(var_lambda[k, :])
        sample_beta = (sample_beta + 1e-4 / S) / (1 + 1e-4)

        dig = psi(var_lambda[k, :])
        var_lambda_sum = np.sum(var_lambda[k, :])
        digsum = psi(var_lambda_sum)

        ln_beta = np.log(sample_beta)  # S * V matrix
        ln_beta = np.where(np.isnan(ln_beta) | (ln_beta < lbound), lbound, ln_beta)

        dqdb = ln_beta - dig + digsum  # [:, np.newaxis]  # S * V matrix
        # raise ValueError("i")
        # ln_p_w = ln_beta[:, doc_words[n]] # S length vector

        # monte-carlo estimated expectation
        E_p_w_z_beta = np.zeros(S)  # S length vector
        for sample_id in range(S):
            sampled_ln_beta = 0.
            for n in range(doc_length):
                which_j = sample_zs[n, :]
                # (doc_counts[n] * list(cur_ln_theta[which_j]))
                sampled_ln_beta += ln_beta[which_j == k, doc_words[n]].sum()
            E_p_w_z_beta[sample_id] = sampled_ln_beta / doc_length

        # S length vectors
        ln_p_beta = np.zeros(S)
        for s in range(S):
            ln_p_beta[s] = (
                dirichlet_logpdf(
                    sample_beta[s, :],
                    eta * np.ones(num_terms))
            )
        ln_p_beta = np.where(np.isnan(ln_p_beta) | (ln_p_beta < lbound), lbound, ln_p_beta)
        ln_q_beta = np.zeros(S)
        for s in range(S):
            ln_q_beta[s] = dirichlet_logpdf(sample_beta[s, :], var_lambda[k, :])
        ln_q_beta = np.where(np.isnan(ln_q_beta) | (ln_q_beta < lbound), lbound, ln_q_beta)

        grad_lambda = np.zeros(num_terms)
        for s in range(S):
            # print(ln_p_beta[s], ln_q_beta[s], E_p_w_z_beta[s], dqdb[s])
            grad_lambda += dqdb[s] * (ln_p_beta[s] - ln_q_beta[s] + E_p_w_z_beta[s])
        grad_lambda /= S
        var_lambda[k, :] = np.where(
            var_lambda[k, :] + rho * grad_lambda >= 1.e-1,
            var_lambda[k, :] + rho * grad_lambda,
            1e-1
        )
    return var_lambda


if __name__ == "__main__":
    corpus = read_data.read_corpus("./data/tmp.data")
    lda = lda_model.load_model("./model/final")
    niter = 200000
    docset = np.random.randint(0, len(corpus), niter)

    for i, e in enumerate(docset):
        print(i)
        if lda.var_phi[e].shape[0] == 0:
            lda.var_phi[e] = np.ones((len(corpus[e].words), lda.num_topics)) / lda.num_topics
        lda.var_gamma[e, :], lda.var_phi[e] = \
            jit_inference(
            corpus[e].words,
            corpus[e].counts,
            lda.alpha,
            lda.var_gamma[e, :],
            lda.var_phi[e],
            lda.log_prob_w
        )
        if i == 100:
            raise ValueError("oh!")
        lda.var_lambda = jit_m_step(
            corpus[e].words,
            corpus[e].counts,
            lda.eta,
            lda.var_gamma[e, :],
            lda.var_phi[e],
            lda.log_prob_w,
            lda.var_lambda
        )
        if (i % 10) == 0:
            lda_model.print_topics(lda)
        if (i % 100) == 0:
            with open(f"./data/lambda_{i}", "w") as ofile:
                np.savetxt(ofile, lda.var_lambda)
            with open(f"./data/gamma_{i}", "w") as ofile:
                np.savetxt(ofile, lda.var_gamma)
