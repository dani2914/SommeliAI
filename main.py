""" main driver """
import time
import argparse
import functools
import importlib
import util
import numpy as np
import pandas as pd
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
import torch
import pyro
from pyro.optim import Adam
from sklearn.preprocessing import StandardScaler
from customised_stopword import customised_stopword
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
import matplotlib.pyplot as plt

import pyro.poutine as poutine

import warnings
warnings.filterwarnings("ignore")

from models import (
    plainLDA,
    vaeLDA,
    supervisedLDA,
    sLDA_mcmc,
    originalLDA
)


def main(neural_args):
    ADAM_LEARN_RATE = 1e-3
    TESTING_SUBSIZE = 1000 #use None if want to use full dataset
    SUBSAMPLE_SIZE = 100
    USE_CUDA = False
    num_topic = 10
    RUN_NAME = f"vaeLDA_{TESTING_SUBSIZE}_{SUBSAMPLE_SIZE}_{num_topic}_enumerated"   # 1/k 1/k error 1/v 1/k error
    print(RUN_NAME)
    jitter = 1e-8
    # # #
    if USE_CUDA:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.DoubleTensor")

    full_df = util.fetch_dataset()

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = util.filter_by_topic(full_df, keep_top_n_topics=10)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.head(TESTING_SUBSIZE)

    # remove stop words, punctuation, digits and then change to lower case
    clean_df = util.preprocess(full_df, preprocess=True)
    print(clean_df)
    clean_df, indexed_txt_list, vocab_dict, vocab_count = util.preprocess_and_index(clean_df, ngram=1, custom_stopwords=customised_stopword)
    topic_vec = clean_df["variety"]

    scaler = StandardScaler()
    score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))
    unique_topics = np.unique(topic_vec)

    topic_map = {unique_topics[i]: i + 1 for i in range(len(unique_topics))}
    clean_df.loc[:, "class"] = clean_df["variety"].apply(lambda row: topic_map[row])
    label_list = score_vec.iloc[:, 0].tolist()
    #label_list = clean_df["class"].tolist()

    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    # create object of LDA class
    #
    # orig_lda = vaeLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)
    # orig_lda = originalLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE, jitter)
    # orig_lda = plainLDA(num_txt, num_words_per_txt,num_topic, num_vocab, SUBSAMPLE_SIZE)
    orig_lda = supervisedLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)
    # orig_lda = sLDA_mcmc(num_txt, num_words_per_txt,
    #                       num_topic, num_vocab, SUBSAMPLE_SIZE)
    if isinstance(orig_lda, supervisedLDA):
        args = (indexed_txt_list, label_list)
    else:
        args = (indexed_txt_list,)

    if isinstance(orig_lda, vaeLDA):
        predictor = orig_lda.make_predictor(neural_args)
        guide = functools.partial(orig_lda.parametrized_guide, predictor, vocab_count)
    elif isinstance(orig_lda, sLDA_mcmc):
        print("mcmc")
    else:
        guide = orig_lda.guide

    if isinstance(orig_lda, sLDA_mcmc):
        nuts_kernel = NUTS(orig_lda, jit_compile=0.01, )
        mcmc = MCMC(nuts_kernel,
                    num_samples=1000,
                    warmup_steps=5,
                    num_chains=5)
        mcmc.run(orig_lda, indexed_txt_list, label_list)
        mcmc.summary(prob=0.5)
    else:
        svi = pyro.infer.SVI(
            model=orig_lda.model,
            guide=guide,
            optim=Adam({"lr": 0.001, "betas": (0.90, 0.999)}),
            loss=orig_lda.loss)

        losses, alpha, beta = [], [], []
        num_step = 5001
        s1 = time.time()
        output = open("output" + str(s1) + ".txt", "a")
        loss_f = open("loss.txt", "a")
        score_loss = []
        for step in range(num_step):
            loss = svi.step(*args)
            losses.append(loss)
            if step % 10 == 0:
                print("{}: {}".format(step, np.round(loss, 1)))
                output.write("{}: {}".format(step, np.round(loss, 1)) + '\n')
                loss_f.write(str(np.round(loss, 1)) + '\n')
            if step % 100 == 0:
                alpha, alpha_ix = [], []
                phi, phi_ix = [], []
                # and np.isnan(site["value"].detach().numpy()).any():
                for i in range(num_txt):
                    try:
                        tensor = pyro.param(f"alpha_q_{i}")
                        alpha.append(tensor.detach().numpy())
                        alpha_ix.append(i)
                        tensor = pyro.param(f"phi_q_{i}")
                        phi.append(tensor.detach().numpy())
                        phi_ix.append(i)
                    except:
                        pass
                print("covered doc:" + str(len(alpha_ix)))
                # if step % 1000 == 0:
                #     for ix in alpha_ix:
                #         pd.DataFrame(alpha, index=alpha_ix).to_csv(f"results/alpha_{ix}_{step}.csv")
                #         pd.DataFrame(eta, index=eta_ix).to_csv(f"results/eta_{ix}_{step}.csv")
                # evaluate results
                dtype = [("word", "<U17"), ("index", int)]
                vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
                vocab = np.sort(vocab, order="index")
                if isinstance(orig_lda, supervisedLDA):
                    tr = poutine.trace(guide).get_trace(*args)
                    for name, site in tr.nodes.items():
                        if name == "eta":
                            eta = site["value"]
                        if name == "sigma":
                            sigma = site["value"]
                        if name == "beta":
                            beta = site["value"]
                    # beta, _, eta, sigma, y_label = guide(indexed_txt_list, label_list)
                    # beta = torch.tensor(beta)
                    y_label = []
                    y_gold = []
                    for ix in range(len(alpha_ix)):
                        phi_p = phi[ix]
                        z_assignment = np.random.choice(np.arange(num_topic), len(indexed_txt_list[alpha_ix[ix]]),
                                                        p=phi_p)
                        z_bar = np.zeros(num_topic)
                        for z in z_assignment:
                            z_bar[z] += 1
                        z_bar /= len(indexed_txt_list[alpha_ix[ix]])
                        mean = np.dot(eta.detach().numpy(), z_bar)
                        predicted_label = np.random.normal(mean, sigma.detach().numpy(), 1)
                        y_label.append(predicted_label)
                        y_gold.append(label_list[ix])
                    print("score loss:", np.mean((np.array(y_label) - np.array(y_gold)) ** 2))
                    output.write("score loss:" + str(np.mean((np.array(y_label) - np.array(y_gold)) ** 2)) + '\n')
                    score_loss.append(np.mean((np.array(y_label) - np.array(y_gold)) ** 2))
                else:
                    beta, _ = guide(indexed_txt_list)
                    beta = torch.tensor(beta)
                posterior_topics_x_words = beta.data.numpy()
                print(posterior_topics_x_words)
                for i in range(num_topic):
                    sorted_words_ix = np.argsort(posterior_topics_x_words[i])[::-1]
                    print("topic %s" % i)
                    print([word[0] for word in vocab[sorted_words_ix][:10]])
                    output.write("topic %s" % i + '\n')
                    output.write(" ".join([word[0] for word in vocab[sorted_words_ix]][:10]) + '\n')

        plt.plot(losses)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")

    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="128-128")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
