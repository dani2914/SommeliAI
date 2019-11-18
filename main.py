""" main driver """

import argparse
import functools
import importlib
import re
import os
import util
import numpy as np
import pandas as pd
import time
import torch
import pyro
from pyro.optim import Adam
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive

from models import (
    plainLDA#,
    #vaniLDA,
    #vaeLDA,
    #supervisedLDA
)

#pyro.set_rng_seed(0)
#pyro.clear_param_store()
#pyro.enable_validation(False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")

    # parser.add_argument("-n", "--num-steps", default=1000, type=int)
    # parser.add_argument("-l", "--layer-sizes", default="128-128")
    # parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    # parser.add_argument('--jit', action='store_true')
    # neural_args = parser.parse_args()

    # CONSTANTS
    ADAM_LEARN_RATE = 0.01
    TESTING_SUBSIZE = None #use None if want to use full dataset
    SUBSAMPLE_SIZE = 100
    USE_CUDA = False
    ix = round(time.time())
    os.mkdir(f"results/{ix}")

    print("alpha: rand 1/10; eta: rand 1; phi: rand 1/10")
    print(ix)

    if USE_CUDA:
        torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        torch.set_default_tensor_type("torch.DoubleTensor")

    full_df = util.fetch_dataset()

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = util.filter_by_topic(full_df, keep_top_n_topics=10)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.head(TESTING_SUBSIZE)



    # stop_words = ['acidity', 'age', 'apple', 'aroma', 'balance', 'berry', 'black',
    #   'blackberry', 'blend', 'cabernet', 'cherry', 'chocolate', 'citrus',
    #   'crisp', 'currant', 'dark', 'drink', 'dry', 'finish', 'flavor',
    #   'fresh', 'fruit', 'full', 'give', 'good', 'green', 'ha', 'herb',
    #   'hint', 'juicy', 'lemon', 'light', 'make', 'merlot', 'nose',
    #   'note', 'oak', 'offer', 'palate', 'peach', 'pepper', 'pinot',
    #   'plum', 'raspberry', 'red', 'rich', 'ripe', 'sauvignon', 'show',
    #   'soft', 'spice', 'structure', 'sweet', 'tannin', 'texture',
    #   'toast', 'vanilla', 'vineyard', 'well', 'wine', 'year']

    # remove stop words, punctuation, digits and then change to lower case
    #clean_df = util.preprocess(full_df, preprocess=True)
    clean_df, indexed_txt_list, vocab_dict, vocab_count = \
        util.preprocess_and_index(full_df, ngram=1)#, pre_stopwords=stop_words)
    #ix_dict = {v: k for k, v in vocab_dict.items()}

    txt_vec = clean_df["description"]
    topic_vec = clean_df["variety"]
    score_vec = clean_df["points"].astype(np.float64)
    unique_topics = np.unique(topic_vec)

    #indexed_txt_list, vocab_dict, vocab_count = util.conv_word_to_indexed_txt(txt_vec)

    topic_map = {unique_topics[i]:i for i in range(len(unique_topics))}
    clean_df.loc[:, "class"] = clean_df["variety"].apply(lambda row: topic_map[row])
    label_list = clean_df.loc[:, "class"].tolist()

    num_topic = len(unique_topics)
    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    # evaluate results
    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")


    # create object of LDA class
    # orig_lda = origLDA(num_txt, num_words_per_txt, num_topic, num_vocab)
    #orig_lda = vaeLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)

    orig_lda = plainLDA(num_txt, num_words_per_txt,
                        num_topic, num_vocab, SUBSAMPLE_SIZE)
    # orig_lda = supervisedLDA(num_txt, num_words_per_txt,
    #                     num_topic, num_vocab, SUBSAMPLE_SIZE)
    #orig_lda = plainLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)

    # if isinstance(orig_lda, supervisedLDA):
    #     args = (indexed_txt_list, label_list)
    # else:
    args = (indexed_txt_list,)

    # if isinstance(orig_lda, vaeLDA):
    #     predictor = orig_lda.make_predictor(neural_args)
    #     guide = functools.partial(orig_lda.parametrized_guide, predictor, vocab_count)
    #
    # else:
    guide = orig_lda.guide

    svi = pyro.infer.SVI(
        model=orig_lda.model,
        guide=guide,
        optim=Adam({"lr": ADAM_LEARN_RATE}),
        loss=orig_lda.loss)

    pd.DataFrame(vocab).to_csv(f"results/{ix}/dict_{ix}.csv")

    losses = []
    num_step = 5001
    for step in range(num_step):
        loss = svi.step(*args)
        losses.append(loss)

        if step % 10 == 0:
            print("{}: {}".format(step, np.round(loss, 1)))

            beta_q = guide()[0].cpu()
            pd.DataFrame(beta_q.detach().numpy().T).to_csv(f"results/{ix}/beta_q_{ix}_{step}.csv")

            theta_q = guide()[1].cpu()
            pd.DataFrame(theta_q.detach().numpy().T).to_csv(f"results/{ix}/theta_q_{ix}_{step}.csv")

            pd.DataFrame(losses).to_csv(f"results/{ix}/loss_{ix}.csv")

        if step % 100 == 0:
            for i in range(num_topic):
                sorted_words_ix = torch.argsort(beta_q[i])
                print("topic %s" % i)
                print([word[0] for word in vocab[sorted_words_ix][-10:]])

        if step % 100 == 0:
            lamda, lamda_ix = [], []
            for k in range(num_topic):
                tensor = pyro.param(f"lamda_q_{k}")
                lamda.append(tensor.detach().cpu().numpy())
                lamda_ix.append(k)

            gamma, gamma_ix = [], []
            #phi, w_ix, d_ix = [[]], []
            for d in range(num_txt):
                try:
                    tensor = pyro.param(f"gamma_q_{d}")
                    gamma.append(tensor.detach().cpu().numpy())
                    gamma_ix.append(d)

                    # for w in range(num_words_per_txt[d]):
                    #     tensor = pyro.param(f"phi_q_{d}_{w}")
                    #     phi.append(tensor.detach().cpu().numpy())
                    #     phi_ix.append(i)

                except:
                    pass


            pd.DataFrame(lamda, index=lamda_ix).to_csv(f"results/{ix}/lamda_{ix}_{step}.csv")
            pd.DataFrame(gamma, index=gamma_ix).to_csv(f"results/{ix}/gamma_{ix}_{step}.csv")
            #pd.DataFrame(phi, index=phi_ix).to_csv(f"results/{ix}/phi_{ix}_{step}.csv")



        # if step % 100 == 0:
        #     alpha, alpha_ix = [], []
        #     eta, eta_ix = [], []
        #     for i in range(num_txt):
        #         try:
        #             tensor = pyro.param(f"alpha_q_{i}")
        #             alpha.append(tensor.detach().cpu().numpy())
        #             alpha_ix.append(i)

        #             tensor = pyro.param(f"eta_q_{i}")
        #             eta.append(tensor.detach().cpu().numpy())
        #             eta_ix.append(i)
        #         except:
        #             pass

        #     pd.DataFrame(alpha, index=alpha_ix).to_csv(f"results/{ix}/alpha_{ix}_{step}.csv")
        #     pd.DataFrame(eta, index=eta_ix).to_csv(f"results/{ix}/eta_{ix}_{step}.csv")

        # if step % 500 == 0:
        #     for i in range(num_topic):
        #         sorted_words_ix = torch.argsort(posterior_topics_x_words[i])
        #         print("topic %s" % i)
        #         print([word[0] for word in vocab[sorted_words_ix][-10:]])

    # posterior_topics_x_words = dist.Dirichlet(pyro.param("phi")).sample()

    #posterior_doc_x_words, posterior_topics_x_words = \
    #        orig_lda.model(*args)
    # posterior_topics_x_words = posterior_topics_x_words.cpu()

    # preds = []
    #
    # for i in range(1000):
    #     if isinstance(orig_lda, supervisedLDA):
    #         pred = guide(indexed_txt_list, label_list).data.numpy()
    #     else:
    #         pred = guide(indexed_txt_list).data.numpy()
    #     preds.append(pred)
    #
    # posterior_topics_x_words = np.stack(preds).mean(0)

#    for i in range(num_topic):
#        non_trivial_words_ix = np.where(posterior_topics_x_words[i] > 0.005)[0]
#        print("topic %s" % i)
#        print([word[0] for word in vocab[non_trivial_words_ix]])
#
#     posterior_topics_x_words = guide()[0]
#     posterior_topics_x_words = posterior_topics_x_words.cpu()
#     theta_q = guide()[1]
#
#     for i in range(num_topic):
#         sorted_words_ix = torch.argsort(posterior_topics_x_words[i])
#         print("topic %s" % i)
#         print([word[0] for word in vocab[sorted_words_ix][-10:]])
#
#     ix = round(time.time())
#
#     tmp_df = posterior_topics_x_words
#     pd.DataFrame(tmp_df.detach().numpy().T).to_csv(f"results/beta_q_{ix}.csv")
#     pd.DataFrame(vocab).to_csv(f"results/dict_{ix}.csv")
