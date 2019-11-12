""" main driver """

import argparse
import functools
import importlib
import util
import numpy as np
import torch
import pyro
from pyro.optim import Adam
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive

from models import (
    plainLDA,
    vaniLDA,
    vaeLDA,
    supervisedLDA
)

#pyro.set_rng_seed(0)
#pyro.clear_param_store()
#pyro.enable_validation(False)


def main(neural_args):
    """ main function """

    # CONSTANTS
    ADAM_LEARN_RATE = 0.01
    TESTING_SUBSIZE = None #use None if want to use full dataset
    SUBSAMPLE_SIZE = 50
    USE_CUDA = True

    # if USE_CUDA:
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")

    full_df = util.fetch_dataset()

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = util.filter_by_topic(full_df, keep_top_n_topics=10)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.head(TESTING_SUBSIZE)

    # remove stop words, punctuation, digits and then change to lower case
    clean_df = util.preprocess(full_df, preprocess=True)

    txt_vec = clean_df["description"]
    topic_vec = clean_df["variety"]
    score_vec = clean_df["points"].astype(np.float64)
    unique_topics = np.unique(topic_vec)

    indexed_txt_list, vocab_dict, vocab_count = util.conv_word_to_indexed_txt(txt_vec)

    topic_map = {unique_topics[i]: i + 1 for i in range(len(unique_topics))}
    clean_df.loc[:, "class"] = clean_df["variety"].apply(lambda row: topic_map[row])
    label_list = score_vec.tolist()
    #label_list = clean_df["class"].tolist()

    num_topic = len(unique_topics)
    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

#    # purely for testing purposes, overrides original val with toy dataset
#    indexed_txt_list = [
#        torch.tensor([1, 2, 3, 4, 5]),
#        torch.tensor([0, 2, 4, 6, 8, 9]),
#        torch.tensor([1, 3, 5, 7]),
#        torch.tensor([5, 6, 7])]
#    num_topic = 3
#    num_vocab = len(np.unique([word for txt in indexed_txt_list for word in txt]))
#    num_txt = len(indexed_txt_list)
#    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    # create object of LDA class
    # orig_lda = origLDA(num_txt, num_words_per_txt, num_topic, num_vocab)
    # orig_lda = vaeLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)

    # orig_lda = plainLDA(num_txt, num_words_per_txt,
    #                      num_topic, num_vocab, SUBSAMPLE_SIZE)
    orig_lda = supervisedLDA(num_txt, num_words_per_txt,
                          num_topic, num_vocab, SUBSAMPLE_SIZE)
    #orig_lda = plainLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)

    if isinstance(orig_lda, supervisedLDA):
        args = (indexed_txt_list, label_list)
    else:
        args = (indexed_txt_list,)

    if isinstance(orig_lda, vaeLDA):
        predictor = orig_lda.make_predictor(neural_args)
        guide = functools.partial(orig_lda.parametrized_guide, predictor, vocab_count)
    else:
        guide = orig_lda.guide

    svi = pyro.infer.SVI(
        model=orig_lda.model,
        guide=guide,
        optim=Adam({"lr": ADAM_LEARN_RATE}),
        loss=orig_lda.loss)

    losses, alpha, beta = [], [], []
    num_step = 500
    for step in range(num_step):
        loss = svi.step(*args)
        losses.append(loss)
        if isinstance(orig_lda, plainLDA):
           alpha.append(pyro.param("alpha_q"))
           beta.append(pyro.param("beta_q"))
        if step % 10 == 0:
            print("{}: {}".format(step, np.round(loss, 1)))
    # evaluate results
    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")

    # posterior_topics_x_words = dist.Dirichlet(pyro.param("phi")).sample()

    #posterior_doc_x_words, posterior_topics_x_words = \
    #        orig_lda.model(*args)
    # posterior_topics_x_words = posterior_topics_x_words.cpu()

    preds = []

    for i in range(1000):
        if isinstance(orig_lda, supervisedLDA):
            pred = guide(indexed_txt_list, label_list).data.numpy()
        else:
            pred = guide(indexed_txt_list).data.numpy()
        preds.append(pred)

    posterior_topics_x_words = np.stack(preds).mean(0)
    np.savetxt('posterior_topic_words.csv', posterior_topics_x_words, delimiter=',')
#    for i in range(num_topic):
#        non_trivial_words_ix = np.where(posterior_topics_x_words[i] > 0.005)[0]
#        print("topic %s" % i)
#        print([word[0] for word in vocab[non_trivial_words_ix]])
    output = open("posterior_topic_words.csv", "a")
    for i in range(num_topic):
        sorted_words_ix = np.argsort(posterior_topics_x_words[i])[::-1]
        print("topic %s" % i)
        print([word[0] for word in vocab[sorted_words_ix][:20]])
        output.write(" ".join([word[0] for word in vocab[sorted_words_ix]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")

    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="128-128")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
