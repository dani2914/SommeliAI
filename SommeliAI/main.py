""" main driver """

import argparse
import functools
import pandas as pd
import data_util
import numpy as np
import pyro
from pyro.optim import Adam
from sklearn.preprocessing import MinMaxScaler

from models import (
    plainLDA,
    supervisedLDA
)


if __name__ == "__main__":

    # CONSTANTS
    ADAM_LEARN_RATE = 0.01
    TESTING_SUBSIZE = 0  # use None if want to use full dataset
    SUBSAMPLE_SIZE = 100
    USE_CUDA = True

    # if USE_CUDA:
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")

    full_df = data_util.fetch_dataset()

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=10)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.sample(frac=TESTING_SUBSIZE, replace=False, random_state=666)

    # remove stop words, punctuation, digits and then change to lower case
    clean_df = data_util.preprocess(full_df, preprocess=True)

    txt_vec = clean_df["description"]
    topic_vec = clean_df["variety"]
    scaler = MinMaxScaler()
    score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))

    unique_topics = np.unique(topic_vec)

    (indexed_txt_list,
        vocab_dict,
        vocab_count) = data_util.conv_word_to_indexed_txt(
        txt_vec
    )

    topic_map = {unique_topics[i]: i + 1 for i in range(len(unique_topics))}
    clean_df.loc[:, "class"] = clean_df["variety"].apply(
        lambda row: topic_map[row]
    )
    label_list = score_vec.tolist()

    num_topic = len(unique_topics)
    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    orig_lda = supervisedLDA(
        num_txt, num_words_per_txt,
        num_topic, num_vocab, SUBSAMPLE_SIZE
    )
    # orig_lda = plainLDA(
    #   num_txt, num_words_per_txt,
    #   num_topic, num_vocab, SUBSAMPLE_SIZE
    # )

    if isinstance(orig_lda, supervisedLDA):
        args = (indexed_txt_list, label_list)
    else:
        args = (indexed_txt_list,)

    guide = orig_lda.guide

    svi = pyro.infer.SVI(
        model=orig_lda.model,
        guide=guide,
        optim=Adam({"lr": ADAM_LEARN_RATE}),
        loss=orig_lda.loss)

    losses, alpha, beta = [], [], []
    num_step = 1000
    for step in range(num_step):
        loss = svi.step(*args)
        losses.append(loss)
        if isinstance(orig_lda, plainLDA):
            alpha.append(pyro.param("alpha_q"))
            beta.append(pyro.param("beta_q"))
        if step % 50 == 0:
            print("{}: {}".format(step, np.round(loss, 1)))

            # evaluate results
            dtype = [("word", "<U17"), ("index", int)]
            vocab = np.array(
                [item for item in vocab_dict.items()],
                dtype=dtype
            )
            vocab = np.sort(vocab, order="index")

            preds = []

            for i in range(100):
                if isinstance(orig_lda, supervisedLDA):
                    pred, mean, sigma = guide(indexed_txt_list, label_list)
                    pred = pred.data.numpy()
                else:
                    pred = guide(indexed_txt_list).data.numpy()
                preds.append(pred)

            posterior_topics_x_words = np.stack(preds).mean(0)
            np.savetxt(
                'posterior_topic_words.csv',
                posterior_topics_x_words,
                delimiter=','
            )
            with open(
                f"./outputs/posterior_topic_words_{step}.csv", "w"
            ) as output:
                for j in range(num_topic):
                    sorted_words_ix = np.argsort(
                        posterior_topics_x_words[j]
                    )[::-1]
                    # print("topic %s" % i)
                    # print([word[0] for word in vocab[sorted_words_ix][:10]])
                    output.write(
                        " ".join([w[0] for w in vocab[sorted_words_ix[:10]]])
                        + "\n"
                    )
