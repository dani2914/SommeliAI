""" main driver for meanfield LDA"""

import os
import torch
import data_util
import numpy as np

import pyro
from pyro.optim import Adam

from models import lda_meanfield


def main():
    # CONSTANTS
    ADAM_LEARN_RATE = 0.01
    TESTING_SUBSIZE = 0.02  # use None if want to use full dataset
    SUBSAMPLE_SIZE = 100
    NUM_OF_TOPICS = 10
    NUM_ITERATIONS = 10
    FILTER_CUSTOM_STOPWORDS = False
    RANDOM_STATE = 666
    USE_CUDA = False

    stop_words = [
        'acidity', 'age', 'apple', 'aroma', 'balance', 'berry',
        'blackberry', 'blend', 'cabernet', 'cherry', 'chocolate', 'citrus',
        'crisp', 'currant', 'dark', 'drink', 'dry', 'finish', 'flavor',
        'fresh', 'fruit', 'full', 'give', 'good', 'green', 'ha', 'herb',
        'hint', 'juicy', 'lemon', 'light', 'make', 'merlot', 'nose',
        'note', 'oak', 'offer', 'palate', 'peach', 'pepper', 'pinot',
        'plum', 'raspberry', 'red', 'rich', 'ripe', 'sauvignon', 'show',
        'soft', 'spice', 'structure', 'sweet', 'tannin', 'texture',
        'toast', 'vanilla', 'vineyard', 'well', 'wine', 'year', 'black'
    ]

    if USE_CUDA:
        torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        torch.set_default_tensor_type("torch.DoubleTensor")

    data_root_dir = os.path.join(".", "data")
    full_df = data_util.fetch_dataset(data_root_dir)

    # keep topics with the highest number of txt
    full_df = data_util.filter_by_topic(
        full_df,
        keep_top_n_topics=NUM_OF_TOPICS
    )

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.sample(
            frac=TESTING_SUBSIZE, replace=False, random_state=RANDOM_STATE)

    # if toggle doesn't ask for stop words, then set to None
    if not FILTER_CUSTOM_STOPWORDS:
        stop_words = None

    # remove stop words, punctuation, digits and then change to lower case
    clean_df, indexed_txt_list, vocab_dict, vocab_count = (
        data_util.preprocess_and_index(
            full_df,
            ngram=1,
            custom_stopwords=stop_words
        )
    )

    # txt_vec = clean_df["description"]
    topic_vec = clean_df["variety"]
    unique_topics = np.unique(topic_vec)

    num_topic = len(unique_topics)
    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    # evaluate results
    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")

    # create object of LDA class
    lda = lda_meanfield.meanfieldLDA(
        num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)

    args = (indexed_txt_list,)

    svi = pyro.infer.SVI(
        model=lda.model,
        guide=lda.guide,
        optim=Adam({"lr": ADAM_LEARN_RATE}),
        loss=lda.loss)

    print(f"Running {NUM_ITERATIONS} iterations...")
    for step in range(NUM_ITERATIONS):

        loss = svi.step(*args)
        print(f"{step}: {loss}")


if __name__ == "__main__":
    main()
