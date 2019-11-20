""" hand written cavi for viewing latent variable behavior"""

import os
import torch
import data_util
import numpy as np
import pandas as pd

def main():
    # CONSTANTS
    ADAM_LEARN_RATE = 0.01
    TESTING_SUBSIZE = 0.001 #use None if want to use full dataset
    SUBSAMPLE_SIZE = 100
    NUM_OF_TOPICS = 10
    NUM_ITERATIONS = 10
    FILTER_CUSTOM_STOPWORDS = False
    RANDOM_STATE = 666
    USE_CUDA = False

    stop_words = ['acidity', 'age', 'apple', 'aroma', 'balance', 'berry', 
      'blackberry', 'blend', 'cabernet', 'cherry', 'chocolate', 'citrus',
      'crisp', 'currant', 'dark', 'drink', 'dry', 'finish', 'flavor',
      'fresh', 'fruit', 'full', 'give', 'good', 'green', 'ha', 'herb',
      'hint', 'juicy', 'lemon', 'light', 'make', 'merlot', 'nose',
      'note', 'oak', 'offer', 'palate', 'peach', 'pepper', 'pinot',
      'plum', 'raspberry', 'red', 'rich', 'ripe', 'sauvignon', 'show',
      'soft', 'spice', 'structure', 'sweet', 'tannin', 'texture',
      'toast', 'vanilla', 'vineyard', 'well', 'wine', 'year', 'black']

    if USE_CUDA:
        torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        torch.set_default_tensor_type("torch.DoubleTensor")

    data_root_dir = os.path.join(".", "data")
    full_df = data_util.fetch_dataset(data_root_dir)

    # keep topics with the highest number of txt
    full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=NUM_OF_TOPICS)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.sample(
            frac=TESTING_SUBSIZE, replace=False, random_state=RANDOM_STATE)

    # if toggle doesn't ask for stop words, then set to None
    if not FILTER_CUSTOM_STOPWORDS:
        stop_words = None

    # remove stop words, punctuation, digits and then change to lower case
    clean_df, indexed_txt_list, vocab_dict, vocab_count = \
        data_util.preprocess_and_index(full_df, ngram=1, 
        custom_stopwords=stop_words)

    txt_vec = clean_df["description"]
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

    doc_list = indexed_txt_list

    doc_list = [doc.to(dtype=torch.int) for doc in doc_list]


    # actual cavi model
    K = num_topic
    V = num_vocab
    D = num_txt
    W = num_words_per_txt

    eta_0 = 1+0.01*(2*torch.rand(K, V)-1)
    alpha_0 = torch.ones(D, K)/K
#
    eta = eta_0
    alpha = alpha_0

    print(f"Running {NUM_ITERATIONS} iterations...")
    for step in range(NUM_ITERATIONS):
        print(f"Iteration: {step}")

        phi = [torch.empty(len(doc), K) for doc in doc_list]

        # update phi (where phi = beta[z_assignment])
        for d in torch.arange(D):
            for i in torch.arange(W[d]):
                x_di = doc_list[d][i]

                phi_numerator = torch.digamma(eta[:, x_di]) - \
                        torch.digamma(torch.sum(eta, axis=1)) + \
                        torch.digamma(alpha[d, :]) - \
                        torch.digamma(torch.sum(alpha[d]))
                phi_numerator = torch.exp(phi_numerator)

                phi[d][i] = phi_numerator / torch.sum(phi_numerator)

        # update alpha
        for d in torch.arange(D):
            alpha[d] = alpha_0[d] + torch.sum(phi[d], axis=0)


        # update eta
        for v in torch.arange(V, dtype=torch.int):
            v_match = [doc==v for doc in doc_list]

            phi_sub = [phi[d][v_match[d]] for d in torch.arange(d) \
                    if len(phi[d][v_match[d]]) != 0]

            if phi_sub == []:
                eta[:,v] = eta_0[:,v]
            else:
                phi_sub = torch.sum(torch.cat(
                    [phi[d][v_match[d]] for d in torch.arange(d) \
                        if len(phi[d][v_match[d]]) != 0]), axis=0)
                eta[:,v] = eta_0[:,v] + phi_sub


        phi_df = pd.DataFrame([torch.mean(p, axis=0).cpu().numpy() for p in phi])
        extremes_df = pd.concat((phi_df.min(axis=0), phi_df.max(axis=0)), axis=1)
        extremes_df.columns = ["min", "max"]

        print(extremes_df)

if __name__ == "__main__":
    main()