""" main driver """
import time
import os
import data_util
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
import torch
from sklearn.model_selection import train_test_split
import pyro
from pyro.optim import Adam
from customised_stopword import customised_stopword
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

print(mp.cpu_count())
import pyro.poutine as poutine

import warnings

warnings.filterwarnings("ignore")

from models import supervisedLDA


def main():
    ADAM_LEARN_RATE = 1e-3
    TESTING_SUBSIZE = 100  # use None if want to use full dataset
    SUBSAMPLE_SIZE = 100
    USE_CUDA = False
    num_topic = 10
    RUN_NAME = "sLDA_" + str(TESTING_SUBSIZE) + "_" + str(SUBSAMPLE_SIZE) + "_" + str(
        num_topic) + "_enumerated"  # 1/k 1/k error 1/v 1/k error
    print(RUN_NAME)

    # # #
    if USE_CUDA:
        torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        torch.set_default_tensor_type("torch.DoubleTensor")

    data_root_dir = os.path.join(".", "data")
    full_df = data_util.fetch_dataset(data_root_dir)

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=10)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.sample(n=TESTING_SUBSIZE, replace=False, random_state=666)

    # remove stop words, punctuation, digits and then change to lower case
    clean_df = data_util.preprocess(full_df, preprocess=True)
    print(clean_df)
    clean_df, indexed_txt_list, vocab_dict, vocab_count = data_util.preprocess_and_index(clean_df, ngram=1,
                                                                                    custom_stopwords=customised_stopword)
    topic_vec = clean_df["variety"]
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab_dict, f)

    scaler = MinMaxScaler()
    score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))
    plt.hist(score_vec.iloc[:, 0])
    plt.title("Historgram of scaled wine score")
    plt.show()
    np.save("SommeliAI_labels.npy", np.array(score_vec))
    unique_topics = np.unique(topic_vec)

    topic_map = {unique_topics[i]: i + 1 for i in range(len(unique_topics))}
    clean_df.loc[:, "class"] = clean_df["variety"].apply(lambda row: topic_map[row])
    label_list = score_vec.iloc[:, 0].tolist()

    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]
    print(num_vocab)
    # create object of LDA class
    orig_lda = supervisedLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)

    args = (indexed_txt_list, label_list)

    guide = orig_lda.guide

    svi = pyro.infer.SVI(
        model=orig_lda.model,
        guide=guide,
        optim=Adam({"lr": ADAM_LEARN_RATE, "betas": (0.90, 0.999)}),
        loss=orig_lda.loss)

    losses, alpha, beta = [], [], []
    num_step = 8001
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
            lamb, lamb_ix = [], []
            for i in range(num_txt):
                try:
                    tensor = pyro.param("alpha_q_" + str(i))
                    alpha.append(tensor.detach().numpy())
                    alpha_ix.append(i)
                    temp = []
                    for j in range(len(indexed_txt_list[i])):
                        tensor = pyro.param("phi_q_" + str(i) + "_" + str(j))
                        temp.append(tensor.detach().numpy())
                    phi_ix.append(i)
                    phi.append(temp)

                    lamb.append(pyro.param("lamda_q_" + str(i)))
                    lamb_ix.append(i)

                except:
                    pass
            print("covered doc:" + str(len(alpha_ix)))
            dtype = [("word", "<U17"), ("index", int)]
            vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
            vocab = np.sort(vocab, order="index")
            tr = poutine.trace(guide).get_trace(*args)
            for name, site in tr.nodes.items():
                if name == "eta":
                    eta = site["value"]
                if "lamda" in name:
                    print(name)
                if "phi" in name:
                    print(name)
                if name == "beta":
                    beta = site["value"]
            y_label = []
            y_gold = []
            X = []
            for ix in range(len(alpha_ix)):
                phi_p = phi[ix]
                phi_in = np.mean(np.array(phi_p), axis=0)
                X.append(phi_in)
                predicted_label = np.dot(eta.detach().numpy(), phi_in)
                y_label.append(predicted_label)

                y_gold.append(label_list[alpha_ix[ix]])
            print(X)
            print(eta.detach().numpy())
            X_train, X_test, y_train, y_test = \
                train_test_split(np.array(X), np.array(y_gold), test_size=0.2,
                                 random_state=3)
            rr = LassoCV(cv=5, random_state=3)
            rr.fit(X_train, y_train)
            train_score = rr.score(X_train, y_train)
            test_score = rr.score(X_test, y_test)
            print("lasso regression on lda: " + str(train_score) + str(test_score))
            output.write("lasso regression on lda: " + str(train_score) + " " + str(test_score))
            print("score loss:", r2_score(np.array(y_gold), np.array(y_label)))
            output.write("score loss:" + str(r2_score(np.array(y_gold), np.array(y_label))) + '\n')
            score_loss.append(r2_score(np.array(y_gold), np.array(y_label)))
            np.save("phi_" + str(step), np.array(X))
            np.save("phi_ix" + str(step), np.array(phi_ix))
            np.save("alpha_ix" + str(step), np.array(alpha_ix))
            np.save("eta_" + str(step), eta.detach().numpy())
            np.save("lambda_" + str(step), np.array(lamb))
            np.save("alpha_" + str(step), np.array(alpha))
            print(np.array(lamb).shape)
            posterior_topics_x_words = beta.data.numpy()
            print(posterior_topics_x_words)
            for i in range(num_topic):
                sorted_words_ix = np.argsort(posterior_topics_x_words[i])[::-1]
                print("topic %s" % i)
                print([word[0] for word in vocab[sorted_words_ix][:10]])
                output.write("topic %s" % i + '\n')
                output.write(" ".join([word[0] for word in vocab[sorted_words_ix]][:10]) + '\n')
    fig = plt.figure()
    plt.plot(losses)
    fig.savefig("trainig_loss" + RUN_NAME + ".png")

    fig = plt.figure()
    plt.plot(score_loss)
    fig.savefig("score_loss" + RUN_NAME + ".png")


if __name__ == '__main__':
    main()
