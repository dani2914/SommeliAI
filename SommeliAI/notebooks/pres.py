from SommeliAI import data_util
import matplotlib.pyplot as plt
import os
import SommeliAI.notebooks.util as util
import torch
from torch import dist
import pandas as pd
import numpy as np
import pickle
import matplotlib
from sklearn.preprocessing import MinMaxScaler

from SommeliAI.models import regression_baseline

matplotlib.rcParams.update({'font.size': 12})


def load_pyro_lda_1_theta_tsne(refresh=False):
    gamma_fname = "pyro_lda_1_gamma.csv"
    theta_tsne_fname = "pyro_lda_1_tsne.csv"

    gamma_path = util.get_filepath(gamma_fname)
    theta_tsne_path = util.get_filepath(theta_tsne_fname)

    theta_tsne_exist = os.path.exists(theta_tsne_path)

    if refresh or not theta_tsne_exist:
        gamma_df = pd.read_csv(gamma_path, index_col=0)
        theta_arr = dist.Dirichlet(torch.tensor(gamma_df.values)).sample()

        theta_df = pd.DataFrame(theta_arr.numpy(), index=gamma_df.index)

        theta_tsne_df = util.build_tsne(theta_df, theta_tsne_path)

    else:
        theta_tsne_df = util.load_tsne(theta_tsne_path)
    samp = torch.randint(0, theta_tsne_df.shape[0], (20000,))
    theta_tsne_df = theta_tsne_df.iloc[samp]

    return theta_tsne_df


def plot_pyro_lda_1_theta_tsne(refresh=False):
    theta_tsne_df = load_pyro_lda_1_theta_tsne()
    util.graph_tsne(theta_tsne_df)


def plot_pyro_lda_1_beta():
    beta_fname = "pyro_lda_1_beta.csv"
    beta_path = util.get_filepath(beta_fname)

    beta_df = pd.read_csv(beta_path, index_col=0)

    util.graph_word_dist(beta_df)


def plot_regression_features():
    TESTING_SUBSIZE = 0.02

    data_root_dir = os.path.join("..", "data")
    clean_df, data, vocab_dict = util.read_data(TESTING_SUBSIZE, data_root_dir)
    X = util.generate_matrix(data, len(vocab_dict))
    scaler = MinMaxScaler()
    score_vec = pd.DataFrame(scaler.fit_transform(
        np.vstack(
            clean_df['points'].values
        ).astype(np.float64))
    )
    rr = regression_baseline(vocab_dict)
    model = rr.model(X, score_vec)
    rr.plot(model)


def plot_filtered_variety_wine():
    data_root_dir = os.path.join("..", "data")
    full_df = data_util.fetch_dataset(data_root_dir)
    full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=10)

    ax = full_df.variety.value_counts().plot(kind="bar", figsize=(20, 10),
                                             title="Document Counts by Topic")

    ax.set_xlabel("Variety of Wines (TOPICS)")
    ax.set_ylabel("Number of Documents")
    plt.show()


def plot_regression_response_distribution():
    data_root_dir = os.path.join("..", "data")
    TESTING_SUBSIZE = 0.02

    clean_df, indexed_txt_list, vocab_dict = util.read_data(TESTING_SUBSIZE, data_root_dir)

    scaler = MinMaxScaler()
    score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))

    plt.hist(score_vec.iloc[:, 0])
    plt.title("Historgram of scaled wine score")
    plt.show()


def plot_slda_regression_topic_words():
    eta = np.load("files/pyro_slda_eta_5000.npy")
    lamb = np.load("files/pyro_slda_lambda_5000.npy")
    phi = np.load("files/pyro_slda_phi_5000.npy")

    num_topic = 10
    with open('files/trainset_slda_vocab.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)

    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")

    plt.figure(figsize=(12, 8))
    plt.plot(eta, alpha=0.7, linestyle='none', marker='*', markersize=5,
             color='red', zorder=7)  # zorder for ordering the markers
    plt.xlabel('Coefficient Index', fontsize=16)
    plt.ylabel('Coefficient Magnitude', fontsize=16)
    for i in range(num_topic):
        beta = np.random.dirichlet(lamb[i, :])

        word_index = np.argsort(beta)[::-1]
        words = "\n".join([word[0] for word in vocab[word_index][:10]])
        print(word_index[:20])
        print(eta)
        x = i
        y = eta[i]

        plt.scatter(x, y, marker='x', color='blue')
        plt.text(x + .3, y - .5, words, fontsize=10)

    plt.title("sLDA coefficient for each topic")
    plt.show()
