import os

import util
import torch
from torch import dist
import pandas as pd
import numpy as np

import matplotlib
from sklearn.preprocessing import MinMaxScaler

from models import regression_baseline

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