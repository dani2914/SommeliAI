import os

import util
import torch
from torch import dist
import pandas as pd

# from util import *


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
