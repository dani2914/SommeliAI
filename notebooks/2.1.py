
# %% [markdown]
# # TSNE of Vanilla LDA run with BBVI

# #
# imports
import notebooks.util as util
import pandas as pd
import pyro.distributions as dist
import torch

# mechanism to load tsne
def load_pyro_lda_1_tsne(refresh=False):

    gamma_fname = "pyro_lda_1_gamma.csv"
    tsne_fname = "pyro_lda_1_tsne.csv"

    gamma_path = util.get_filepath(gamma_fname)
    tsne_path = util.get_filepath(tsne_fname)

    if refresh:
        gamma_df = pd.read_csv(gamma_path, index_col=0)
        theta_arr = dist.Dirichlet(torch.tensor(gamma_df.values)).sample()

        theta_df = pd.DataFrame(theta_arr.numpy(), index=gamma_df.index)

        tsne_df = util.build_tsne(theta_df, tsne_path)

    else:
        tsne_df = util.load_tsne(tsne_path)

    return tsne_df

# %%

tsne_df = load_pyro_lda_1_tsne()
util.graph_tsne(tsne_df)

