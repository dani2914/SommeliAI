# %% [markdown]
# Word distribution of Vanilla LDA run with BBVI

# imports
import notebooks.util as util
import pandas as pd

# %%
beta_fname = "pyro_lda_1_beta.csv"
beta_path = util.get_filepath(beta_fname)

beta_df = pd.read_csv(beta_path, index_col=0)

# %%
util.graph_word_dist(beta_df)

