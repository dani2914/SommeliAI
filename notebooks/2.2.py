# %% Imports

import os
import glob
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors

# %% [markdown]
# # Latent Dirichlet Allocation
# Vanilla LDA run with BBVI

# %%
data_root_dir = os.path.join(".", "data", "files")
pattern = "pyro_lda_beta_5000.csv"

file_path = glob.glob(os.path.join(data_root_dir, pattern))

beta_df = pd.read_csv(file_path[0], index_col=0)
# %%
beta_df.plot()
