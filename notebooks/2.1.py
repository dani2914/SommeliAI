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
pattern = "pyro_lda_gamma_5000.csv"

file_path = glob.glob(os.path.join(data_root_dir, pattern))

gamma_df = pd.read_csv(file_path[0], index_col=0)
gamma_arr = gamma_df.values

theta_arr = dist.Dirichlet(torch.tensor(gamma_arr)).sample()


# %%
# Keep the well separated points (optional)
#theta_arr = theta_arr[torch.max(theta_arr, axis=1).values > 0.9]
#theta_arr = theta_arr[torch.randint(0, theta_arr.shape[0], (20000,))]

# highest prob topic for each doc [D x 1]
topic_ix = torch.argmax(theta_arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(theta_arr)

# %%

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 10
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_ix])
show(plot)

# %%
