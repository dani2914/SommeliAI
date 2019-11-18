# %% Imports
import util
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors

# %% [markdown]
# Section 1.1
# Histogram of Topics (Variants of Wine)

# %%

full_df = util.fetch_dataset()
full_df = util.filter_by_topic(full_df, keep_top_n_topics=10)

ax = full_df.variety.value_counts().plot(kind="bar", figsize=(20,10), 
title="Document Counts by Topic")

ax.set_xlabel("Variety of Wines (TOPICS)")
ax.set_ylabel("Number of Documents")
plt.show()


# %%
