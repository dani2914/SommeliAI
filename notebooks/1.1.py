# %% Imports
import data_util
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

mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

# %% [markdown]
# Section 1.1
# Histogram of Topics (Variants of Wine)

# %%

full_df = data_util.fetch_dataset()

value_counts = full_df.variety.value_counts()
tmp_colors = np.repeat(mycolors[0:2], np.array([10, value_counts.shape[0]-10]))

ax = full_df.variety.value_counts().plot(kind="bar", figsize=(20,10),
title="Document Counts by Topic", color=tmp_colors)
ax.set_xlabel("Variety of Wines (TOPIC)")
ax.set_ylabel("Number of Documents")
plt.show()
