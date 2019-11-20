# %% Imports
from SommeliAI import data_util
import matplotlib.pyplot as plt
import os

# %% [markdown]
# Section 1.1
# Histogram of Topics (Variants of Wine)

# %%

data_root_dir = os.path.join("..", "data")
full_df = data_util.fetch_dataset(data_root_dir)
full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=10)

ax = full_df.variety.value_counts().plot(kind="bar", figsize=(20,10), 
title="Document Counts by Topic")

ax.set_xlabel("Variety of Wines (TOPICS)")
ax.set_ylabel("Number of Documents")
plt.show()


# %%
