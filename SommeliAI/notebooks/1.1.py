# %% Imports
from SommeliAI import data_util
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from SommeliAI.customised_stopword import customised_stopword
print(customised_stopword)
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

# %% [markdown]
# Section 1.1
# Histogram of Topics (Variants of Wine)

# %%

data_root_dir = os.path.join("..", "data")
full_df = data_util.fetch_dataset(data_root_dir)

clean_df = data_util.preprocess(full_df, preprocess=True)
print(clean_df.head)
clean_df, indexed_txt_list, vocab_dict, vocab_count = data_util.preprocess_and_index(clean_df, ngram=1,
                                                                                     custom_stopwords=customised_stopword)

value_counts = full_df.variety.value_counts()
tmp_colors = np.repeat(mycolors[0:2], np.array([10, value_counts.shape[0]-10]))

ax = full_df.variety.value_counts().plot(kind="bar", figsize=(20,10),
title="Document Counts by Topic", color=tmp_colors)
ax.set_xlabel("Variety of Wines (TOPIC)")
ax.set_ylabel("Number of Documents")
plt.show()
