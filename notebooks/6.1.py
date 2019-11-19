# %% Imports

import os
from customised_stopword import customised_stopword
import glob
import pandas as pd
import data_util
import numpy as np
import pyro
import pyro.distributions as dist
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt

# %% [markdown]
# # Supervised Latent Dirichlet Allocation

# %%

data_root_dir = os.path.join("..", "data")
TESTING_SUBSIZE = 0.02
full_df = data_util.fetch_dataset(data_root_dir)

# keep topics with the highest number of txt, and add min threshold if want
full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=10)

# if not none, then subset the dataframe for testing purposes
if TESTING_SUBSIZE is not None:
    full_df = full_df.sample(frac=TESTING_SUBSIZE, replace=False)

# remove stop words, punctuation, digits and then change to lower case
clean_df = data_util.preprocess(full_df, preprocess=True)
print(clean_df)
clean_df, indexed_txt_list, vocab_dict, vocab_count = data_util.preprocess_and_index(clean_df, ngram=1, custom_stopwords=customised_stopword)

# %%
scaler = MinMaxScaler()
score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))
label_list = score_vec.iloc[:, 0].tolist()


# %%
plt.hist(score_vec.iloc[:, 0])
plt.title("Historgram of scaled wine score")
plt.show()

