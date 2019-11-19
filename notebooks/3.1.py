# %% Imports
import notebooks.util as util
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# %%

data_root_dir = os.path.join("..", "data")
TESTING_SUBSIZE = 0.02

clean_df, indexed_txt_list, vocab_dict = util.read_data(TESTING_SUBSIZE, data_root_dir)

# %%
scaler = MinMaxScaler()
score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))
label_list = score_vec.iloc[:, 0].tolist()


# %%
plt.hist(score_vec.iloc[:, 0])
plt.title("Historgram of scaled wine score")
plt.show()

