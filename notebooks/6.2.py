import util
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from models import regression_baseline
import matplotlib.pyplot as plt


TESTING_SUBSIZE = 0.02


data_root_dir = os.path.join("..", "data")
clean_df, data, vocab_dict = util.read_data(TESTING_SUBSIZE, data_root_dir)
X = util.generate_matrix(data, len(vocab_dict))
scaler = MinMaxScaler()
score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))
plt.hist(score_vec)
plt.show()
rr = regression_baseline(vocab_dict)
model = rr.model(X, score_vec)
rr.plot(model)