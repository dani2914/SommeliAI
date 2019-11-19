import matplotlib.pyplot as plt
import util
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

TESTING_SUBSIZE = 0.02


data_root_dir = os.path.join("..", "data")
clean_df, data, vocab_dict = util.read_data(TESTING_SUBSIZE, data_root_dir)
X = util.generate_matrix(data, len(vocab_dict))
scaler = MinMaxScaler()
score_vec = pd.DataFrame(scaler.fit_transform(np.vstack(clean_df['points'].values).astype(np.float64)))

X_train, X_test, y_train, y_test = train_test_split(X, score_vec.iloc[:, 0], test_size=0.2, random_state=3)
plt.hist(y_train)
plt.show()
rr = LassoCV(cv=5)
rr.fit(X_train, y_train)
print(rr.alpha_)
train_score = rr.score(X_train, y_train)
test_score = rr.score(X_test, y_test)
print("lasso regression train score:", train_score)
print("lasso regression test score:", test_score)
print("mse:", mean_squared_error(y_test, rr.predict(X_test)))
dtype = [("word", "<U17"), ("index", int)]

vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
vocab = np.sort(vocab, order="index")
word_index = np.argsort(np.abs(rr.coef_))[::-1]
print([word[0] for word in vocab[word_index][:20]])
words = [word[0] for word in vocab[word_index][:10]]
plt.plot(rr.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5,
             color='red', zorder=7)  # zorder for ordering the markers
plt.xlabel('Coefficient Index', fontsize=16)
plt.ylabel('Coefficient Magnitude', fontsize=16)
print(word_index[:20])
print(rr.coef_[word_index][:20])
for i in range(len(words)):
    x = word_index[:10][i]
    y = rr.coef_[word_index][:10][i]
    plt.scatter(x, y, marker='x', color='blue')
    plt.text(x + .01, y + .01, words[i], fontsize=9)
plt.title("Lasso regression import words (alpha =" + str(rr.alpha_) + ")")
plt.show()

plt.savefig("coefficients.png")

