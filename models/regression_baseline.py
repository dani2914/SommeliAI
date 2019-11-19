import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import numpy as np


class regression_baseline():
    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict

    def model(self, X, score_vec):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            score_vec.iloc[:, 0],
            test_size=0.2,
            random_state=3
        )

        rr = LassoCV(cv=5)
        rr.fit(X_train, y_train)
        print("alpha selected:", rr.alpha_)
        train_score = rr.score(X_train, y_train)
        test_score = rr.score(X_test, y_test)
        print("lasso regression train score:", train_score)
        print("lasso regression test score:", test_score)
        print("mse:", mean_squared_error(y_test, rr.predict(X_test)))
        return rr

    def plot(self, rr):
        matplotlib.rcParams.update({'font.size': 12})

        dtype = [("word", "<U17"), ("index", int)]

        vocab = np.array(
            [item for item in self.vocab_dict.items()],
            dtype=dtype
        )
        vocab = np.sort(vocab, order="index")
        word_index = np.argsort(np.abs(rr.coef_))[::-1]
        words = [word[0] for word in vocab[word_index][:10]]
        plt.plot(
            rr.coef_, alpha=0.7, linestyle='none',
            marker='*', markersize=5,
            color='red', zorder=7
        )  # zorder for ordering the markers
        plt.xlabel('Coefficient Index', fontsize=16)
        plt.ylabel('Coefficient Magnitude', fontsize=16)
        for i in range(len(words)):
            x = word_index[:10][i]
            y = rr.coef_[word_index][:10][i]
            plt.scatter(x, y, marker='x', color='blue')
            plt.text(x + .01, y + .01, words[i], fontsize=9)
        plt.title(
            "Lasso regression import words\n (alpha =" +
            str(np.round(rr.alpha_, 5)) + ")"
        )
        plt.show()
        plt.savefig("coefficients.png")
