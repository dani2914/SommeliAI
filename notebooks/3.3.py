import numpy as np
import matplotlib as plt
import notebooks.util as util
import os

# %%
num_topic = 10
eta = np.load("files/pyro_slda_eta_5000.npy")
lamb = np.load("files/pyro_slda_lambda_5000.npy")
phi = np.load("files/pyro_slda_phi_5000.npy")

# %%

dtype = [("word", "<U17"), ("index", int)]
TESTING_SUBSIZE = 0.02

data_root_dir = os.path.join("..", "data")
clean_df, data, vocab_dict = util.read_data(TESTING_SUBSIZE, data_root_dir)
vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
vocab = np.sort(vocab, order="index")

for i in range(num_topic):
    beta = np.random.dirichlet(lamb[i, :])
    word_index = np.argsort(beta)[::-1]
    words = [word[0] for word in vocab[word_index][:10]]
    plt.plot(eta, alpha=0.7, linestyle='none', marker='*', markersize=5,
             color='red', zorder=7)  # zorder for ordering the markers
    plt.xlabel('Coefficient Index', fontsize=16)
    plt.ylabel('Coefficient Magnitude', fontsize=16)
    print(word_index[:20])
    print(eta[word_index][:20])
    for i in range(len(words)):
        x = word_index[:10][i]
        y = eta[word_index][:10][i]
        plt.scatter(x, y, marker='x', color='blue')
        plt.text(x + .01, y + .01, words[i], fontsize=9)
    plt.title("sLDA for topic" + str(i))
    plt.show()
    plt.savefig("sLDA_coefficients.png")
