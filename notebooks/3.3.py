import numpy as np
import matplotlib.pyplot as plt
import pickle

# %%
num_topic = 10
eta = np.load("files/pyro_slda_full_eta_5000.npy")
lamb = np.load("files/pyro_slda_full_lambda_5000.npy")
phi = np.load("files/pyro_slda_full_phi_5000.npy")

# %%
num_topic = 10
with open('files/SommeliAI_vocab_dict.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

dtype = [("word", "<U17"), ("index", int)]
vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
vocab = np.sort(vocab, order="index")

plt.figure(figsize=(12, 8))
plt.plot(eta, alpha=0.7, linestyle='none', marker='*', markersize=5,
             color='red', zorder=7)  # zorder for ordering the markers
plt.xlabel('Coefficient Index', fontsize=16)
plt.ylabel('Coefficient Magnitude', fontsize=16)
for i in range(num_topic):
    beta = np.random.dirichlet(lamb[i, :])

    word_index = np.argsort(beta)[::-1]
    words = "\n".join([word[0] for word in vocab[word_index][:10]])
    print(word_index[:20])
    print(eta)
    x = i
    y = eta[i]

    plt.scatter(x, y, marker='x', color='blue')
    plt.text(x + .3, y - .5, words, fontsize=10)

plt.title("sLDA coefficient for each topic")
plt.show()
plt.savefig("sLDA_coefficients.png")