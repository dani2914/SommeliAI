import time
import numpy as np
from pyro.infer.mcmc import NUTS
import pyro.poutine as poutine
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pyro
from pyro.optim import Adam
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt


# %% initialize model sLDA
ADAM_LEARN_RATE = 0.01
num_topic = 10

num_vocab = len(vocab_dict)
num_txt = len(indexed_txt_list)
num_words_per_txt = [len(txt) for txt in indexed_txt_list]
print("vocabulary size:", num_vocab)
# create object of LDA class
orig_lda = supervisedLDA(num_txt, num_words_per_txt, num_topic, num_vocab, SUBSAMPLE_SIZE)
args = (indexed_txt_list, label_list)
guide = orig_lda.guide

svi = pyro.infer.SVI(
            model=orig_lda.model,
            guide=guide,
            optim=Adam({"lr": ADAM_LEARN_RATE, "betas": (0.90, 0.999)}),
            loss=orig_lda.loss)

# %% run sLDA

losses, alpha, beta = [], [], []
num_step = 100
s1 = time.time()
output = open("output" + str(s1) + ".txt", "a")
loss_f = open("loss.txt", "a")
score_loss = []
for step in range(num_step):
    loss = svi.step(*args)
    losses.append(loss)
    if step % 10 == 0:
        print("{}: {}".format(step, np.round(loss, 1)))
        output.write("{}: {}".format(step, np.round(loss, 1)) + '\n')
        loss_f.write(str(np.round(loss, 1)) + '\n')
    if step % 10 == 0:
        alpha, alpha_ix = [], []
        phi, phi_ix = [], []
        # and np.isnan(site["value"].detach().numpy()).any():
        for i in range(num_txt):
            try:
                tensor = pyro.param(f"alpha_q_{i}")
                alpha.append(tensor.detach().numpy())
                alpa_ix.append(i)
                # phi_ix.append(i)
                # phi.append(pyro.param(f"theta_{i}"))
                temp = []
                for j in range(len(indexed_txt_list[i])):
                    tensor = pyro.param(f"phi_q_{i}_{j}")
                    temp.append(tensor.detach().numpy())
                phi_ix.append(i)
                phi.append(temp)
            except:
                pass
        print("covered doc:" + str(len(alpha_ix)))
        dtype = [("word", "<U17"), ("index", int)]
        vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
        vocab = np.sort(vocab, order="index")
        tr = poutine.trace(guide).get_trace(*args)
        for name, site in tr.nodes.items():
            if name == "eta":
                eta = site["value"]
            if name == "beta":
                beta = site["value"]
        y_label = []
        y_gold = []
        X = []
        for ix in range(len(alpha_ix)):
            phi_p = phi[ix]
            phi_in = np.mean(np.array(phi_p), axis=0)
            X.append(phi_in)
            print(phi_in)
            predicted_label = np.dot(eta.detach().numpy(), phi_in)
            y_label.append(predicted_label)

            y_gold.append(label_list[ix])
        print(X)
        X_train, X_test, y_train, y_test = \
                        train_test_split(np.array(X), np.array(y_gold), test_size=0.2,
                                                                        random_state=3)
        rr = LassoCV(cv=5, random_state=3)
        rr.fit(X_train, y_train)
        train_score = rr.score(X_train, y_train)
        test_score = rr.score(X_test, y_test)
        print("lasso regression on lda: " + str(train_score) + str(test_score))
        output.write("lasso regression on lda: " + str(train_score) + " " + str(test_score))
        print("score loss:", r2_score(np.array(y_gold), np.array(y_label)))
        output.write("score loss:" + str(r2_score(np.array(y_gold), np.array(y_label))) + '\n')
        score_loss.append(r2_score(np.array(y_gold), np.array(y_label)))
        np.save("phi_" + str(step), np.array(X))
        posterior_topics_x_words = beta.data.numpy()
        print(posterior_topics_x_words)
        for i in range(num_topic):
            sorted_words_ix = np.argsort(posterior_topics_x_words[i])[::-1]
            print("topic %s" % i)
            print([word[0] for word in vocab[sorted_words_ix][:10]])
            output.write("topic %s" % i + '\n')
            output.write(" ".join([word[0] for word in vocab[sorted_words_ix]][:10]) + '\n')
fig = plt.figure()
plt.plot(losses)
plt.show(title="training loss sLDA")

fig = plt.figure()
plt.plot(score_loss)
plt.show(title="score loss")
