import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# %%
phi = np.load("files/pyro_slda_full_phi_5000.npy")
eta = np.load("files/pyro_slda_full_eta_5000.npy")
phi_ix = np.load("files/pyro_slda_full_phi_ix5000.npy")
labels = np.load("files/SommeliAI_labels.npy")[phi_ix]
TESTING_SUBSIZE = None

select = np.random.choice(np.arange(labels.shape[0]), size=1000)

print(labels[select].shape)
print(phi[select].shape)
r2 = r2_score(labels[select], np.dot(phi[select], eta))
plt.plot(labels[select])
plt.plot(np.dot(phi[select], eta))
plt.title("training result on sLDA regression with\n r2 score "
          + str(r2_score(labels[select], np.dot(phi[select], eta))))
plt.show()