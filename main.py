""" main driver """
import importlib
import util
import lda
import numpy as np
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import TraceEnum_ELBO


pyro.set_rng_seed(0)
pyro.clear_param_store()
pyro.enable_validation(True)

def main():
    """ main function """

    # CONSTANTS
    ADAM_LEARN_RATE = 0.01
    TESTING_SUBSIZE = 50 #use None if want to use full dataset

    full_df = util.fetch_dataset()

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = util.filter_by_topic(full_df, keep_top_n_topics=100)

    # if not none, then subset the dataframe for testing purposes
    if(TESTING_SUBSIZE is not None):
        full_df = full_df.head(TESTING_SUBSIZE)

    # remove stop words, punctuation, digits and then change to lower case
    clean_df = util.preprocess(full_df, preprocess=True)

    txt_vec = clean_df["description"]
    topic_vec = clean_df["variety"]
    unique_topics = np.unique(topic_vec)

    indexed_txt_list, vocab_dict = util.conv_word_to_indexed_txt(txt_vec)

    num_topic = len(unique_topics)
    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

#    # purely for testing purposes, overrides original val with toy dataset
#    indexed_txt_list = [
#        torch.tensor([1, 2, 3, 4, 5]),
#        torch.tensor([0, 2, 4, 6, 8, 9]),
#        torch.tensor([1, 3, 5, 7]),
#        torch.tensor([5, 6, 7])]
#    num_topic = 3
#    num_vocab = len(np.unique([word for txt in indexed_txt_list for word in txt]))
#    num_txt = len(indexed_txt_list)
#    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    # create object of LDA class
    orig_lda = lda.origLDA()

    svi = pyro.infer.SVI(
        model=orig_lda.model,
        guide=orig_lda.guide,
        optim=Adam({"lr": ADAM_LEARN_RATE}),
        loss=TraceEnum_ELBO(max_plate_nesting=2))

    losses, alpha, beta = [], [], []
    num_step = 100
    for step in range(num_step):

        loss = svi.step(indexed_txt_list, num_txt, num_words_per_txt,
                          num_topic, num_vocab)
        losses.append(loss)
        alpha.append(pyro.param("alpha_q"))
        beta.append(pyro.param("beta_q"))
        if step % 10 == 0:
            print("{}: {}".format(step, np.round(loss, 1)))

    # evaluate results
    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")

    posterior_doc_x_words, posterior_topics_x_words = \
            orig_lda.model(indexed_txt_list, num_txt, num_words_per_txt,
                           num_topic, num_vocab)

    for i in range(num_topic):
        non_trivial_words_ix = np.where(posterior_topics_x_words[i] > 0.01)[0]
        print("topic %s" % i)
        print([word[0] for word in vocab[non_trivial_words_ix]])

if __name__ == "__main__":

    main()
