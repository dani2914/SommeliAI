""" LDA implementation """


import argparse

import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import TraceEnum_ELBO

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


class original_lda:

    def __init__(self):
        return


    def model(self, doc_list=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
        """pyro model for lda"""

        # beta => prior for the per-topic word distributions
        beta_0 = torch.ones(num_vocabs) / num_vocabs

        # returns t x w matrix
        with pyro.plate("topics", num_topics):
            topics_x_words = pyro.sample("topics_x_words", dist.Dirichlet(beta_0))

            # alpha => prior for the per-document topic distribution
            alpha_0 = torch.ones(num_topics) / num_topics

        # returns d x t matrix 
        doc_x_words = []
        for i in pyro.plate("documents", num_docs):

            docs_x_topics = pyro.sample("docs_x_topics_{}".format(i),
                                        dist.Dirichlet(alpha_0))

            data = None if doc_list is None else doc_list[i]

            with pyro.plate("words_{}".format(i), num_words_per_doc_vec[i]):
                word_x_topics = pyro.sample("word_x_topics_{}".format(i),
                                            dist.Categorical(docs_x_topics),
                                            infer={"enumerate": "parallel"})

                words = pyro.sample("docs_x_words_{}".format(i),
                                    dist.Categorical(topics_x_words[word_x_topics]),
                                    obs=data)

            doc_x_words.append(words)

        return doc_x_words, topics_x_words


def guide(self, doc_list=None, num_docs=None, num_words_per_doc_vec=None,
                       num_topics=None, num_vocabs=None):
    """pyro guide for lda inference"""

    # beta_q => q for the per-topic word distribution
    beta_q = pyro.param("beta_q", torch.ones(num_vocabs), constraint=constraints.positive)

    with pyro.plate("topics", num_topics):
        pyro.sample("topics_x_words", dist.Dirichlet(beta_q))

        # alpha_q => q for the per-document topic distribution
        alpha_q = pyro.param("alpha_q", torch.ones(num_topics), constraint=constraints.positive)

    for i in pyro.plate("documents", num_docs):
        pyro.sample("docs_x_topics_{}".format(i), dist.Dirichlet(alpha_q))


def fetch_preprocess_sklearn_dataset(num_topics, num_docs):
    """ fetch dataset from sklearn, turn into vector of indexes """

    # get dataset first to get topic names / inefficient but who cares -- not costly
    tmp_train = fetch_20newsgroups(subset="train", shuffle=False)
    sub_topics = tmp_train.target_names[0:num_topics]

    # get the dataset for real
    newsgroups_train_dataset = fetch_20newsgroups(
        subset="train",
        categories=sub_topics,
        #remove=("headers", "footers", "quotes"),
        shuffle=True)

    # **if removing headers+footers+quotes, need to add filtering to remove empty datasets

    # keep only desired number of documents
    newsgroups_train_data = newsgroups_train_dataset.data[0:num_docs]


    # vectorize data and retrieve count vector, indexed vector, dict to return
    vectorizer = CountVectorizer()
    newsgroups_train_vector = vectorizer.fit_transform(newsgroups_train_data)

    count_vec = newsgroups_train_vector.toarray()
    doc_ix_list = [torch.tensor(np.repeat(range(count_vec.shape[1]), count_vec[i, :]))
                   for i in range(count_vec.shape[0])]
    vocab_dict = vectorizer.vocabulary_

    return doc_ix_list, vocab_dict


def fetch_n_preprocess_data():
    df_list = [pd.read_csv("winemag_dataset_%s.csv" % i) for i in range(1,7)]
    df = pd.concat(df_list)
    txt_vec = df["description"]

    # vectorize data and retrieve count vector, indexed vector, dict to return
    vectorizer = CountVectorizer()
    transformed_vec = vectorizer.fit_transform(txt_vec)

    doc_ix_list = []
    print("Running %s items..." % transformed_vec.shape[0])
    for i in range(transformed_vec.shape[0]):
        if i % 1000 == 0:
            print("Running item %s" % i)

        count_vec = transformed_vec[i].toarray()
        doc_ix_list.append(torch.tensor(np.repeat(range(count_vec.shape[1]), count_vec[0])))

    vocab_dict = vectorizer.vocabulary_

    return doc_ix_list, vocab_dict


def main():
    """ main function"""

    # constants
    num_topics = 5
    num_docs = 50
    adam_learn_rate = 0.01
    #use_cuda = False 
    test_doc_tf = False         #should we use a test set

    # if use_cuda:
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # pyro settings
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    # set svi params
    svi = pyro.infer.SVI(
        model=lda_model,
        guide=parametrized_guide,
        optim=Adam({"lr": adam_learn_rate}),
        loss=TraceEnum_ELBO(max_plate_nesting=2))#Trace_ELBO())

    # if flag for using test doc is yes, then use a toy test set
    if test_doc_tf:
        doc_list = [
            torch.tensor([1, 2, 3, 4, 5]),#.cuda(),
            torch.tensor([0, 2, 4, 6, 8, 9]),#.cuda(),
            torch.tensor([1, 3, 5, 7]),#.cuda(),
            torch.tensor([5, 6, 7])]#.cuda()]
        num_vocabs = len(np.unique([words for docs in doc_list for words in docs]))
        num_docs = len(doc_list)

    else:
        doc_list, vocabs_dict = fetch_preprocess_sklearn_dataset(num_topics, num_docs)
        num_vocabs = len(vocabs_dict)

    num_docs = torch.tensor(len(doc_list))#.cuda()
    num_words_per_doc_vec = torch.tensor([len(doc) for doc in doc_list])#.cuda()

    # run inference
    losses, alpha, beta = [], [], []
    step_count = 20
    for step in range(step_count):

        loss = svi.step(doc_list, num_docs, num_words_per_doc_vec, num_topics, num_vocabs)
        losses.append(loss)
        alpha.append(pyro.param("alpha_q"))
        beta.append(pyro.param("beta_q"))
        if step % 10 == 0:
            print("{}: {}".format(step, np.round(loss, 1)))

    # evalutate results
    if not test_doc_tf:
        print("in this loop")

        dtype = [("word", "<U17"), ("index", int)]
        vocab = np.array([item for item in vocabs_dict.items()], dtype=dtype)
        vocab = np.sort(vocab, order="index")

        posterior_doc_x_words, posterior_topics_x_words = \
                lda_model(doc_list, num_docs, num_words_per_doc_vec, num_topics, num_vocabs)


        for i in range(num_topics):
            non_trivial_word_ix = np.where(posterior_topics_x_words[i].cpu() > 0.01)[0]
            print("topic {}:".format(i))
            print(vocab[non_trivial_word_ix])




if __name__ == "__main__":

    main()
