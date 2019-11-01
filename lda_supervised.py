
import argparse
import functools
import numpy as np
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import util
import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

def model(data=None, label=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
    # Globals.
    with pyro.plate("topics", num_topics):
        topic_words = pyro.sample("beta",
                                  dist.Dirichlet(torch.ones(num_vocabs) / num_vocabs))

    # returns d x t matrix
    doc_x_words = []
    # Locals.
    with pyro.plate("labels", num_topics):
        label_topics = pyro.sample("theta",
                                  dist.Dirichlet(torch.ones(num_topics) / num_topics))

    for i in pyro.plate("documents", num_docs):
        words = data[i]
        doc_topics = label_topics[label[i]]
        with pyro.plate("words_{}".format(i), num_words_per_doc_vec[i]):
            word_topics = pyro.sample("z_{}".format(i), dist.Categorical(doc_topics),
                                          infer={"enumerate": "parallel"})
            words = pyro.sample("docs_x_words_{}".format(i), dist.Categorical(topic_words[word_topics]),
                                   obs=words)

        doc_x_words.append(words)


    return topic_words, doc_x_words


# We will use amortized inference of the local topic variables, achieved by a
# multi-layer perceptron. We'll wrap the guide in an nn.Module.
def make_predictor(num_words, num_topics, args):
    layer_sizes = ([num_words] +
                   [int(s) for s in args.layer_sizes.split('-')] +
                   [num_topics])
    logging.info('Creating MLP with sizes {}'.format(layer_sizes))
    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layer = nn.Linear(in_size, out_size)
        layer.weight.data.normal_(0, 0.001)
        layer.bias.data.normal_(0, 0.001)
        layers.append(layer)
        layers.append(nn.Sigmoid())
    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


def guide(data=None, label=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
    # beta_q => q for the per-topic word distribution
    beta_q = pyro.param("beta_q", torch.ones(num_vocabs), constraint=constraints.positive)
    eta_q = pyro.param("eta_q", torch.ones(num_topics), constraint=constraints.positive)

    with pyro.plate("topics", num_topics):
        pyro.sample("beta", dist.Dirichlet(beta_q))

    with pyro.plate("labels", num_topics):
        pyro.sample("theta", dist.Dirichlet(eta_q))


def main():
    TESTING_SUBSIZE = 50
    learning_rate = 0.01
    num_steps = 1000
    logging.info('Generating data')
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    wine1 = pd.read_csv('winemag_dataset_0.csv')
    clean_df = util.preprocess(wine1, preprocess=True)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        clean_df = clean_df.head(TESTING_SUBSIZE)

    indexed_txt_list, vocab_dict = util.conv_word_to_indexed_txt(clean_df['description'])

    #true_topic_weights, true_topic_words, data = model(indexed_txt_list, args=args)

    topic_vec = clean_df["variety"]
    unique_topics = np.unique(topic_vec)
    topic_map = {unique_topics[i]:i for i in range(len(unique_topics))}

    clean_df.loc[:, "class"] = clean_df["variety"].apply(lambda row: topic_map[row])

    num_topic = len(unique_topics)
    num_vocab = len(vocab_dict)
    num_txt = len(indexed_txt_list)
    num_words_per_txt = [len(txt) for txt in indexed_txt_list]

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(num_txt))

    optim = ClippedAdam({'lr': learning_rate})
    svi = SVI(model, guide, optim, TraceEnum_ELBO(max_plate_nesting=2))
    logging.info('Step\tLoss')
    for step in range(num_steps):
        loss = svi.step(indexed_txt_list, clean_df.loc[:, "class"].tolist(), num_txt, num_words_per_txt,
                          num_topic, num_vocab)
        if step % 100 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    # loss = elbo.loss(model, guide, data, args=args)
    # logging.info('final loss = {}'.format(loss))

    # evaluate results
    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")

    posterior_topics_x_words, posterior_doc_x_words =  \
        model(indexed_txt_list, clean_df.loc[:, "class"].tolist(),
              num_txt, num_words_per_txt,
                           num_topic, num_vocab)

    for i in range(num_topic):
        non_trivial_words_ix = np.where(posterior_topics_x_words[i] > 0.01)[0]
        print("topic %s" % i)
        print([word[0] for word in vocab[non_trivial_words_ix]])


if __name__ == '__main__':
    main()
