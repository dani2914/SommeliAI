
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

def model(data=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
    # Globals.
    with pyro.plate("topics", num_topics):
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / num_topics, 1.))
        topic_words = pyro.sample("topic_words",
                                  dist.Dirichlet(torch.ones(num_vocabs) / num_vocabs))

    # returns d x t matrix
    doc_x_words = []
    # Locals.
    for i in pyro.plate("documents", num_docs):
        words = data[i]
        doc_topics = pyro.sample("doc_topics_{}".format(i), dist.Dirichlet(topic_weights))
        with pyro.plate("words_{}".format(i), num_words_per_doc_vec[i]):
            word_topics = pyro.sample("word_topics_{}".format(i), dist.Categorical(doc_topics),
                                          infer={"enumerate": "parallel"})
            words = pyro.sample("doc_words_{}".format(i), dist.Categorical(topic_words[word_topics]),
                                   obs=words)

        doc_x_words.append(words)

    return topic_weights, topic_words, doc_x_words


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


def parametrized_guide(predictor, data=None, num_docs=None, num_words_per_doc_vec=None,
              num_topics=None, num_vocabs=None):
    topic_weights_posterior = pyro.param(
            "topic_weights_posterior",
            lambda: torch.ones(num_topics),
            constraint=constraints.positive)
    topic_words_posterior = pyro.param(
            "topic_words_posterior",
            lambda: torch.ones(num_topics, num_vocabs),
            constraint=constraints.positive)
    with pyro.plate("topics", num_topics):
        pyro.sample("topic_weights", dist.Gamma(topic_weights_posterior, 1.))
        pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

    pyro.module("predictor", predictor)
    for i in pyro.plate("documents", num_docs):
        words = data[i]
        # The neural network will operate on histograms rather than word
        # index vectors, so we'll convert the raw data to a histogram.
        counts = (torch.zeros(num_vocabs)
                       .scatter_add(0, words, torch.ones(words.shape[0])))
        doc_topics = predictor(counts)
        pyro.sample("doc_topics_{}".format(i), dist.Delta(doc_topics, event_dim=1))


def main(args):

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(num_txt))
    predictor = make_predictor(num_vocab, num_topic, args)
    guide = functools.partial(parametrized_guide, predictor)
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = TraceEnum_ELBO(max_plate_nesting=2)
    optim = ClippedAdam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(indexed_txt_list, num_txt, num_words_per_txt,
                          num_topic, num_vocab)
        if step % 100 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    # loss = elbo.loss(model, guide, data, args=args)
    # logging.info('final loss = {}'.format(loss))

    # evaluate results
    dtype = [("word", "<U17"), ("index", int)]
    vocab = np.array([item for item in vocab_dict.items()], dtype=dtype)
    vocab = np.sort(vocab, order="index")

    posterior_topic_weights, posterior_topics_x_words, posterior_doc_x_words =  \
        model(indexed_txt_list, num_txt, num_words_per_txt,
                           num_topic, num_vocab)

    for i in range(num_topic):
        non_trivial_words_ix = np.where(posterior_topics_x_words[i] > 0.01)[0]
        print("topic %s" % i)
        print([word[0] for word in vocab[non_trivial_words_ix]])


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.4.1')
    main(args)
