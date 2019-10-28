""" utility functions """

import pandas as pd
import numpy as np
import torch
import re
import string

from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

stop_n_punct_words = set(stopwords.words("english") + list(string.punctuation))


def fetch_dataset():
    """ fetch data from disk and return a dataframe """

    df_list = [pd.read_csv("winemag_dataset_%s.csv" % i) for i in range(6)]
    full_df = pd.concat(df_list)

    # give unique row names to all
    full_df.index = range(full_df.shape[0])

    print("Dataset fetched.")
    return full_df


def filter_by_topic(tmp_df, keep_top_n_topics=0, min_count_threshold=0):
    """ filter based on topic information """

    topic_count_df = tmp_df["variety"].value_counts()

    # filter by top n number of topics if specified
    if keep_top_n_topics is not None:
        topics_to_keep = topic_count_df.head(keep_top_n_topics).index
        tmp_df = tmp_df[tmp_df["variety"].isin(topics_to_keep)]

    # filter out any topics that doesn't meet the minimum count threshold
    if min_count_threshold >= 0:
        topics_to_keep = topic_count_df[topic_count_df > min_count_threshold].index
        tmp_df = tmp_df[tmp_df["variety"].isin(topics_to_keep)]

    return tmp_df


def clean_stop_punct_digit_n_lower(txt):

    token = word_tokenize(txt)
    clean_token = [word.lower() for word in token if word.lower()
                   not in stop_n_punct_words and not word.isdigit()]

    return " ".join(clean_token)


def preprocess(tmp_df, preprocess=False):
    """ removing stop words, punctuations, digits and make all lower case """

    # all in one go in order to just have to tokenize once
    if preprocess:
        tmp_df["description"] = tmp_df["description"].apply(
            clean_stop_punct_digit_n_lower)

    print("Dataset cleaned.")
    return tmp_df


def conv_word_to_indexed_txt(txt_vec):
    """ change txt into int indexes, return with dict of indexes & count of each word """

    # transform words into integer indexes, comes out as n x m
    # where n = # txt doc, m = # unique words for whole universe
    vectorizer = CountVectorizer()
    sparse_count_vec = vectorizer.fit_transform(txt_vec)

    # create n x p list of words represented by ints,  where p = # words in each documentx
    # written in such a convoluted way for speed optimization purposes 
    x_vec, y_vec, count_vec = sparse.find(sparse_count_vec)

    # add in duplicates
    x_vec = np.repeat(x_vec, count_vec)
    y_vec = np.repeat(y_vec, count_vec)

    # sort the vecs
    sort_ix = np.argsort(x_vec)
    x_vec = x_vec[sort_ix]
    y_vec = y_vec[sort_ix]

    last_x = 0
    indexed_txt_list = []
    tmp_list = []
    for x, y in zip(x_vec, y_vec):
        if x > last_x:
            indexed_txt_list.append(torch.FloatTensor(tmp_list))
            tmp_list = [y]
        else:
            tmp_list.append(y)

        last_x = x

    # the dictionary key to match each int to the original word
    vocab_dict = vectorizer.vocabulary_

    print("Converted words to indexes of integers.")
    return indexed_txt_list, vocab_dict