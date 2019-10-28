""" utility functions """

import os
import glob

import pandas as pd
import numpy as np
import torch

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

def fetch_dataset(n=None):
    """ fetch data from disk and return a dataframe """
    data_root_dir = os.path.join(".", "data")
    pattern = "winemag_dataset_*.csv"

    file_list = glob.glob(os.path.join(data_root_dir, pattern))

    df_list = [pd.read_csv(fname) for fname in file_list]
    
    full_df = pd.concat(df_list)

    # give unique row names to all
    full_df.index = range(full_df.shape[0])

    if n is not None:
        full_df = full_df.head(n)

    print("Dataset fetched.")
    return full_df

def preprocess(full_df):
    """ maybe clean up text, filter out rows with empty cells, remove stop words? """

    clean_df = full_df

    print("Dataset cleaned.")
    return clean_df

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
