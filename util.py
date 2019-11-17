""" utility functions """

import os
import glob
import functools

import pandas as pd
import numpy as np
import torch
import re
import string
from customised_stopword import customised_stopword

import nltk
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

lemmatizer = WordNetLemmatizer()
snow = SnowballStemmer('english')

stop_n_punct_words = set(stopwords.words("english") + list(string.punctuation))


def fetch_dataset():
    """ fetch data from disk and return a dataframe """
    data_root_dir = os.path.join(".", "data")
    pattern = "winemag_dataset_*.csv"

    file_list = glob.glob(os.path.join(data_root_dir, pattern))

    df_list = [pd.read_csv(fname) for fname in file_list]

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


def remove_duplicate(txt):
    from collections import defaultdict
    word_count = defaultdict(int)
    words = txt.split(" ")
    for word in words:
        word_count[word] += 1

    return " ".join(word_count.keys())


def clean_stop_punct_digit_n_lower(txt):

    txt = re.sub(r"[\'\.,-?!]", " ", txt)
    #token = txt.split(" ")
    token = word_tokenize(txt)
    #token = lemmatizer.lemmatize(token)

    clean_token = [word.lower() for word in token if word.lower()
                   not in stop_n_punct_words and re.match(r"^.*\d+.*$", word) is None
                   and len(word) >= 4 and "\'" not in word and not word.isnumeric()]

    return " ".join(clean_token)


def preprocess_tokens(txt, ngram, custom_stopwords):
    tokens = word_tokenize(txt)
    stop_words = stop_n_punct_words if custom_stopwords is None \
        else set(list(stop_n_punct_words) + custom_stopwords)

    if ngram > 1:
        tokens = ["x_____x" if token.lower() in stop_words or re.match(r"^.*\d+.*$", token)
                  else token.lower() for token in tokens]
    else:
        tokens = [token.lower() for token in tokens if token
                   not in stop_words and re.match(r"^.*\d+.*$", token) is None]

    tokens = " ".join(tokens)

    return tokens


def preprocess_and_index(tmp_df, ngram=1, custom_stopwords=None):

    print("Preprocessing tokens... (this part is slow)")
    tmp_df["description"] = tmp_df["description"].apply(
        preprocess_tokens, args=(ngram, custom_stopwords))

    print("Building Index and ngram...")
    vectorizer = CountVectorizer(
        ngram_range=(1, ngram), analyzer="word", strip_accents="unicode")
    sparse_count_vec = vectorizer.fit_transform(tmp_df["description"])

    if ngram > 1:
        tmp_df["description"] = tmp_df["description"].apply(
            lambda txt: re.sub("x_____x ", "", txt))

    x_vec, y_vec, count_vec = sparse.find(sparse_count_vec)

    # add in duplicates
    x_vec = np.repeat(x_vec, count_vec)
    y_vec = np.repeat(y_vec, count_vec)

    # the dictionary key to match each int to the original word
    vocab_dict = vectorizer.vocabulary_

    key_list, value_list = [], []
    for key, value in list(vocab_dict.items()):
        key_list.append(key)
        value_list.append(value)

    sort_ix = np.argsort(value_list)
    key_list = np.array(key_list)[sort_ix]
    #value_list = np.array(value_list)[sort_ix]

    sort_ix = np.argsort(y_vec)
    y_vec = y_vec[sort_ix]
    x_vec = x_vec[sort_ix]

    unique_y_vec, y_count_vec = np.unique(y_vec, return_counts=True)

    dict_del_key_vec = [key.startswith('x_____x') or key.endswith('x_____x') for key in key_list]
    key_list = key_list[np.logical_not(dict_del_key_vec)]
    val_list = np.arange(len(key_list))

    del_key_vec = np.repeat(dict_del_key_vec, y_count_vec)
    x_vec = x_vec[np.logical_not(del_key_vec)]
    y_vec = np.repeat(np.arange(len(key_list)), y_count_vec[np.logical_not(dict_del_key_vec)])

    filtered_dict = dict(zip(key_list, val_list))
    unique, counts = np.unique(y_vec, return_counts=True)
    vocab_count = dict(zip(unique, counts))

    # convert to torch variables
    x_vec = torch.tensor(x_vec, dtype=torch.int32)
    y_vec = torch.tensor(y_vec, dtype=torch.float)

    # sort the vecs
    sort_ix = torch.argsort(x_vec)
    x_vec = x_vec[sort_ix]
    y_vec = y_vec[sort_ix]

    x_vec_bincount = torch.bincount(x_vec.cpu())
    bincount_tup = tuple(int(bincount) for bincount in x_vec_bincount)
    indexed_txt_list = list(torch.split(y_vec, bincount_tup))

    print("Preprocessing + Indexing complete.")

    return tmp_df, indexed_txt_list, filtered_dict, vocab_count


def preprocess(tmp_df, preprocess=False):
    """ removing stop words, punctuations, digits and make all lower case """

    # all in one go in order to just have to tokenize once
    if preprocess:
        tmp_df["description"] = tmp_df["description"].apply(
            clean_stop_punct_digit_n_lower)
    # words = tmp_df['description'].str.split(expand=True).stack().value_counts()
    # ratio = tmp_df['description'].apply(remove_duplicate).str.split(expand=True).stack().value_counts() / tmp_df.shape[0]
    # words.to_csv('freq_words.csv')
    # ratio.to_csv("ratio.csv")

    return tmp_df


def conv_word_to_indexed_txt(txt_vec):
    """ change txt into int indexes, return with dict of indexes & count of each word """

    # transform words into integer indexes, comes out as n x m
    # where n = # txt doc, m = # unique words for whole universe
    vectorizer = CountVectorizer(analyzer='word')
    #CountVectorizer(ngram_range=(1,2), analyzer='word')
    sparse_count_vec = vectorizer.fit_transform(txt_vec)

    # create n x p list of words represented by ints,  where p = # words in each documentx
    # written in such a convoluted way for speed optimization purposes 
    x_vec, y_vec, count_vec = sparse.find(sparse_count_vec)

    # add in duplicates
    x_vec = np.repeat(x_vec, count_vec)
    y_vec = np.repeat(y_vec, count_vec)

    #convert to torch variables
    x_vec = torch.tensor(x_vec, dtype=torch.int32)
    y_vec = torch.tensor(y_vec, dtype=torch.float)

    # sort the vecs
    sort_ix = torch.argsort(x_vec)
    x_vec = x_vec[sort_ix]
    y_vec = y_vec[sort_ix]

    x_vec_bincount = torch.bincount(x_vec.cpu())
    bincount_tup = tuple(int(bincount) for bincount in x_vec_bincount)
    indexed_txt_list = list(torch.split(y_vec, bincount_tup))

    # the dictionary key to match each word to int
    vocab_dict = vectorizer.vocabulary_

    print("Converted words to indexes of integers.")

    vocab_count = sparse_count_vec.data

    return indexed_txt_list, vocab_dict, vocab_count


def generate_hierarchical_mapping(data, hierarchy):
    mapping = {}
    depth = len(hierarchy)
    coord = np.zeros(depth, dtype=np.int64)
    for key in data.set_index(hierarchy).sort_index().index:
        tmp = mapping
        for d in range(depth):
            if key[d] not in tmp:
                tmp[key[d]] = {}
                coord[d] += 1
                coord[d+1:] = 0
            if d < depth - 1:
                tmp = tmp[key[d]]
        tmp[key[d]] = tuple(coord)
    
    return mapping


def enrich_data_hierarchical_coordinates(data, hierarchy, mapping):
    data = data.set_index(hierarchy).sort_index()
    depth = len(hierarchy)
    for d in range(depth):
        data.loc[:, f"coord_{d}"] = 0

    cols = [f"coord_{d}" for d in range(depth)]

    for idx in data.index:
        data.loc[idx, cols] = (
            functools.reduce(lambda x, l: x.get(l), idx, mapping)
        )
    
    data = data.reset_index()

    return data.reset_index()


def simplify_topic_hierarchy(tree):
    out = []

    def tree_names_to_nested_list(tree, out, nlabels):
        if isinstance(tree, dict):
            for key in tree.keys():
                out.append([key])
                nlabels = tree_names_to_nested_list(tree[key], out[-1], nlabels+1)
        return nlabels
    
    nlabels = tree_names_to_nested_list(tree, out, 0)
    return out, nlabels


def flatten(nested_list):
    acc = []
    for ii in nested_list:
        if isinstance(ii, list):
            acc += flatten(ii)
        else:
            acc += [ii]
    return acc


def get_all_supervised_requirements(fname=None, max_samples=0, max_annotations=500, max_classes=15):
    if fname is not None:
        wine = pd.read_csv(fname)
    else:
        wine = fetch_dataset()

    clean_df = preprocess(wine, preprocess=True)
    clean_df = preprocess_and_index(clean_df, 2, customised_stopword)
    clean_df = clean_df.dropna()  # some countries are nan

    if max_samples > 0:
        clean_df = clean_df.head(max_samples)

    # make our lives easier and drop any documents with too few words
    indexed_txt_list, vocab_dict, vocab_counts = conv_word_to_indexed_txt(clean_df['description'])
    document_word_counts = np.array([len(d) for d in indexed_txt_list])
    min_acceptable_words = np.quantile(document_word_counts, 0.05)
    drop = document_word_counts < min_acceptable_words
    clean_df = clean_df.drop(clean_df.index[drop], axis=0)               
                                                                                
    # generate topic labels, potentially dropping rows corresponding to infrequent topics
    class_assignments = clean_df["variety"]
    topics, topic_counts = np.unique(class_assignments, return_counts=True)
    if max_classes > 0:
        topics = topics[np.argsort(topic_counts)[::-1]]
        topics = topics[:min(len(topics), max_classes)]
    topic_map = {topics[i]: i for i in range(len(topics))}
    clean_df.loc[:, "class"] = clean_df.loc[:, "variety"].apply(
        lambda row: topic_map[row] if row in topic_map else np.nan
    )
    clean_df = clean_df.dropna(subset=["class"])

    # generate annotations
    # assume locations are 'annotations', and each wine has 3 annotations
    annotation_cols = ["country", "province", "winery"]
    annots, anno_counts = np.unique(
        clean_df[annotation_cols].values.reshape(
            (np.product(clean_df[annotation_cols].values.shape), )
        ), return_counts=True
    )
    if max_annotations > 0:
        annots = annots[np.argsort(anno_counts)[::-1]]
        annots = annots[:min(len(annots), max_annotations)]
    annots_map = {annots[i]: i for i in range(len(annots))}
    clean_df[annotation_cols] = clean_df[annotation_cols].applymap(
        lambda s: annots_map[s] if s in annots_map else len(annots)
    ) # force annote all infrequent words as an "other" category

    num_topics = len(topics)
    num_vocab = len(vocab_dict)

    doc_hist_words, doc_hist_counts = (
        zip(*[(dd, 
        (dd == d[:, np.newaxis].numpy()).sum(0))
        for d in indexed_txt_list for dd in [np.unique(d)]])
    )

    docs_words = np.full(
        (len(doc_hist_words), max([len(d) for d in doc_hist_words])),
        np.nan
    )
    for e, d in enumerate(doc_hist_words):
        docs_words[e, :len(d)] = d

    docs_counts = np.full(
        docs_words.shape,
        np.nan
    )
    for e, d in enumerate(doc_hist_counts):
        docs_counts[e, :len(d)] = d

    return clean_df, docs_words, docs_counts, num_topics, num_vocab, min_acceptable_words

