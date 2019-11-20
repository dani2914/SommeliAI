import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from SommeliAI.customised_stopword import customised_stopword
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import data_util

files_root_dir = os.path.join("files")
main_color_scheme = mcolors.TABLEAU_COLORS.items()
graph_colors = np.array([color for name, color in main_color_scheme])

# get default filepath for notebook modules


def get_filepath(fname):
    fpath = os.path.join(files_root_dir, fname)

    return fpath


# build tsne and save for quick load
def build_tsne(trgt_df, save_fname):

    trgt_arr = trgt_df.values

    # highest prob topic for each doc [D x 1]
    group_ix = np.argmax(trgt_arr, axis=1)

    # run TSNE
    tsne_model = TSNE(
        n_components=2,
        verbose=1, random_state=0, angle=.99,
        init='pca'
    )
    tsne_df = tsne_model.fit_transform(trgt_arr)

    # add group ix to the first column
    tsne_df = pd.DataFrame(
        np.append(group_ix[:, None], tsne_df, axis=1),
        columns=['group', '0', '1'],
        index=trgt_df.index
    )

    # Save the tsne for faster loading on non-refresh setting
    tsne_df.to_csv(save_fname)

    return tsne_df


# load tsne
def load_tsne(load_fname):

    tsne_df = pd.read_csv(load_fname, index_col=0)

    return tsne_df


# graph tsne type dataframes
def graph_tsne(tsne_df):
    # Plot the Topic Clusters using Bokeh
    output_notebook()

    group_ix = tsne_df["group"].values.astype("int")

    n_topics = len(np.unique(group_ix))

    trgt_plot = figure(
        title=f"t-SNE Clustering of {n_topics} LDA Topics",
        plot_width=900, plot_height=700
    )

    trgt_plot.scatter(
        x=tsne_df["0"],
        y=tsne_df["1"],
        color=graph_colors[group_ix]
    )
    show(trgt_plot)


def graph_tsne_pair(tsne_tup):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('theta vs. phi')


def graph_word_dist(word_df):

    word_df.plot(
        figsize=(900, 700),
        title=f"Probability Distribution of {word_df.shape[0]} words",
        color=graph_colors[np.arange(word_df.shape[1])]
    )


def read_data(TESTING_SUBSIZE, data_root_dir):

    full_df = data_util.fetch_dataset(data_root_dir)

    # keep topics with the highest number of txt, and add min threshold if want
    full_df = data_util.filter_by_topic(full_df, keep_top_n_topics=10)

    # if not none, then subset the dataframe for testing purposes
    if TESTING_SUBSIZE is not None:
        full_df = full_df.sample(
            frac=TESTING_SUBSIZE,
            replace=False,
            random_state=666
        )

    # remove stop words, punctuation, digits and then change to lower case
    clean_df = data_util.preprocess(full_df, preprocess=True)
    print(clean_df)
    clean_df, indexed_txt_list, vocab_dict, vocab_count = (
        data_util.preprocess_and_index(
            clean_df,
            ngram=1,
            custom_stopwords=customised_stopword
        )
    )
    return clean_df, indexed_txt_list, vocab_dict


def generate_matrix(indexed_txt_list, vocab_size):
    counts = [
        [0 for i in range(vocab_size)]
        for j in range(len(indexed_txt_list))
    ]
    for j in range(len(indexed_txt_list)):
        for i in range(len(indexed_txt_list[j])):
            counts[j][int(indexed_txt_list[j].tolist()[i])] += 1
    counts = np.array(counts)
    return counts
