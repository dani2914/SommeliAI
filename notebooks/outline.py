from SommeliAI import util

# %% [markdown]
# # Motivation
# We're interested in studying topic modelling -- specifically LDA and its variants -- in probabilistic programming languages
# This is hard.

# %%
#Section 1: This section will describe the data briefly
#(1) Histogram of Topics (Variants of Wine)
#(2) Histogram of top 10 Topics

# %%
# Section 2: This Section highlights a full run of vanilla LDA (Run 1)
# Normal stop word processing, lemmatizing
# Model
# - show model graphical plate notation
# - show guide graphical plate notation
# Inference
# - Explain use of Trace ELBO
# Criticism
#(1) show T-SNE
#(2) show word distribution
#(3) number of unique words in top 10 words for each

# %%
# Section 3: This Section highlights a deeper dive into the data
# - show most frequent words
# - show custom list of stop words to remove

# %%
# Secton 4: Show plain LDA run with custom stop words removed
# - show the word cloud of top 10 words and reveal basically next 10 overlap words
# - show word distribution again with spikes of next 10 words

# %%
# Section 5: Maybe data is noise? Not true as shown by CAVI
# - show CAVI results

# %%
# Section 6: Maybe still have predictive power - sLDA
