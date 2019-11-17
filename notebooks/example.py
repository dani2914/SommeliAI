from . import *

# %% [markdown]
# # Why do we care?
# We're interested in exploring probabilistic programming as it 
# applies to LDA and its variants.  LDA describes an extremely 
# intuitive generative process, and because of this it enjoys
# continued research into its expansions. It is also both flexible
# and interpretable.  Since everybody here knows what LDA is, we
# won't go over the details.
#
# But, inference is really, really hard.  Most expansions to LDA that
# get published require subtle tricks to even get inference working.
# Examples are:
#   1. Supervised LDA for classification only works for real-valued supervision;
#      (Blei & McAuliffe, 2008)
#   2. Multiple Classification LDA was published a full year later; it required a
#      subtle application of Jensen's inequality to reduce O(K^N) time to O(K^2)
#      (Wang et al., 2009)
#   3. Hierarchical Supervised LDA (Perotte, 2011) doesn't model a true is-a
#      hierarchical relationship (objects can be any number of nodes in a level)
# Just about every incremental idea requires some special trick
#
# ## What does this mean?
# In spite of how understandable and flexible LDA is, even statisticians and
# practitioners familiar with statistical models will have a tough time
# deploying criticizable models for their needs.
#
# Of course, this was covered on day 1; so, let's get started with pyro and LDA

#%% [markdown]
# # Dataset and Course of Research
# We study almost 300,000 wine reviews from WineEnthusiast.com.  This dataset is
# richly tagged, with numerical scores from 0 to 100, hierarchical region information
# (country->region->winery) and variety.
# Reviews are short but basically already bags of words; we don't anticipate many words
# being wasted on semantic nuance or anything, so it's not inconceivable that these reviews
# are long enough for LDA to work.
# > "Damp earth, black plum and dank forest herbs show 
# > on the nose of this single-vineyard expression. The palate offers 
# > cranberry and raspberry as well as savory soy and dried beef flavors,
# > all with earthy herbs in the background."

#%% [markdown]
# Given all of that, we expect to see a very interesting progression of topics as we march
# down our list of models:
# - LDA, 
# - LDA + classification v. supervised LDA,
# - Hierarchical LDA + classification v. supervised LDA
# - Hierarchical supervised LDA v. supervised LDA
# - Spectral Methods for supervised LDA v. supervised LDA
#%% [markdown]
# # LDA's Top Words Per Topic (10 topics)
# - aroma,dry,black,tannin,drink,cherry,finish,fruit,wine,flavor
# - black,drink,acidity,palate,finish,tannin,cherry,fruit,wine,flavor
# - aroma,ripe,dry,tannin,acidity,finish,cherry,fruit,wine,flavor
# - acidity,dry,drink,tannin,aroma,finish,cherry,fruit,flavor,wine
# - drink,black,palate,acidity,tannin,finish,cherry,fruit,wine,flavor
# - acidity,drink,tannin,aroma,dry,cherry,finish,fruit,flavor,wine
# - aroma,acidity,palate,black,tannin,finish,cherry,fruit,wine,flavor
# - acidity,black,aroma,tannin,dry,cherry,finish,fruit,wine,flavor
# - drink,aroma,cherry,acidity,dry,tannin,finish,fruit,flavor,wine
# - aroma,dry,palate,acidity,tannin,cherry,finish,fruit,wine,flavor
# I'm not certain what iteration or hyperparameter was used for this run,
# but I can assure you we ran many, and the results were all the same.

# # Supervised LDA?
# Perhaps our data is truly just a very poor fit for LDA; this wasn't
# unexpected.  We do expect that using more supervision, however, we can
# separate out meaningful topics for classification; the objective of LDA
# may itself not offer enough reward to finding distinct topics.
# # sLDA's top words per topic
# [insert words here]

#%% [markdown]
# # Posterior Collapse

#%% [markdown]
# # BBVI for LDA

#%% [markdown]
# # CAVI for LDA





# %% [markdown]
# # References
# (Blei & McAuliffe, 2008) https://papers.nips.cc/paper/3328-supervised-topic-models.pdf
# (Wang et al., 2009) http://vision.stanford.edu/pdf/WangBleiFei-Fei_CVPR2009.pdf
# (Schroeder, 2018) https://edoc.hu-berlin.de/bitstream/handle/18452/19516/thesis_schroeder_ken.pdf?sequence=3
# (Perotte et al., 2011) https://papers.nips.cc/paper/4313-hierarchically-supervised-latent-dirichlet-allocation
# (Thoutt, 2017) https://www.kaggle.com/zynicide/wine-reviews



# %%
x = 1.
print(x)

# %%
