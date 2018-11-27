from .config import PostConfig


class WorldSeriesProjections(PostConfig):
    title = 'World Series Projections'
    date_created = '2018-10-22'
    filename = 'mlb-statcast/world-series-projections.ipynb'
    tags = ['pymc3', 'MLB', 'statcast', 'World Series', 'forecasting', 'projections']
    github_repo = 'mlb-statcast'
    rebuild = False  # Some code/output was manually deleted from this post


class HierarchicalBayesianRanking(PostConfig):
    title = 'Hierarchical Bayesian Ranking'
    date_created = '2018-09-20'
    filename = 'mlb-statcast/bayesian-ranking-post.ipynb'
    tags = ['Bayesian', 'Ranking', 'pymc3', 'MLB', 'statcast', 'World Series']
    github_repo = 'mlb-statcast'
    rebuild = False  # Some code/output was manually deleted from this post


class HypothesisTestingForHumans(PostConfig):
    title = 'Hypothesis Testing For Humans - Do The Umps Really Want to Go Home'
    date_created = '2018-09-17'
    filename = 'mlb-statcast/Hypothesis Testing For Humans - Do the Umps Really Want to Go Home.ipynb'
    tags = ['Bayesian Inference', 'Monte Carlo', 'pymc3', 'MLB', 'statcast']
    github_repo = 'mlb-statcast'


class ImageSearchV2(PostConfig):
    title = 'Image Search Take 2 - Convolutional Autoencoders'
    date_created = '2018-09-12'
    filename = 'image-search/image-search-cnn.ipynb'
    tags = ['image search', 'autoencoders', 'keras', 'CNN',
            'Convolutional Neural Networks', 'CIFAR']
    github_repo = 'image-search'
    rebuild = False  # if this gets rebuilt the static files included
                     # will be messed up


class KerasFeatureColumns(PostConfig):
    title = 'Keras Feature Columns'
    date_created = '2018-07-17'
    filename = 'keras-feature-columns/keras-feature-columns.ipynb'
    tags = ['keras', 'tensorflow', 'opinion']
    github_repo = 'keras-feature-columns'


class AnOverviewOfAttentionIsAllYouNeed(PostConfig):
    title = 'An Overview of Attention Is All You Need'
    date_created = '2018-07-06'
    filename = 'attention-is-all-you-need-overview/An overview of Attention Is All You Need.ipynb'
    tags = ['attention is all you need', 'attention', 'keras', 'NLP']
    github_repo = 'attention-is-all-you-need'
    rebuild = False  # if this gets rebuilt the static files included
                     # will be messed up


class KnowYourTrees(PostConfig):
    title = 'Know Your Trees'
    date_created = '2018-06-15'
    filename = 'know-your-trees/Know-your-trees.ipynb'
    tags = ['random forest', 'decision tree', 'trees']
    github_repo = 'know-your-trees'


class MonitoringML(PostConfig):
    title = 'Monitoring Machine Learning Models in Production'
    date_created = '2018-06-12'
    filename = 'monitoring-ml-models/Monitoring-machine-learning-models-in-production.ipynb'
    tags = ['monitoring', 'logging', 'machine learning', 'production']
    github_repo = 'monitoring-ml'


class FromDockerToKubernetes(PostConfig):
    title = 'From Docker to Kubernetes'
    date_created = '2018-05-24'
    filename = 'from-docker-to-kubernetes/From-Docker-to-Kubernetes.ipynb'
    tags = ['docker', 'kubernetes', 'helm']
    github_repo = 'from-docker-to-kubernetes'


class BayesianOnlineLearning(PostConfig):
    title = 'Bayesian Online Learning'
    date_created = '2018-05-11'
    filename = 'bayesian-online-learning/bayesian-online-learning.ipynb'
    tags = ['bayesian', 'conjugate priors', 'online learning']
    github_repo = 'bayesian-online-learning'


class UnderstandingPriors(PostConfig):
    title = 'A brief primer on conjugate priors'
    date_created = '2018-05-11'
    filename = 'bayesian-online-learning/a-brief-primer-on-conjugate-priors.ipynb'
    tags = ['bayesian', 'conjugate priors', 'online learning']
    github_repo = 'bayesian-online-learning'


class FastOneHotEncoder(PostConfig):
    title = 'A fast one hot encoder with sklearn and pandas'
    date_created = '2018-05-04'
    filename = 'fast-one-hot-encoder/fast-one-hot-encoder.ipynb'
    tags = ['one hot encoder', 'sklearn', 'pandas']
    github_repo = 'fast-one-hot-encoder'


class ImageSearch(PostConfig):
    title = 'Image search with autoencoders'
    date_created = '2018-05-01'
    filename = 'image-search/image-search.ipynb'
    tags = ['image search', 'autoencoders', 'keras']
    github_repo = 'image-search'
