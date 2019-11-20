from .config import PostConfig


class ClusteringAndImageSegmentation(PostConfig):
    title = 'Clustering and Image Segmentation'
    date_created = '2019-10-27'
    filename = 'clustering-and-image-segmentation/cluster-and-image-segmentation.ipynb'
    tags = ['Image Processing', 'Image Segmentation', 'Clustering', 'sklearn', 'KMeans']
    github_repo = 'clustering-and-image-segmentation'


class TensorflowFeatureColumnsKeras(PostConfig):
    title = 'Tensorflow 2 Feature Columns and Keras'
    date_created = '2019-10-24'
    filename = 'keras-feature-columns/tensorflow2-feature-columns.ipynb'
    tags = ['Keras', 'tensorflow', 'tensorflow2', 'Feature Columns']
    github_repo = 'keras-feature-columns'


class WorldSeriesProjections2019(PostConfig):
    title = '2019 World Series Pitcher Matchups'
    date_created = '2019-10-22'
    filename = 'mlb-statcast/player-ranking-post.ipynb'
    tags = ['pymc3', 'MLB', 'Statcast', 'World Series', 'Forecasting', 'Projections', 'Bayesian']
    github_repo = 'mlb-statcast'
    rebuild = False  # cell manually removed



class DeepLearningForTabularData2(PostConfig):
    title = 'Deep learning for tabular data 2 - Debunking the myth of the black box'
    date_created = '2019-06-14'
    filename = 'mlb-statcast/deep-learning-post.ipynb'
    tags = ['Deep Learning', 'AI', 'Tabular Data', 'MLB', 'Statcast']
    github_repo = 'mlb-statcast'
    rebuild = False  # text on this post was edited by hand in the _posts/ file


class DeepLearningForTabularData(PostConfig):
    title = 'Deep learning for tabular data'
    date_created = '2019-01-30'
    filename = 'deep-learning-for-tabular-data/post.ipynb'
    tags = ['Deep Learning', 'AI', 'Tabular Data', 'Word Embeddings', 'LSTM']
    github_repo = 'deep-learning-for-tabular-data'


class DeepProbabalisticEnsembles(PostConfig):
    title = 'Active learning and deep probabalistic ensembles'
    date_created = '2019-01-19'
    filename = 'deep-probabalistic-ensembles/deep-probabalistic-ensembles-cifar.ipynb'
    tags = ['Deep Learning', 'Active Learning', 'Bayesian Neural Networks', 'Variational Inference']
    github_repo = 'deep-probabalistic-ensembles'


class ModelEvaluationForHumans(PostConfig):
    title = 'Model Evaluation For Humans'
    date_created = '2019-01-07'
    filename = 'model-evaluation-for-humans/post.ipynb'
    tags = ['Data Science', 'Machine Learning', 'Industry', 'Model Evaluation']
    github_repo = 'model-evaluation-for-humans'
    rebuild = False


class WorldSeriesProjections(PostConfig):
    title = 'World Series Projections'
    date_created = '2018-10-22'
    filename = 'mlb-statcast/world-series-projections.ipynb'
    tags = ['pymc3', 'MLB', 'Statcast', 'World Series', 'Forecasting', 'Projections', 'Bayesian']
    github_repo = 'mlb-statcast'
    rebuild = False  # Some code/output was manually deleted from this post


class HierarchicalBayesianRanking(PostConfig):
    title = 'Hierarchical Bayesian Ranking'
    date_created = '2018-09-20'
    filename = 'mlb-statcast/bayesian-ranking-post.ipynb'
    tags = ['Bayesian', 'Ranking', 'pymc3', 'MLB', 'Statcast', 'World Series']
    github_repo = 'mlb-statcast'
    rebuild = False  # Some code/output was manually deleted from this post


class HypothesisTestingForHumans(PostConfig):
    title = 'Hypothesis Testing For Humans - Do The Umps Really Want to Go Home'
    date_created = '2018-09-17'
    filename = 'mlb-statcast/hypothesis-testing-for-humans.ipynb'
    tags = ['Bayesian Inference', 'Monte Carlo', 'pymc3', 'MLB', 'Statcast']
    github_repo = 'mlb-statcast'


class ImageSearchV2(PostConfig):
    title = 'Image Search Take 2 - Convolutional Autoencoders'
    date_created = '2018-09-12'
    filename = 'image-search/image-search-cnn.ipynb'
    tags = ['Image Search', 'Autoencoders', 'keras',
            'Convolutional Neural Networks', 'CIFAR']
    github_repo = 'image-search'
    rebuild = False  # if this gets rebuilt the static files included
                     # will be messed up


class KerasFeatureColumns(PostConfig):
    title = 'Keras Feature Columns'
    date_created = '2018-07-17'
    filename = 'keras-feature-columns/keras-feature-columns.ipynb'
    tags = ['keras', 'tensorflow', 'Opinion']
    github_repo = 'keras-feature-columns'


class AnOverviewOfAttentionIsAllYouNeed(PostConfig):
    title = 'An Overview of Attention Is All You Need'
    date_created = '2018-07-06'
    filename = 'attention-is-all-you-need-overview/An overview of Attention Is All You Need.ipynb'
    tags = ['Attention Is All You Need', 'Attention', 'keras', 'NLP']
    github_repo = 'attention-is-all-you-need'
    rebuild = False  # if this gets rebuilt the static files included
                     # will be messed up


class KnowYourTrees(PostConfig):
    title = 'Know Your Trees'
    date_created = '2018-06-15'
    filename = 'know-your-trees/Know-your-trees.ipynb'
    tags = ['Random Forest', 'Decision Tree', 'Trees']
    github_repo = 'know-your-trees'


class MonitoringML(PostConfig):
    title = 'Monitoring Machine Learning Models in Production'
    date_created = '2018-06-12'
    filename = 'monitoring-ml-models/Monitoring-machine-learning-models-in-production.ipynb'
    tags = ['Monitoring', 'Logging', 'Machine Learning', 'Production', 'Industry']
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
    tags = ['Bayesian', 'Conjugate Priors', 'Online Learning']
    github_repo = 'bayesian-online-learning'


class UnderstandingPriors(PostConfig):
    title = 'A brief primer on conjugate priors'
    date_created = '2018-05-11'
    filename = 'bayesian-online-learning/a-brief-primer-on-conjugate-priors.ipynb'
    tags = ['Bayesian', 'Conjugate Priors', 'Online Learning']
    github_repo = 'bayesian-online-learning'


class FastOneHotEncoder(PostConfig):
    title = 'A fast one hot encoder with sklearn and pandas'
    date_created = '2018-05-04'
    filename = 'fast-one-hot-encoder/fast-one-hot-encoder.ipynb'
    tags = ['One Hot Encoder', 'sklearn', 'pandas']
    github_repo = 'fast-one-hot-encoder'


class ImageSearch(PostConfig):
    title = 'Image search with autoencoders'
    date_created = '2018-05-01'
    filename = 'image-search/image-search.ipynb'
    tags = ['Image Search', 'Autoencoders', 'keras']
    github_repo = 'image-search'
