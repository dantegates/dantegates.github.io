from .config import PostConfig
from .posts import MarkdownPost


class PyMConAfterword(PostConfig):
    title = 'PyMCon Afterword'
    date_created = '2023-02-22'
    filename = 'the-power-of-bayes-in-industry/Afterword.md'
    tags = ['Bayesian Inference', 'pymc']
    github_repo = 'the-power-of-bayes-in-industry'
    post_type = MarkdownPost


class PandasTimeseriesIndex(PostConfig):
    title = 'Time Series With Pandas'
    date_created = '2021-01-03'
    filename = 'pandas-timeseries/pandas-timeseries-and-index.ipynb'
    tags = ['pandas', 'Time Series']
    github_repo = 'pandas-timeseries-demo'


class EvaluatingMLBProjectionsPart2(PostConfig):
    title = 'Evaluating my 2020 MLB Predictions - Part 2, The Postseason'
    date_created = '2020-11-10'
    filename = 'mlb-statcast/evaluating-2020-predictions-part2.md'
    tags = ['MLB', 'Model Evaluations']
    github_repo = 'mlb-statcast'
    post_type = MarkdownPost


class EvaluatingMLBProjectionsPart1(PostConfig):
    title = 'Evaluating my 2020 MLB Predictions - Part 1, Pete Alonso'
    date_created = '2020-11-10'
    filename = 'mlb-statcast/evaluating-2020-predictions-part1.md'
    tags = ['MLB', 'Model Evaluations']
    github_repo = 'mlb-statcast'
    post_type = MarkdownPost
    # consider incorporating the "rule of 40" in to the analysis


class MLBPostseasonProjections2020(PostConfig):
    title = 'MLB 2020 Postseason Projections'
    date_created = '2020-09-29'
    filename = 'mlb-statcast/2020-postseason-post.ipynb'
    tags = ['MLB', 'Bayesian Inference']
    github_repo  = 'mlb-statcast'
    


class WhyKuberenetes(PostConfig):
    title = "A Beginner's Guide to Why You Should or Shouldn't Be Using Kubernetes for Machine Learning (With Illustrations)"
    date_created = '2020-08-07'
    filename = 'why-kubernetes/post.md'
    tags = ['Opinion', 'kubernetes']
    github_repo  = 'why-kubernetes'
    post_type = MarkdownPost


class CollaborativeFilteringTutorial(PostConfig):
    title = 'A Tutorial on Collaborative Filtering in sklearn'
    date_created = '2020-04-21'
    filename = 'practical-collaborative-filtering/collaborative-filtering.ipynb'
    tags = ['Recommendation Systems', 'sklearn']
    github_repo = 'collaborative-filtering-tutorial'


class MyQuarantinePlaylist(PostConfig):
    title = 'My Quarantine Playlist'
    date_created = '2020-04-20'
    filename = 'my-quarantine-playlist/post.md'
    tags = ['Bayesian Inference']
    github_repo = 'my-quarantine-playlist'
    post_type = MarkdownPost


class PredictingPeteAlonso2020(PostConfig):
    title = 'Predicting Pete Alonso\'s 2020 Performance'
    date_created = '2020-03-03'
    filename = 'mlb-statcast/predicting-pete-alonsos-2020-season.md'
    tags = ['Bayesian Inference', 'MLB']
    github_repo = 'mlb-statcast'
    post_type = MarkdownPost


class ScalingPredictions(PostConfig):
    title = 'Scaling Predictions'
    date_created = '2020-02-08'
    filename = 'scaling-predictions/post.md'
    tags = ['Machine Learning']
    github_repo = 'scaling-predictions'
    post_type = MarkdownPost


# class DeepLearningForIndustry3(PostConfig):
#     title = 'Deep Learning for Industry - Debunking the Myth of the Black Box'
#     date_created = '2020-01-06'
#     filename = 'deep-ar/uncertainty-plots.md'
#     tags = ['Deep Learning', 'Time Series']
#     github_repo = 'deep-ar'
#     post_type = MarkdownPost


class DeepLearningForTimeSeries(PostConfig):
    title = 'Deep Learning for Time Series'
    date_created = '2020-01-02'
    filename = 'deep-ar/post.md'
    tags = ['Deep Learning', 'Time Series']
    github_repo = 'deep-ar'
    post_type = MarkdownPost


class NeuralNetworksExplained(PostConfig):
    title = 'Neural Networks Explained'
    date_created = '2019-12-12'
    filename = 'neural-networks-explained/post.md'
    tags = ['Deep Learning', 'Beginner', 'Opinion']
    github_repo = 'neural-networks-explained'
    post_type = MarkdownPost


class ClusteringAndImageSegmentation(PostConfig):
    title = 'Clustering and Image Segmentation'
    date_created = '2019-10-27'
    filename = 'clustering-and-image-segmentation/cluster-and-image-segmentation.ipynb'
    tags = ['Images', 'Machine Learning', 'sklearn']
    github_repo = 'clustering-and-image-segmentation'


class TensorflowFeatureColumnsKeras(PostConfig):
    title = 'Tensorflow 2 Feature Columns and Keras'
    date_created = '2019-10-24'
    filename = 'keras-feature-columns/tensorflow2-feature-columns.ipynb'
    tags = ['keras', 'tensorflow', 'tensorflow2']
    github_repo = 'keras-feature-columns'


class WorldSeriesProjections2019(PostConfig):
    title = '2019 World Series Pitcher Matchups'
    date_created = '2019-10-22'
    filename = 'mlb-statcast/player-ranking-post.ipynb'
    tags = ['pymc3', 'MLB', 'Projections', 'Bayesian Inference']
    github_repo = 'mlb-statcast'
    rebuild = False  # cell manually removed



class DeepLearningForIndustry2(PostConfig):
    title = 'Deep Learning for Tabular Data 2 - Debunking the Myth of the Black Box'
    date_created = '2019-06-14'
    filename = 'mlb-statcast/deep-learning-post.ipynb'
    tags = ['Deep Learning', 'MLB', 'Attention', 'Industry']
    github_repo = 'mlb-statcast'
    rebuild = False  # text on this post was edited by hand in the _posts/ file


class DeepLearningForIndustry(PostConfig):
    title = 'Deep Learning for Tabular Data'
    date_created = '2019-01-30'
    filename = 'deep-learning-for-tabular-data/post.ipynb'
    tags = ['Deep Learning', 'Industry']
    github_repo = 'deep-learning-for-tabular-data'


class DeepProbabalisticEnsembles(PostConfig):
    title = 'Active Learning and Deep Probabilistic Ensembles'
    date_created = '2019-01-19'
    filename = 'deep-probabalistic-ensembles/deep-probabalistic-ensembles-cifar.ipynb'
    tags = ['Deep Learning', 'Active Learning', 'Bayesian Inference', 'Variational Inference', 'AI']
    github_repo = 'deep-probabalistic-ensembles'


class ModelEvaluationForHumans(PostConfig):
    title = 'Model Evaluation For Humans'
    date_created = '2019-01-07'
    filename = 'model-evaluation-for-humans/post.ipynb'
    tags = ['Industry', 'Model Evaluation']
    github_repo = 'model-evaluation-for-humans'
    rebuild = False


class WorldSeriesProjections(PostConfig):
    title = 'World Series Projections'
    date_created = '2018-10-22'
    filename = 'mlb-statcast/world-series-projections.ipynb'
    tags = ['MLB', 'Bayesian Inference', 'Projections', 'Monte Carlo', 'pymc3']
    github_repo = 'mlb-statcast'
    rebuild = False  # Some code/output was manually deleted from this post


class HierarchicalBayesianRanking(PostConfig):
    title = 'Hierarchical Bayesian Ranking'
    date_created = '2018-09-20'
    filename = 'mlb-statcast/bayesian-ranking-post.ipynb'
    tags = ['Bayesian Inference', 'Learning to Rank', 'pymc3', 'MLB']
    github_repo = 'mlb-statcast'
    rebuild = False  # Some code/output was manually deleted from this post


class HypothesisTestingForHumans(PostConfig):
    title = 'Hypothesis Testing For Humans - Do The Umps Really Want to Go Home'
    date_created = '2018-09-17'
    filename = 'mlb-statcast/hypothesis-testing-for-humans.ipynb'
    tags = ['Bayesian Inference', 'MLB', 'Monte Carlo', 'pymc3']
    github_repo = 'mlb-statcast'


class ImageSearchV2(PostConfig):
    title = 'Image Search Take 2 - Convolutional Autoencoders'
    date_created = '2018-09-12'
    filename = 'image-search/image-search-cnn.ipynb'
    tags = ['Images', 'Recommendation Systems', 'Auto Encoders', 'Deep Learning', 'AI', 'keras']
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
    tags = ['Attention', 'Deep Learning', 'keras']
    github_repo = 'attention-is-all-you-need'
    rebuild = False  # if this gets rebuilt the static files included
                     # will be messed up


class KnowYourTrees(PostConfig):
    title = 'Know Your Trees'
    date_created = '2018-06-15'
    filename = 'know-your-trees/Know-your-trees.ipynb'
    tags = ['Machine Learning']
    github_repo = 'know-your-trees'


class MonitoringML(PostConfig):
    title = 'Monitoring Machine Learning Models in Production'
    date_created = '2018-06-12'
    filename = 'monitoring-ml-models/Monitoring-machine-learning-models-in-production.ipynb'
    tags = ['Industry', 'Software', 'Machine Learning']
    github_repo = 'monitoring-ml'


class FromDockerToKubernetes(PostConfig):
    title = 'From Docker to Kubernetes'
    date_created = '2018-05-24'
    filename = 'from-docker-to-kubernetes/From-Docker-to-Kubernetes.ipynb'
    tags = ['docker', 'kubernetes', 'helm', 'Software', 'Industry']
    github_repo = 'from-docker-to-kubernetes'


class BayesianOnlineLearning(PostConfig):
    title = 'Bayesian Online Learning'
    date_created = '2018-05-11'
    filename = 'bayesian-online-learning/bayesian-online-learning.ipynb'
    tags = ['Bayesian Inference', 'Online Learning']
    github_repo = 'bayesian-online-learning'


class UnderstandingPriors(PostConfig):
    title = 'A Brief Primer on Conjugate Priors'
    date_created = '2018-05-11'
    filename = 'bayesian-online-learning/a-brief-primer-on-conjugate-priors.ipynb'
    tags = ['Bayesian Inference']
    github_repo = 'bayesian-online-learning'


class FastOneHotEncoder(PostConfig):
    title = 'A Fast One Hot Encoder With sklearn and pandas'
    date_created = '2018-05-04'
    filename = 'fast-one-hot-encoder/fast-one-hot-encoder.ipynb'
    tags = ['sklearn', 'pandas']
    github_repo = 'fast-one-hot-encoder'


class ImageSearch(PostConfig):
    title = 'Image Search With Autoencoders'
    date_created = '2018-05-01'
    filename = 'image-search/image-search.ipynb'
    tags = ['Recommendation Systems', 'Images', 'Auto Encoders', 'AI', 'Deep Learning', 'keras']
    github_repo = 'image-search'
