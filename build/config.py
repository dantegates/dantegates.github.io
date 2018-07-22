import os

from .posts import IpynbPost

POSTS = tuple()


class PostConfig:
    assets_dir = 'assets'
    posts_dir = '_posts'
    parent_directory = '../'
    filename = None
    post_type = IpynbPost

    def __init_subclass__(cls):
        POSTS += (cls,)
        cls.filename = os.path.join(parent_directory, filename)


class (PostConfig):
    filename = 'bayesian-online-learning/a-brief-primer-on-conjugate-priors.ipynb'
    tags = ['bayesian', 'conjugate priors', 'online learning']


class (PostConfig):
    filename = 'bayesian-online-learning/bayesian-online-learning.ipynb'
    tags = ['bayesian', 'conjugate priors', 'online learning']


class (PostConfig):
    filename = 'fast-one-hot-encoder/fast-one-hot-encoder.ipynb'
    tags = ['one hot encoder', 'sklearn', 'pandas']


class (PostConfig):
    filename = 'from-docker-to-kubernetes/From-Docker-to-Kubernetes.ipynb'
    tags = ['docker', 'kubernetes', 'helm']


class (PostConfig):
    filename = 'image-search/image-search.ipynb'
    tags = ['image search', 'autoencoders', 'keras']


class (PostConfig):
    filename = 'know-your-trees/Know-your-trees.ipynb'
    tags = ['random forest', 'decision tree', 'trees']


class (PostConfig):
    filename = 'monitoring-ml-models/Monitoring-machine-learning-models-in-production.ipynb'
    tags = ['monitoring', 'logging', 'machine learning', 'production']


class (PostConfig):
    filename = 'attention-is-all-you-need-overview/An overview of Attention Is All You Need.ipynb'
    tags = ['attention is all you need', 'attention', 'keras', 'NLP']


class (PostConfig):
    filename = 'keras-feature-columns/keras-feature-columns.ipynb'
    tags = ['keras', 'tensorflow', 'opinion']
