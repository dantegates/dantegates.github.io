---
layout: post
mathjax: true
title: Active learning and deep probabilistic ensembles
github: https://github.com/dantegates/deep-probabilistic-ensembles
creation_date: 2019-01-19
last_modified: 2019-01-19 22:02:36
tags: 
  - Deep Learning
  - Active Learning
  - Bayesian Inference
  - Variational Inference
  - AI
---


[Active learning](http://www.cs.northwestern.edu/~pardo/courses/mmml/papers/active_learning/improving_generalization_with_active_learning_ML94.pdf), loosely described, is an iterative process for getting the most out of your training data. This is especially useful for cases where you have a lot of unlabeled data that you would like to use for [supervised training](https://en.wikipedia.org/wiki/Supervised_learning) but labeling the data is extremely time consuming and/or costly. In this case you want to be able intelligently choose which data points to label next so that you total training set consists of a rich and diverse set of examples with minimal redundancy.

Last week at PyData Miami, Clément Farabet of NVIDIA discussed how this problem relates to the massive amount of training data NVIDIA has collected for their autonomous car project. In particular he mentioned a paper that some of their researchers recently released that introduces "[deep probabilistic ensembles](https://arxiv.org/pdf/1811.03575.pdf)" which they apply alongside active learning as a solution to this problem. In this post we'll take a look at the ideas introduced in the paper.

## Motivation

Why "deep probabilistic ensembles"? To answer this question, let's take a step back and consider a common pattern for active learning. Suppose we have two data sets at hand, $D_{L}=\{X_{L}, y_{L}\}$ and $D_{U}=\{X_{U}\}$, where the first data set is labeled and the second is not. Additionally suppose we have a model $f: X\rightarrow y$ and a function $u: f(x)\rightarrow \mathbb{R}$ that measures the uncertainty of a given prediction $f(x)$. We can train $f$ on $D_{L}$ and exploit the uncertainty of the predictions $\{f(x); x\in X_{U}\}$ to determine which data points in $X_{U}$ should be labeled next. For example, a sketch of this algorithm looks like

1. Train $f$ on $D_{L}$.
2. Obtain the predictions $\{f(x); x\in X_{U}\}$.
3. Use $u$ to calculate the uncertainty of each prediction.
4. Select the top $n$ data points in $X_{U}$ where the model's predictions are most uncertain.

After labeling these points we can update $D_{L}$ and repeat the process as desired.

Of course, since we've mentioned the word "uncertainty" several times by now it should be clear why we are interested in deep probabilistic networks. Theoretically we can define a deep probabilistic, or Bayesian, neural network as

$$
\begin{equation}
P(w \ \vert \ x)=\frac{P(w)P(w\ \vert \ x)}{P(x)}
\label{eq:posterior}
\end{equation}
$$

where $w$ is the set of all weights in the network for which we define the prior $P(w)$. From such a model we could obtain not only predictions of our target variable $y$ but also a measure of uncertainty for those predictions.

However, it is well known that training a deep network like this is a difficult, if not impossible, task. Enter "deep probabilistic ensembles" which approximate of the posterior $P(w \ \vert \ x)$.

## Approximating deep Bayesian neural networks

Since NVIDIA paper is pretty short and self explanatory so I'll only cover the details necessary to grok the code below.

The key idea the author's introduce is a loss function that allows us to learn an ensemble of deep neural networks so that the ensemble itself approximates samples from the posterior in \eqref{eq:posterior}.

To derive the objective, the authors begin by defining the usual objective in variational inference

$$
q^{*}=\underset{q}{\text{arg min}} \mathbb{E}\left [ \text{log }{\frac{q(w)}{p(w\ \vert \ x)}} \right]
$$

With some algebra they rewrite the objective as

$$
KL(q(w)\vert\vert p(w)) - \mathbb{E}\left[\text{log }p(x \ \vert \ w )\right] + \text{log }p(x)
$$

To make the objective computationally tractable the last term, which is independent of $w$, is removed, resulting in a new objective, the [Evidence Lower Bound](https://en.wikipedia.org/wiki/Evidence_lower_bound),

$$
KL(q(w)\vert\vert p(w)) - \mathbb{E}\left[\text{log }p(x \ \vert \ w )\right]
$$

During training, the second term in the equation above is approximated by aggregating the cross entropy for each model in the ensemble. The prior placed on the weights is specified as a guassian and the weights are completely pooled such that the first term can be calculated for a given layer (with some manipulation removing terms independent of $w$) as

$$
KL(q(w)\vert\vert p(w))=\sum_{i}^{n_{i}n_{o}wh}{\text{log }\sigma_{i}^{2}+\frac{2}{n_{o}wh\sigma_{i}^{2}}+\frac{\mu_{i}^{2}}{\sigma_{i}^{2}}}
$$

where $n_{i}$, $n_{o}$, $w$ and $h$ are the number of inputs and outputs and width and height of the kernel respectively. The means and variance are over the values of a particular weights accross the ensemble. Note deriving the equation above relies on centering the prior at 0 with variance $\frac{2}{n_{o}wh}$.

# keras implementation

Below is an implementation of the "Deep probabilistic ensemble" using `keras` and a simulation of the active learning experiments from the paper. Note that I was able to run the experiments on a [GTX 1060](https://www.nvidia.com/en-in/geforce/products/10series/geforce-gtx-1060/) but it was an overnight job.

As in the NVIDIA paper, the ensemble consists of several ResNet18s. I used the [keras_contrib implementation](https://github.com/keras-team/keras-contrib/blob/d638cf409f7c8d3d042feac5269c12d507398eeb/keras_contrib/applications/resnet.py#L1) with just a few minor modifications so the returned models could be used as an ensemble.


```python
import keras
import keras.backend as K
from resnet import ResNet
import numpy as np


def kl_regularization(layers):
    """Return the KL regularization penalty computed from a list of `layers`."""
    layers = K.stack(layers, axis=0)
    layer_dims = K.cast_to_floatx(K.int_shape(layers[0]))
    n_w = layer_dims[0]
    n_h = layer_dims[1]
    n_o = layer_dims[3]
    mu_i = K.mean(layers, axis=0)
    var_q_i = K.var(layers, axis=0)
    var_p_i = 2 / (n_w * n_h * n_o)
    kl_i = K.log(var_q_i) + (var_p_i / var_q_i) + (mu_i**2 / var_q_i)
    return K.sum(kl_i)


def ensemble_crossentropy(y_true, y_pred):
    """Return the cross entropy for the ensemble.
    
    Args:
        y_true: Tensor with shape (None, n_classes).
        y_pred: Tensor with shape (None, n_ensemble_members, n_classes).
    """
    ensemble_entropy = K.categorical_crossentropy(y_true, y_pred, axis=-1)
    return K.sum(ensemble_entropy, axis=-1)


class Stack(keras.layers.Layer):
    """Subclass of a `keras.layers.Layer` that stacks outputs from several models
    to create output for an ensemble.
    """
    def call(self, X):
        return K.stack(X, axis=1)
    
    def compute_output_shape(self, input_shape):
        # assumes all input shapes are the same
        return (input_shape[0][0], len(input_shape), input_shape[0][1])


class DeepProbabilisticEnseble(keras.models.Model):
    def __init__(self, input_shape, n_classes, n_members, beta=10**-5):
        # instantiate the first member of the ensemble so we can reuse its input layer
        # with the other layers
        self.members = [ResNet(input_shape, classes=n_classes, block='basic', repetitions=[2, 2, 2, 2])]
        self.members += [ResNet((32, 32, 3), classes=10, block='basic', input_layer=self.members[0].inputs[0],
                                repetitions=[2, 2, 2, 2])
                         for _ in range(n_members-1)]
        outputs = Stack()([m.output for m in self.members])
        self.beta = beta
        super().__init__(inputs=self.members[0].inputs, outputs=outputs)
    
    @property
    def losses(self):
        """Return all of the regularization penalties for the model.
        
        Overriding this property is the easiest way to have a loss that accounts for
        weights in several layers at once in `keras`.
        """
        losses = super().losses

        # compute KL regularization
        conv_layers = [
            # kernel is index 0, bias is index 1
            [L.trainable_weights[0] for L in m.layers if isinstance(L, keras.layers.Conv2D)]
            for m in self.members]
        # currently, each sublist is a list of model layers.
        # realign these sublists to correspond to layers
        conv_layers = [[L for L in layers] for layers in zip(*conv_layers)]
        kl_regularizations = [self.beta * kl_regularization(layers) for layers in conv_layers]
        losses.extend(kl_regularizations)

        return losses
```


```python
dpe = DeepProbabilisticEnseble((32, 32, 3), 10, 8)
```


```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(dpe).create(prog='dot', format='svg'))
```


```python
dpe.summary()
```

# Simulation of experiments


```python
import keras
from keras.datasets.cifar10 import load_data
import numpy as np
from unittest import mock

(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_train.std(axis=0)

y_train = keras.utils.to_categorical(y_train, 10)
y_train = y_train[:, np.newaxis, :]
y_test = keras.utils.to_categorical(y_test, 10)
y_test = y_test[:, np.newaxis, :]
```


```python
X_train.shape, y_train.shape
```


```python
X_test.shape, y_test.shape
```


```python
from functools import partial
import gc

# the parameters below replicate the "linear-4" experiment from the paper
# with some minor adjustments and shortcuts (e.g. using "h_cat" over "h_ens"
# and not doing random crops on the images).
n_ensemble_members = 8
budget = 16000
n_iterations = 4
max_epochs = 400
patience = 25
# keras param, if non-zero this can blow up the output in the cell below
verbosity = 0

# simulation
dpe_builder = partial(DeepProbabilisticEnseble, (32, 32, 3), 10, n_ensemble_members)
b = budget // n_iterations
n_acquisitions = b
idx = np.random.choice(len(X_train), size=len(X_train), replace=False)
X_labeled, y_labeled = X_train[idx[:b]], y_train[idx[:b]]
X_unlabeled, y_unlabeled = X_train[idx[b:]], y_train[idx[b:]]

while n_acquisitions < budget:
    print('building ensemble')
    dpe = dpe_builder()
    dpe.compile(loss=ensemble_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    print('training ensemble')
    history = dpe.fit(
        X_labeled, y_labeled,
        batch_size=32,
        epochs=max_epochs,
        validation_data=(X_test, y_test),
        callbacks=[keras.callbacks.ReduceLROnPlateau(patience=25)],
        verbose=verbosity)
    print('trained for %d epochs' % len(history.history['val_loss']))
    print('validation loss:', history.history['val_loss'][-1],
          'validation accuracy:', history.history['val_acc'][-1])

    # aggregate along the individual model predictions
    print('calculating uncertainty of predictions')
    p = dpe.predict(X_unlabeled).sum(axis=1)
    h_cat = (-p * np.log(p)).sum(axis=(1, 2))  # this is "H_cat" in the paper
    idx_acquisitions = np.argsort(h_cat)[-b:]
    idx_rest = np.argsort(h_cat)[:-b]
    
    print('adding %d examples to training data' % len(idx_acquisitions))
    X_labeled = np.concatenate([X_labeled, X_unlabeled[idx_acquisitions]])
    y_labeled = np.concatenate([y_labeled, y_unlabeled[idx_acquisitions]])
    X_unlabeled = X_unlabeled[idx_rest]
    y_unlabeled = y_unlabeled[idx_rest]
    n_acquisitions += b
    print('%d labeled examples' % len(X_labeled))
    print('%d unlabeled examples' % len(X_unlabeled))

    print('releasing ensemble from GPU memory')
    K.clear_session()
    del dpe
    del history
    gc.collect()
```