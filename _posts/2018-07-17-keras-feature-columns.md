---
layout: post
mathjax: true
title: Keras Feature Columns
github: https://github.com/dantegates/keras-feature-columns
creation_date: 2018-07-17
last_modified: 2018-07-18 16:12:41
tags: 
  - keras
  - tensorflow
  - opinion
---


`tensorflow's` [feature columns](https://www.tensorflow.org/guide/feature_columns) are a great idea. Feature columns allow user's to easily transform input to `tensorflow's` premade models. For example with feature columns you can specify

- how a feature should be normalized.
- that a feature should be one-hot-encoded
- that a feature should be encoded using hashing
- that a feature should be transformed to an embedding

In my opinion feature columns provide a key feature I find missing in [keras](https://keras.io/) when working with proprietary data from day to day. `keras` has some really nice preprocessing features that are useful for hard things like image preprocessing and time series. However, I find that `keras` lacks support for some of the workhorse preprocessing functionality found in `sklearn` (such as one hot encoding and pipelines).

While it is possible to tie `sklearn` preprocessing in to `keras` models it's not exactly elegant (see [here](https://keras.io/scikit-learn-api/)). Additionally it's not straightforward to make preprocessing steps *a part* of your model - for example embedding columns. It seems that `tensorflow` is on to a great idea here.

**Disclaimer**: In this post I'll share my *opinion* of `tensorflow's` feature columns, both good and bad.

# Working with feature columns

While feature columns are a great idea that support both a preprocessing pipeline and a set of basic transformations easily applied to a variety of industry problems, I found that, unfortunately, working with feature columns is awkward if you want to do anything slightly outside the box. In my experience this is mainly due to the fact that feature columns only work with [tensorflow estimators](https://www.tensorflow.org/guide/estimators).

My first attempt at working with feature columns was to try and connect feature columns to `keras` models. Why? Because for time series applications at work I would like to have a convenient way to feed a mixture of numeric and categorical values to an LSTM. Feature columns sound like they should make the first part easy and `keras` makes training an LSTM easy. Since `tensorflow` now houses a `keras` API I thought this would be straightforward. I was wrong. The key to getting this to work is to convert your `keras` model to a `tensorflow` estimator with [tf.keras.estimator.model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator). However, actually connecting your feature columns to your `keras` model is far from trivial requiring more code than seems worth the trouble.

Taking the hint from my first stab at using feature columns, my second attempt was to stick with `tensorflow` estimators and avoid `keras` altogether. In this case I simply tried to re-implement a simple linear model I implemented for a project at work some time ago using feature columns and `tensorflow's` [linear regression estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor). In a short period of time I was able to get the model training. However, there was some outliers that the model predicted poorly on. No problem  - I already faced this in my initial implementation and knew that using huber loss would likely remedy the issue. However, after spending more time researching how to switch from the default loss function (MSE) to huber loss than I wish I had I concluded that it isn't possible to do so without writing your own custom estimator. But writing your own estimator was a deal breaker for me - for me, the whole appeal of feature columns was having something that worked out of the box.

My last comment is that it's worth noting that the web is pretty silent on how to do anything with `tensorflow` estimators outside of what you can find in the docs. This [stackoverlfow post](https://stackoverflow.com/questions/50766718/changing-loss-function-for-training-built-in-tensorflow-estimator) (accessed 7/17/18) is pretty indicative of the kind of help you'll find on the subject... nothing.

# keras feature columns

Given the outcome of this adventure I was pretty dissapointed that working with feature columns was so cumbersome considering they seem to be such a great idea. However, rather than give up on the idea, I wrote a few `keras` classes that accomplish what feature columns do. Snippets from the implementation and a toy example are below.


```python
import keras
import numpy as np


class FeatureColumn:
    def __init__(self, name):
        self.name = name
        self.input = keras.layers.Input((1,), name=self.name)

    @property
    def output(self):
        return self.input

    def transform(self, X):
        return X.values


class NumericColumn(FeatureColumn):
    def __init__(self, *args, normalizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = normalizer

    def fit(self, X):
        if self.normalizer is not None:
            self.normalizer.fit(X)
        return self

    def transform(self, X):
        if self.normalizer is not None:
            return self.normalizer.transform(X).values
        return X.values
    
    
class EmbeddingColumn(FeatureColumn):
    def __init__(self, *args, vocab_size, output_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.output_dim = output_dim

    def fit(self, X):
        self.vocab_map = {v: i for i, v in enumerate(set(X))}
        @np.vectorize
        def apply_mapping(X):
            return self.vocab_map.get(x, self.vocab_size)
        self._apply_mapping = apply_mapping
        return self

    def transform(self, X):
        return self._apply_mapping(X)

    @property
    def output(self):
        embedding = keras.layers.Embedding(
            input_dim=self.vocab_size+1,  # +1 for OOV
            output_dim=self.output_dim,
            input_length=1)(self.input)
        return keras.layers.Flatten()(embedding)


class FeatureSet:
    def __init__(self, *features):
        self.features = features

    def fit(self, X):
        self.features = [f.fit(X[f.name]) for f in self.features]
        return self

    def transform(self, X):
        return [f.transform(X[f.name]) for f in self.features]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def inputs(self):
        return [f.input for f in self.features]

    @property
    def output(self):
        concat = keras.layers.Concatenate(axis=-1)
        return concat([f.output for f in self.features])


class Scaler:
    def __init__(self):
        self.mean = self.std = None

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

```

    Using TensorFlow backend.



```python
import pandas as pd

X = pd.DataFrame({
    'feature1': np.random.randint(10, size=100),
    'feature2': np.random.randint(100, size=100),
    'feature3': np.random.rand(100)
})
y = np.random.rand(100)


def normalize(column):
    mean = X[column].mean()
    std = X[column].std()
    def normalizer(X, mean=mean, std=std):
        return (X - std) / mean


features = FeatureSet(
    EmbeddingColumn('feature1', vocab_size=10, output_dim=10),
    EmbeddingColumn('feature2', vocab_size=100, output_dim=10),
    NumericColumn('feature3', normalizer=Scaler())
)


x = keras.layers.Dense(50, activation='relu')(features.output)
x = keras.layers.Dense(50, activation='relu')(x)
x = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model(inputs=features.inputs, outputs=x)
model.summary()
model.compile(loss='mse', optimizer='adam')
_ = model.fit(features.fit_transform(X), y)
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    feature1 (InputLayer)           (None, 1)            0                                            
    __________________________________________________________________________________________________
    feature2 (InputLayer)           (None, 1)            0                                            
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, 1, 10)        110         feature1[0][0]                   
    __________________________________________________________________________________________________
    embedding_2 (Embedding)         (None, 1, 10)        1010        feature2[0][0]                   
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 10)           0           embedding_1[0][0]                
    __________________________________________________________________________________________________
    flatten_2 (Flatten)             (None, 10)           0           embedding_2[0][0]                
    __________________________________________________________________________________________________
    feature3 (InputLayer)           (None, 1)            0                                            
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 21)           0           flatten_1[0][0]                  
                                                                     flatten_2[0][0]                  
                                                                     feature3[0][0]                   
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 50)           1100        concatenate_1[0][0]              
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 50)           2550        dense_1[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 1)            51          dense_2[0][0]                    
    ==================================================================================================
    Total params: 4,821
    Trainable params: 4,821
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Epoch 1/1
    100/100 [==============================] - 0s 4ms/step - loss: 0.0808