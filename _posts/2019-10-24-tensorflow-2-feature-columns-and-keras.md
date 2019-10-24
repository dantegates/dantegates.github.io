---
layout: post
mathjax: true
title: Tensorflow 2 Feature Columns and Keras
github: https://github.com/dantegates/keras-feature-columns
creation_date: 2019-10-24
last_modified: 2019-10-24 16:31:11
tags: 
  - keras
  - tensorflow
  - tensorflow2
  - feature columns
---


[tensorflow 2.0](https://www.tensorflow.org/guide/effective_tf2) was just recently introduced and one of the most anticipated features, in my opininion, was the revamping its feature columns. [Last July](https://dantegates.github.io/2018/07/17/keras-feature-columns.html) I published a blog post which was enthusiastic about the idea of `tensorflow's` [feature columns](https://www.tensorflow.org/guide/feature_columns) but disappointed by the actual implementation. Mainly because they weren't compatible with `keras` even though `tensorflow` had already adopted the `keras` API.

Fortunately the implementation I posted about last July is, well, so last year, and `tensorflow 2.0` introduces some considerable improvements.

In this post we'll demonstrate how to use the new feature columns with `keras` Sequential and functional APIs.

# Feature Columns and the Sequential API

Working with tensorflow feature columns and the Sequential API is pretty straightforward. [The official documentation](https://www.tensorflow.org/tutorials/structured_data/feature_columns) contains some easy to follow examples. Below we'll basically recreate an example from the documentation, however we will demonstrate how to work the example with `pandas` instead of `tensorflow` `DataSet`s.

The basic workflow is simple

- Create a feature column for each feature you want to train on.
- Use `keras.layers.DenseFeatures` to wrap the feature columns together.
- Include the `DenseFeatures` object as the first layer in the Sequential model.

In code this amounts to

```python
import tensorflow.feature_columns as fc

feature_price = fc.numeric_column('price'),  # same name as column in `df`
feature_product = fc.embedding_column(
    fc.categorical_column_with_vocabulary_list(
        'product_id', df.product_id.unique()
    )
)
...

feature_layer = tf.keras.layers.DenseFeatures([feature_price, feature_product])
model = tf.keras.Sequential([
    feature_layer,
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(...)

# use `pandas.DataFrame.to_dict('series')` to get a dictionary of feature names
# to values as input to training and predicting.
model.fit(df.to_dict('series'), ...)
```

## Comments

Yes, handling categorical columns can be a little awkward. It takes *at least* two function calls and five underscores to create an embedding column. `tensorflow` has never been afraid of verbosity.

Another implementation detail worth pointing out is that `DenseFeatures` layers won't infer shapes until they have a) been called with a `keras` `Input` layer or actual data (i.e. at `model.fit`). This means that in the example above `model.summary()` would raise an `Exception` if called before `model.fit()` (because the layer has not been built yet). This will come up again when we look at the functional API but it's probably worth noting now since it's likely suprising behavior for users who are familiar with the standard `keras` implementation.

# Feature Columns and the keras functional API

If you are reading this section I'm going to assume that you are already familar with `keras` [functiona API](https://keras.io/getting-started/functional-api-guide/).

The key to understanding how to use feature columns with the functional API boils down to this: the object created by

```python
DenseFeature([<feature columns here>])
```

is exactly analogous to

```python
Dense(32, ...)
```

This means that you *must* call all `DenseFeature` objects on a `Tensor` object before connecting them to other layers in your model.

For example this code

```python
feature_layer = DenseFeature([<feature columns here>])
dense_layer = Dense(32)(feature_layer)
```

will raise an `Exception`. However, 

```python
feature_layer = DenseFeature([<feature columns here>])(feature_inputs)
dense_layer = Dense(32)(feature_layer)
```

will not, where `feature_inputs` is a dictionary of feature names to `keras.layers.Input`s.

## Concrete example

Let's implement a simplified version Kaggle's Web Traffic competition's [winning solution](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795) as a concrete example.

The idea is to build an LSTM whose input vectors are concatenations of

- A one hot encoding representing the type of agent for a web page visit (e.g. spider)
- A one hot encoding representing the type of access for a web page visit (e.g. desktop)
- An embedding representing the wikipedia project visited
- Log of total page visits at that time step

For this excercise we'll limit the number of time steps input to the LSTM to a fixed window of 30 days.

Let's start with the categorical columns, first we need to create the feature columns and corresponding feature layer.

```python
static_feature_columns = [
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'agent', X.agent.unique())),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'access', X.access.unique())),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'project', X.project.unique()), 3),
]

static_layer = tf.keras.layers.DenseFeatures(static_feature_columns)
```

Now we need to create `Input` layers connect `static_layer` to them to get a tensor we can connect to future layers in the model.

```python
static_feature_inputs = {
    'agent': tf.keras.Input((1,), dtype=tf.dtypes.string, name='agent'),
    'access': tf.keras.Input((1,), dtype=tf.dtypes.string, name='access'),
    'project': tf.keras.Input((1,), dtype=tf.dtypes.string, name='project'),
}

static_inputs = static_layer(static_feature_inputs)
```

We'll do the same for each of the 30 timesteps, creating one `numeric_column` per timestep. Note that there may be a cleaner way to do this but it sets us up for a good example of how to concatenate these features later.

```python
pageview_feature_columns = [
    tf.feature_column.numeric_column(
        f'pageviews_{i}', normalizer_fn=tf.math.log1p)
    for i in range(30)]

pageview_input_layers = [
    tf.keras.Input((1,), name=f'pageviews_{i}')
    for i in range(30)
]
```

Finally, concatenate the "static" features alongside the timesteps to get the sequence input for the LSTM.

```python
sequences = [tf.keras.layers.Concatenate()([static_inputs, L]) for L in pageview_layers]
```

The rest of the model-building code is typical `keras` code I assume the reader is familiar with


```python
concat = tf.keras.layers.Concatenate()
reshape = tf.keras.layers.Reshape((-1, 30))
lstm_input = reshape(concat(sequences))
lstm = tf.keras.layers.LSTM(512)(lstm_input)
output = tf.keras.layers.Dense(1)(lstm)
```

Similarly, instantiating the model also follows the typical functional API patterns. Pass all of the inputs and outputs to `keras.models.Model`.

```python
inputs = list(static_feature_inputs.values()) + pageview_input_layers
model = tf.keras.models.Model(inputs=inputs, outputs=output)
```

Model training is now exactly the same as described during the section on the Sequential API.

# Summary

Unlike [my earlier post](https://dantegates.github.io/2018/07/17/keras-feature-columns.html) on `tensorflow` feature columns I'm more optimistic about the current implementation. `tensorflow2` contains substantial improvements over the version 1 implementation, mainly, in my opinion, plug-and-playability with `keras`.

As of the time of this writing (Oct. 24, 2019) here's the full list of pros and cons as I see them

**Pros**
1. Plug and play with `keras`.
2. Works nicely with data-frame like data sets, such as `pandas`.
3. Couples feature transformations with the model object, eliminating the need for multiple objects or pipelines at runtime (e.g. a Transformer and Predictor).
4. Supports a broad range of typical categorical feature transformations.
5. Removes some boiler plate code - no need to concatenate multiple embedding layers for example.

**Cons**
1. Can't access learned embeddings, one-hot-encodings, etc. (at least if you can it's not well documented yet).
2. `len('categorical_column_with_vocabulary_list'.split('_')) == 5`
3. The documentation seems to still be a work in progress. For example, official feature columns are [documented](https://www.tensorflow.org/api_docs/python/tf/feature_column/sequence_categorical_column_with_hash_bucket) but are only compatible with [experimental features](https://www.tensorflow.org/api_docs/python/tf/keras/experimental/SequenceFeatures).
4. Sequential feature columns are not yet supported, although they look like they are under development.
5. Leaves some boiler plate that could likely be refactored out. For example

    - It's not hard to imagine the two function calls to create a one-hot encoding feature column being only one call.
    - Or, perhaps, a single function call could create an create `Input` and `Embedding` layer and return the corresponding tensor connecting the two. This suggestion may have its own drawbacks, and perhaps this is why feature columns were not implemented this way, but it could offer an even more user friendly interface - especially since we currently have to specify the shape and data type when creating the `Input` layers.