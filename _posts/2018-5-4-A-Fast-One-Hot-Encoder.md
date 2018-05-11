---
layout: post
title: A fast one hot encoder with sklearn and pandas
mathjax: true
github: https://github.com/dantegates/fast-one-hot-encoder
---

If you've worked in data science for any length of time, you've undoubtedly transformed your data into a one hot encoded format before. In this post we'll explore implementing a *fast* one hot encoder with  [scikit-learn](http://scikit-learn.org/stable/) and [pandas](https://pandas.pydata.org/).

# sklearn's one hot encoders

`sklearn` has implemented several classes for one hot encoding data from various formats (`DictVectorizer`, `OneHotEncoder` and `CategoricalEncoder` - not in current release). In this post we'll compare our implementation to `DictVectorizer` which is the most natural for working with `pandas.DataFrame`s.

## The pros of DictVectorizer

`DictVectorizer` has the following great features which we will preserve in our implementation

1. Works great in `sklearn` pipelines and train/test splits.
    - If feature was present during train time but not in the data at predict time
      `DictVectorizer` will automatically set the corresponding value to 0.
    - If a feature is present at predict time that was not at train time it is not
      encoded.
2. Features to be encoded are inferred from the data (user does not need to specify this).
    - This means numeric values in the input remain unchanged and `str` fields are
      encoded automatically.
3. We can get a mapping from feature names to the one hot encoded transformation values.
    - This is useful for looking at coefficients of feature importances of a model.

## The cons of DictVectorizer

`DictVectorizer` has two main blemishes (as related to a specific but common use case, see disclaimer below).

1. Transforming large `DataFrame`s is **slow**.
2. Transforming large `DataFrame`s sometimes results in `MemoryError`s.
2. The `.fit()` and `.transform()` signatures do not accept `DataFrame`s. To use `DictVectorizer`
    a `DataFrame` must first be converted to a `list` of `dict`s (which is also slow), e.g.
    
```python
    DictVectorizer().fit_transform(X.to_dict('records'))
```
    
Our implementation will guarantee the features of `DictVectorizer` listed in the pros section above and improve the conds by accepting a `DataFrame` as input and vastly improving the speed of the transformation. Our implementation will get a boost in performance by wrapping the super fast `pandas.get_dummies()` with a subclass of `sklearn.base.TransformerMixin`.

Before we get started let's compare the speed of `DictVectorizer` with `pandas.get_dummies()`.

## An improved one hot encoder

Our improved implementation will mimic the `DictVectorizer` interface (except that it accepts `DataFrame`s as input) by wrapping the super fast `pandas.get_dummies()` with a subclass of `sklearn.base.TransformerMixin`. Subclassing the `TransformerMixin` makes it easy for our class to integrate with popular `sklearn` paradigms such as their `Pipeline`s.

## Disclaimer

Note that we are specifically comparing the speed of `DictVectorizer` for the following use case only.

1. We are starting with a `DataFrame` which must be converted to a list of `dict`s
2. We are only interested in dense output, e.g. `DictVectorizer(sparse=False)`

# Time trials

Before getting started let's compare the speed of `DictVectorizer` with `pandas.get_dummies()`.


```python
# first create *large* data set

import numpy as np
import pandas as pd

SIZE = 5000000

df = pd.DataFrame({
    'int1': np.random.randint(0, 100, size=SIZE),
    'int2': np.random.randint(0, 100, size=SIZE),
    'float1': np.random.uniform(size=SIZE),
    'str1': np.random.choice([str(x) for x in range(10)], size=SIZE),
    'str1': np.random.choice([str(x) for x in range(75)], size=SIZE),
    'str1': np.random.choice([str(x) for x in range(150)], size=SIZE),
})
```


```python
%%time
_ = pd.get_dummies(df)
```

    CPU times: user 4.36 s, sys: 755 ms, total: 5.11 s
    Wall time: 5.12 s


As we can see, `pandas.get_dummies()` is fast. Let's take a look at `DictVectorizer`s speed.


```python
%%time
from sklearn.feature_extraction import DictVectorizer
_ = DictVectorizer(sparse=False).fit_transform(df.to_dict('records'))
```

    CPU times: user 1min 29s, sys: 15.5 s, total: 1min 45s
    Wall time: 6min 40s


As we can see `pandas.get_dummies()` is *uncomparably* faster. It's also informative to notice that although there is some overhead from calling `to_dict()`, `fit_transform()` is the real bottleneck.


```python
%%time
# time just to get list of dicts
_ = df.to_dict('records')
```

    CPU times: user 1min 5s, sys: 997 ms, total: 1min 6s
    Wall time: 1min 17s


# Implemention


```python
import sklearn


class GetDummies(sklearn.base.TransformerMixin):
    """Fast one-hot-encoder that makes use of pandas.get_dummies() safely
    on train/test splits.
    """
    def __init__(self, dtypes=None):
        self.input_columns = None
        self.final_columns = None
        if dtypes is None:
            dtypes = [object, 'category']
        self.dtypes = dtypes

    def fit(self, X, y=None, **kwargs):
        self.input_columns = list(X.select_dtypes(self.dtypes).columns)
        X = pd.get_dummies(X, columns=self.input_columns)
        self.final_columns = X.columns
        return self
        
    def transform(self, X, y=None, **kwargs):
        X = pd.get_dummies(X, columns=self.input_columns)
        X_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(X_columns)
        for c in missing:
            X[c] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]
    
    def get_feature_names(self):
        return tuple(self.final_columns)
```


```python
%%time
# let's take a look at its speed
get_dummies = GetDummies()
get_dummies.fit_transform(df)
```

    CPU times: user 8.83 s, sys: 647 ms, total: 9.47 s
    Wall time: 9.53 s


As we can see the GetDummies implentation has slowed down a bit from the original `pandas.get_dummes()` due to the overhead of making sure it handles train/test splits correctly, however its still super fast (and that overhead is dependent on the number of columns *not* rows, i.e. we don't have to worry about scaling `GetDummies` to larger `DataFrame`s).

Let's also take a look at some of its other features.


```python
# GetDummies works in sklearn pipelines too
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
model = make_pipeline(
    GetDummies(),
    DecisionTreeClassifier(max_depth=3)
)
model.fit(df.iloc[:100], np.random.choice([0, 1], size=100))
```




    Pipeline(memory=None,
         steps=[('getdummies', <__main__.GetDummies object at 0x7fbd08062dd8>), ('decisiontreeclassifier', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))])




```python
# you can also pull out the feature names to look at feature importances
tree = model.steps[-1][-1]
importances = tree.feature_importances_
std = np.std(tree.feature_importances_)

indices = np.argsort(importances)[:10:-1]
feature_names = model.steps[0][-1].get_feature_names()

print("Feature ranking:")
for f in range(len(feature_names[:10])):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
```

    Feature ranking:
    1. feature int2 (0.472777)
    2. feature float1 (0.270393)
    3. feature str1_1 (0.141982)
    4. feature int1 (0.114848)
    5. feature str1_147 (0.000000)
    6. feature str1_139 (0.000000)
    7. feature str1_14 (0.000000)
    8. feature str1_140 (0.000000)
    9. feature str1_141 (0.000000)
    10. feature str1_142 (0.000000)



```python
# train/test splits are safe!
get_dummies = GetDummies()

# create test data that demonstrates how GetDummies handles
# different train/test conditions
df1 = pd.DataFrame([
    [1, 'a', 'b', 'hi'],
    [2, 'c', 'd', 'there']
], columns=['foo', 'bar', 'baz', 'grok'])
df2 = pd.DataFrame([
    [3, 'a', 'e', 'whoa', 0],
    [4, 'c', 'b', 'whoa', 0],
    [4, 'c', 'b', 'there', 0],
], columns=['foo', 'bar', 'baz', 'grok', 'new'])

get_dummies.fit_transform(df1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar_a</th>
      <th>bar_c</th>
      <th>baz_b</th>
      <th>baz_d</th>
      <th>grok_hi</th>
      <th>grok_there</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 1. the new values of 'e' and 'whoa' are not encoded
# 2. features baz_b and baz_d are both set to 0 when no suitable
#      value for baz is found
get_dummies.transform(df2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar_a</th>
      <th>bar_c</th>
      <th>baz_b</th>
      <th>baz_d</th>
      <th>grok_hi</th>
      <th>grok_there</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

I recommend using the `GetDummies` class as needed only. `sklearn` is a true industry standard and deviations thereof should occur only be when the size of the data necessitates such a change. For most modest sized data sets `DictVectorizer` has served me well. However, as the size of your data scales you may find yourself waiting long periods of time for DV to finish or experience DV spitting out memory errors (in fact, I originally set `SIZE=20000000` in the code above, `get_dummies()` ran in ~90s but DV crashed my kernel twice).
