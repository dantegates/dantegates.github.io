---
layout: post
mathjax: true
title: Time Series With Pandas
github: https://github.com/dantegates/pandas-timeseries-demo
creation_date: 2021-01-03
last_modified: 2021-01-03 11:30:51.169121
tags: 
  - pandas
  - Time Series
---


While [pandas](https://pandas.pydata.org/) has been a part of my daily workflow for the last 6 years, it wasn't until recently that I began to appreciate some of it's most powerful features: in particular its [timeseries](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) feature set, especially when combined with the unsung hero of [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html).

Because `pandas` provides a world of flexibility with methods like [apply()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) and the [.dt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.html) accessor I've often "rolled my own" implementation for time series calculations. Over time, however, I've learned that `pandas` has built-in features for just about every common time series task which not only greatly reduce the amount of code needed, but are often much more performant than the custom solutions passed of to `apply()`.

In this post I'd like to demonstrate how several of these features can be combined to solve a practical problem.

## Rolling sums

For the purpose of demonstration, let's suppose we are training a model on a data set containing a user's purchases over time. Without specifying a particular target variable, it's not difficult to imagine a number of use cases in which we might want to use a customer's total spend over the last 7 days as a feature.

This is a simple calculation at runtime. If our data is in a `DataFrame`, we can do the following

```python
ix = (df.purchase_date - datetime.now()).dt.days <= 7
df[ix].groupby('customer_id').cost.sum()
```

Calculating this feature at train time is not always quite as simple, especially if this feature isn't already stored with our transactional data, in which case we'll need to backfill the values with a rolling sum.

Fortunately `pandas` makes this really easy. All we need to do is set the index of our `DataFrame` to the column representing the purchase dates. Working with a [DatetimeIndex](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html) allows us to pass the window length to `df.cost.rolling()` in terms of days. No need for us to keep track of the dates ourselves or transform the purchase date column in any way. All of that is simply offloaded to `pandas`.


```python
import pandas as pd

purchase_dates = ['2020-01-01', '2020-01-02', '2020-01-05', '2020-01-15',
                  '2020-01-16', '2020-01-21', '2020-01-31']

df = pd.DataFrame({
    'purchase_date': pd.to_datetime(purchase_dates),
    'cost': 1  # set this to a simple value so we can easily spot-check
               # that it's working
})

df['total_cost_7D'] = df.set_index('purchase_date').cost.rolling('7D').sum().values
df
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
      <th>purchase_date</th>
      <th>cost</th>
      <th>total_cost_7D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-05</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-15</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-16</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-01-21</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-01-31</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Note that `'7D'` is one of many convenient [shorthand expressions](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for representing time with `pandas`.

## Indexing and label alignment

Let's take this example one step further: in a real-world scenario, our aggregation must account for the fact that purchases can be refunded over time.

Once again, at runtime, this is simple - sum all transactions less than 7 days old which haven't been refunded (assume `refund_date` is `null` for all such purchases).

```python
ix = (
    (df.purchase_date - datetime.now()).le('7D')  # note that we no longer need to use
                                                  # `.dt.days` now that we know the '7D'
                                                  # shorthand
    & df.refund_date.notnull()
)
df[ix].groupby('customer_id').cost.sum()
```

Applying this same approach to our historical data, however, introduces the risk exposing our training data to [leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)). Once again, a more nuanced solution is needed at train time.

```python
# don't use, introduces leakage
ix = df.refund_date.notnull()
df[ix].set_index('purchase_date').groupby('customer_id').rolling('7D').cost.sum()
```

Requiring our training data to reflect exactly the information known at the time and no more is a general rule that is (almost) always true. Leakage refers to violating this principle and can severely degrade a model's runtime predictions.

In the context of the following example, this means that the purchase made on January 1st should be included in the rolling sum ending on the 2nd, but not in the sum that ends on the 5th. Dropping all records with a refund date would effectively leak information about a refund that did not happen until the 3rd into the training example for the 2nd.


```python
df['refund_date'] = pd.NaT
df.loc[0, 'refund_date'] = pd.to_datetime('2020-01-03')
df.loc[4, 'refund_date'] = pd.to_datetime('2020-01-30')
df
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
      <th>purchase_date</th>
      <th>cost</th>
      <th>total_cost_7D</th>
      <th>refund_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>1</td>
      <td>1.0</td>
      <td>2020-01-03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-05</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-15</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-16</td>
      <td>1</td>
      <td>2.0</td>
      <td>2020-01-30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-01-21</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-01-31</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>



Properly accounting for this fact brings us to our next feature: [label alignment](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#vectorized-operations-and-label-alignment-with-series).

When arithmetic operations are performed on two `Series` or `DataFrame`s, e.g. `s1 + s2` or `df1 * df2`, `pandas` operates on the values by index (i.e. the `Series.index` or `DataFrame.index`), and *not* position as seen in the following example.


```python
import numpy as np
s1 = pd.Series(range(4), index=[0, 1, 2, 3])
s2 = pd.Series(range(4), index=[1, 2, 3, 4])
s3 = s1 + s2
s3
```




    0    NaN
    1    1.0
    2    3.0
    3    5.0
    4    NaN
    dtype: float64



Notice that `s3[1]` is equal to `s1[1] + s2[1]` and *not* the positional result `s1.iloc[1] + s2.iloc[1]`.


```python
assert s3[1] != s1.iloc[1] + s2.iloc[1]
```

**Aside**: *If we were to use string values for the index as shown in the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#vectorized-operations-and-label-alignment-with-series) this result would be obvious. However, because it's common to use the `DataFrame` and `Series` constructors without specifying the index, the indexed result and positional result are often the same. An unfortunate consequence of this behavior, at least in my experience, is that the usefulness of the index was not immediately obvious when learning `pandas`.*

While missing values are interpreted as `nan`s by default, calling the `.add()` method directly provides more control over how this operation is performed.


```python
s1.add(s2, fill_value=0)
```




    0    0.0
    1    1.0
    2    3.0
    3    5.0
    4    3.0
    dtype: float64



We can combine this feature with our work above to efficiently remove refunded purchases from our rolling sum. The idea is

1. Create a series, `a`, of purchase costs indexed by the purchase date.
2. Create a series, `b`, of refund amounts indexed by the refund date.
3. Subtract `b` from `a`, interpreting missing values as 0. Call the result `diff`.
4. Calculate the 7 day rolling sum on `diff`.

Note that the index of both `a` and `b` must have the same name (otherwise we will end up with a `Series` whose index is the cartesian product of `purchase_date` and `refund_date`). Additionally costs corresponding to a refund that happened outside of the 7 day rolling window are "masked" so that they are not subtracted from sums not including the corresponding purchase amount.


```python
df['cost_masked'] = (df.refund_date - df.purchase_date).le('7D').astype(int)
a = df.set_index('purchase_date').rename_axis('date').cost
b = df[df.refund_date.notnull()].set_index('refund_date').rename_axis('date').cost_masked
diff = a.subtract(b, fill_value=0).rolling('7D').sum()
diff
```




    date
    2020-01-01    1.0
    2020-01-02    2.0
    2020-01-03    1.0
    2020-01-05    2.0
    2020-01-15    1.0
    2020-01-16    2.0
    2020-01-21    3.0
    2020-01-30    0.0
    2020-01-31    1.0
    dtype: float64



Because the index of the `Series` resulting from the difference of `a` and `b` is the union of all purchase and refund dates our rolling sum has values corresponding to non-purchase dates. To extract only the values we are actually interested in, we simply index `diff` by `a.index`, and voila we now have the correct rolling sum that accounts for refunds.


```python
df['total_cost_7D_without_refunds'] = diff.loc[a.index].values
df
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
      <th>purchase_date</th>
      <th>cost</th>
      <th>total_cost_7D</th>
      <th>refund_date</th>
      <th>cost_masked</th>
      <th>total_cost_7D_without_refunds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>1</td>
      <td>1.0</td>
      <td>2020-01-03</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>1</td>
      <td>2.0</td>
      <td>NaT</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-05</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaT</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-15</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaT</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-16</td>
      <td>1</td>
      <td>2.0</td>
      <td>2020-01-30</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-01-21</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaT</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-01-31</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaT</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Putting it all together

Up to this point we've worked with a small data set containing just a single user for demonstration purposes so that we could easily verify our results. To conclude this post, we'll now complete this practical example by demonstrating how to perform the above calculations on a more realistic data set containing 2.5M records.

The only new functionality needed here is to combine the method described above with `groupby()`. If you've been following along up until this point the rest should be straightforward.


```python
import numpy as np
import pandas as pd


n_observations = 2_500_000
n_ids = 100_000
base_date = pd.to_datetime('2015-01-01')
max_time_delta = 1200 * 24 * 60 * 60

df = pd.DataFrame({
    'customer_id': np.random.randint(n_ids, size=n_observations),
    'purchase_date': base_date + pd.to_timedelta(np.random.randint(max_time_delta, size=n_observations), unit='s'),
    'cost': np.random.rand(n_observations) * 1000
})

df['refund_date'] = np.where(
    np.random.rand(len(df)) > 0.02,
    pd.NaT,
    df.purchase_date + pd.to_timedelta(np.random.randint(30 * 24 * 60 * 60, size=len(df))))
df['refund_date'] = pd.to_datetime(df.refund_date)

```


```python
%%time

df['cost_masked'] = (df.refund_date - df.purchase_date).le('7D').astype(int)

# `a` and `b` must be indexed by both purchase_date and customer_id for proper label alignment
a = df.set_index(['purchase_date', 'customer_id']).rename_axis(['date', 'customer_id']).cost
b = df[df.refund_date.notnull()].set_index(['refund_date', 'customer_id']).rename_axis(['date', 'customer_id']).cost_masked

diff = a.subtract(b, fill_value=0)

# We need to remove customer_id from the index so we can use it as our
# group by column.
# On the other hand, the date needs to stay in the index in order to perform the
# rolling sum using '7D'
diff.reset_index(level=1).groupby('customer_id').rolling('7D').sum()

# pull out the values corresponding to each (`customer_id`, `purchase_date`)
diff.loc[a.index].shape
```

    CPU times: user 3min 34s, sys: 1.8 s, total: 3min 36s
    Wall time: 3min 36s





    (2500000,)



Notice that this calculation completes in under 4 minutes. As a bench mark I wrote a naive solution to calculate the rolling sum and passed it off to `apply()`. According to [progress_apply()](https://tqdm.github.io/docs/tqdm/#pandas) this solution was going to take around 30 minutes to complete. I killed it after 2.

# Appendix

For further reference I recommend this [10 minute introduction to pandas](https://www.youtube.com/watch?v=_T8LGqJtuGc) by Wes McKinney, the author of `pandas`. In fact, this is the video that brought to my attention just how useful indexes can be when used properly.

Additionally, here are a few more examples showing how to extend the concepts in this post to other common tasks.

**Comparing dates**
```python
# before
df[(df.date1 - df.date2).dt.days.lt(1)]

# after: using DateOffset
df[(df.date1 - df.date2).lt('1D')]
```

**Selecting a range of dates**
```python
# before
df[df.date.between('2020-01-01', '2020-02-31')]

# after: slicing a TimeStamp index
df.set_index('date').loc['2020-01-01':'2020-01-31']
```

**Merging before a calculation**

```python
# before
d = df1.merge(df2, on=cols)
d.col1 * d.col2

# after: using label alignment
df1.set_index(cols).col1 * df.set_index(cols).col2
```

**Daily statistics**
```python
# before
df['day'] = df.date.dt.date
df.groupby('day')[col].sum()

# after: using resampling and DateOffset
df.set_index('date').resample('D').sum()
```