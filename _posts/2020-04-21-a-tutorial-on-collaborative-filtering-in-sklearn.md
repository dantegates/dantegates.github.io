---
layout: post
mathjax: true
title: A Tutorial on Collaborative Filtering in sklearn
github: https://github.com/dantegates/collaborative-filtering-tutorial
creation_date: 2020-04-21
last_modified: 2020-04-21 16:54:43
tags: 
  - Recommendation Systems
  - sklearn
---


Given the [vast amount of entertainment consumed on Netflix](https://www.marketwatch.com/story/we-now-spend-more-time-on-netflix-than-we-do-bonding-with-our-kids-2018-09-13-12882032) and [amount of shopping done through Amazon](https://www.businessinsider.com/amazon-holiday-facts-2012-12) it's a safe bet to claim that *collaborative filtering* gets more public exposure (wittingly or not) than any other machine learning application.

While `sklearn` has all of the tools needed to build a collaborative filtering model, there isn't a lot of documentation on how to accomplish this with real world data. This post will try to close that gap.

# Preamble

Collaborative filtering can be used whenever a data set can be represented as a numeric relationship between *users* and *items*. This relationship is usually expressed as a *user-item* matrix, where the rows represent users and the columns represent items.

For example, a company like Netflix might use their data such that the rows represent accounts, columns movies and the values are the account's movie ratings.

A company like Amazon might express the rows as accounts, the columns as items for purchase, and the values as the number of times item $j$ was purchased by account $i$.

One of the draws of collaborative filtering is that it is such a flexible paradigm. It's very easy to extend this idea and imagine how companies like Spotify and the New York Times might define
user-item matrices for recommending music or articles.

Keep in mind that collaborative filtering is not itself a particular algorithm, but rather a class of algorithms. The distinguishing feature from other recommendation algorithms
is that collaborative filtering learns from the *latent* features in the user-item matrix rather than using explicit features such as genre, rating, article text, etc. (the latter
case is often referred to as content based recommendation).

Although this is usually this is a "big data" problem, but there's no reason your data has to be ["Twitter big"](https://blog.twitter.com/engineering/en_us/a/2014/all-pairs-similarity-via-dimsum.html)
to get benefits from this technique.

# A practical example

Let's work through an example. We'll be recommending movies using the [MovieLens dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset). If we load it up we'll notice it doesn't look a user-item matrix.


```python
import pandas as pd

df = pd.read_csv('rating.csv')
print('Number of user ratings: ', df.shape)
df.head()
```

    Number of user ratings:  (20000263, 4)





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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>2005-04-02 23:53:47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>2005-04-02 23:31:16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>2005-04-02 23:33:39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>2005-04-02 23:32:07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>2005-04-02 23:29:40</td>
    </tr>
  </tbody>
</table>
</div>



Rather than the user-item layout described above we have rows of user-ratings where the same user can appear in multiple rows. This is makes this data set a *practical* example. In practice, it's much more likely that the data is stored "transactionally" than as a big rectangular matrix and there are plenty of good reasons why. E.g.

- Typically business logic is much easier to work out on a transactional structure.
- The transactional structure allows us to preserver other features, such as the timestamp or session tokens.
- Storing the user-items would be inefficient since a given user typically only interacts with a very small fraction of the  items, i.e. the user-item matrix is usually sparse and RDMBs aren't designed to handle sparsity efficiently.

Therefore our first step is to format the data, which we can do easily using `sklearn`s `OrdinalEncoder` to transform the user and movie IDs into row/column indices from which we can easily instantiate a sparse array.


```python
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder

def encode(series, encoder):
    return encoder.fit_transform(series.values.reshape((-1, 1))).astype(int).reshape(-1)

user_encoder, movie_encoder = OrdinalEncoder(), OrdinalEncoder()
df['user_id_encoding'] = encode(df.userId, user_encoder)
df['movie_id_encoding'] = encode(df.movieId, movie_encoder)

X = csr_matrix((df.rating, (df.user_id_encoding, df.movie_id_encoding)))
print('Total size of X:', X.shape[0] * X.shape[1], '\nNumber of non-zero elements in X:', X.count_nonzero())
```

    Total size of X: 3703856792 
    Number of non-zero elements in X: 20000263


It's worth pointing out the importance of using a sparse array here. Notice how much larger the total size of the array is compared to the number of non-zero elements - which is just the number of ratings in the original `DataFrame`. Sparse arrays allow us to represent without explicitly storing the 0-valued elements. This means that if the transactional data can be loaded into memory, the sparse array will fit in memory as well.

## Generating Recommendations

Now that we have the user-item matrix there are several ways to proceed. [This post](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)
does a pretty good job of covering our options, but we'll highlight two of the most common approaches here.

1. We can generate "item-item" recommendations by computing similarity (or distance) between titles based on their user-rating representations (i.e. the columns).
2. We can generate "user-item" recommendations with matrix factorization (such as [sklearn's NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)).

In this post we'll go with the first approach, using cosine similarity to build a square similarity matrix, $V$.


```python
from sklearn.metrics.pairwise import cosine_similarity
V = cosine_similarity(X.T, X.T)
V.shape
```




    (26744, 26744)



Note that $V_{i,j}$ is the cosine of the angle between the the user-item matrix column vectors $i$ and $j$. Given a particular movie $i$, this gives us a mechanism for ranking all movies based on their similarity to $i$. We can then generate recommendations to users who watched $i$ by suggesting the most similar titles in ranked order.

Again, it's important to understand that this is done without creating any explicit features, rather our model has learned *implicit* or *latent* relationships between movies, based on user behavior and not explicit attributes such as "genre", that drive our recommendations.

Some brief commentary on matrix factorization models will also serve to emphasize this point.

A simple content based recommendation model might use the output of a simple linear model, $\hat{y}=x_{u,m}w$, to rank movies $m$ for a user $u$ by trying to predict the user rating for $m$, $\hat{y}$. Note that the predictions here depend explicitly on the feature vector $x_{u,m}$ which contains attributes directly related to the user (e.g. age) and the movie (e.g. genre).

On the other hand, if our user-item matrix is $X$, a matrix factorization model attempts to learn how to factor $X$ into the product of two matrices $UV$. This amounts to learning *both* the linear coefficients *and* the feature vector. To see this, consider that a matrix factorization model learns two mappings: one from the user-item vector $u$ to a lower dimensional vector $u^{\prime}$ and another, $V$, from $u^{\prime}$ to item space. Thus the predicted ratings are the product $u^{\prime}V$, where $u^{prime}$ can be thought of as analogous to the feature vector and $V$ to the linear coefficients. [Simon Funk's blog post](https://sifter.org/~simon/journal/20061211.html) that popularized this technique during the Netflix challenge, though lengthy, captures this intuition really well.

## Serving the recommendations

To complete this practical example, we'll give some suggestions on how to serve recommendations in a production context.

The naive approach is to replicate exactly what we did above. That is take user/movie IDs as input, transform these to the user-item matrix, do some linear algebra and return the result. If your use case accommodates generating predictions in batch this might not be a bad place to start.

If predictions need to be served in real time, however, this is extremely inefficient. Fortunately in many cases, we know before run time all of the items we need to be able to recommend. For example, before a user logs in to Netflix, it is known what titles are currently in their catalog. This situation allows the item-item similarities to be calculated and stored offline. Then, at run time predictions can simply be served from a lookup table (or the key-value store of your choice). This concept is demonstrated in the code below. For a real-world example, the data science team at the New York Times [gave a great talk](https://youtu.be/n07q-rZTLTw?t=1724) that demonstrates how they apply a similar concept at scale.


```python
movies = pd.read_csv('movie.csv')
movies.head(10)
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tom and Huck (1995)</td>
      <td>Adventure|Children</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Sudden Death (1995)</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>GoldenEye (1995)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_encoder.inverse_transform([[1]])
```




    array([[2]])




```python
import numpy as np

offline_results = {
    movie_id: np.argsort(similarities)[::-1]
    for movie_id, similarities in enumerate(V)
}

# in practice we would probably do movieIds in movieIds out, but using
# the title text here makes the example readable
def get_recommendations(movie_title, top_n):
    movie_id = movies.set_index('title').loc[movie_title][0]
    movie_csr_id = movie_encoder.transform([[movie_id]])[0, 0].astype(int)
    rankings = offline_results[movie_csr_id][:top_n]
    ranked_indices = movie_encoder.inverse_transform(rankings.reshape((-1, 1))).reshape(-1)
    return movies.set_index('movieId').loc[ranked_indices]

get_recommendations('Heat (1995)', 10)
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
      <th>title</th>
      <th>genres</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Rock, The (1996)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Twelve Monkeys (a.k.a. 12 Monkeys) (1995)</td>
      <td>Mystery|Sci-Fi|Thriller</td>
    </tr>
    <tr>
      <th>648</th>
      <td>Mission: Impossible (1996)</td>
      <td>Action|Adventure|Mystery|Thriller</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Casino (1995)</td>
      <td>Crime|Drama</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Broken Arrow (1996)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Seven (a.k.a. Se7en) (1995)</td>
      <td>Mystery|Thriller</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Usual Suspects, The (1995)</td>
      <td>Crime|Mystery|Thriller</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Fargo (1996)</td>
      <td>Comedy|Crime|Drama|Thriller</td>
    </tr>
    <tr>
      <th>780</th>
      <td>Independence Day (a.k.a. ID4) (1996)</td>
      <td>Action|Adventure|Sci-Fi|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python

```