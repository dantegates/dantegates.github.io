---
layout: post
mathjax: true
title: Monitoring Machine Learning Models in Production
github: https://github.com/dantegates/monitoring-ml
creation_date: 2018-06-12
last_modified: 2018-06-12 20:44:32
tags: 
  - Industry
  - Software
  - Machine Learning
---


Every production machine learning system is susceptible to covariate shift, when the distribution of production data predicted on at run time "drifts" from the distribution you trained on. This phenomenon can severely degrade a model's performance and can occur for a variety of reasons - for example, your company now sells new products or no longer offers old ones, the price of inventory has increased or there is a bug in the software that sends data to your model. The effect of covariate drift can be subtle and hard to detect but nevertheless important to catch. In [Google's machine learning guide](https://developers.google.com/machine-learning/rules-of-ml/) they report that refreshing a single stale table resulted in a 2% increase in install rates for Google Play. Retraining and redeploying your model is an obvious fix, but it can be ambiguous as to how often you should do this, as soon as new data arrives, every day, every week, every month? What's more this would only work for the first two examples mentioned above and can't be used to fix integration bugs.

While covariate shift is a real problem, fortunately simple monitoring practices can help you catch when this happens in real time. In this post we discuss how to implement simple monitoring practices into an existing `sklearn` workflow. Additionally we'll describe how to take these monitoring practices beyond `sklearn` by leveraging `Docker` and `AWS`.

# Stop!

If you are training and deploying models with `tensorflow` stop reading this blog post and start reading up on [tensorflow extended](https://github.com/TensorLab/tensorfx). This post simply explores how to take the suggestions from [this paper](http://delivery.acm.org/10.1145/3100000/3098021/p1387-baylor.pdf?ip=96.227.139.21&id=3098021&acc=OPENTOC&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E054E54E275136550&__acm__=1528232488_9428c653977a1be26af908c3c5b37eeb) on `tensorflow extended` and fit them into a framework where models are trained with `sklearn`.

# Monitoring

The entire [tensorflow extended paper](https://github.com/TensorLab/tensorfx) is worth reading, but the relevant sections on data validation/monitoring can be summarized in two points.

1. Keep validations simple, that is, only check simple facts about the data. Histograms suffice for numeric features and a list of unique values along with their prevalence will do for categorical features.
2. Keep insights actionable. If a categorical value is found at run time that was not in the training data the appropriate response should be clear. If the value is `state=Pennsylvania` when the model was trained on the value `state=PA` then it is clear that there is an integration issue that needs to be addressed. On the other hand if the state initials were used during training then we're seeing a new part of the population at run time that we didn't have at train time and it's likely time to retrain the model.

To further summarize, as a first pass, simply checking the mins and maxes of numeric features and checking for new values of categorical features is a good place to start.

# Implementing monitoring in an sklearn workflow

So how do we integrate these monitoring checks into an `sklearn` workflow? If you've read some of my previous posts you know that I'm a big fan of `sklearn` pipelines. My suggestion would be to implement a simple transformer that collects some of the basic statistics discussed above in a `.fit()` method and checks that the input data lines up with these facts in a `.transform()` method. Some may object to implementing such a class as a transformer, since it doesn't really transform the data, but I consider this as a valid approach for the following reasons

1. The monitoring mechanism needs to be present whenever the model is trained. This is satisfied by including an instance of our class in the pipeline.
2. Similary, the monitoring mechanism must be present whenever the model is used to make predictions on new data. This also is satisfied by including an instance of our class in the pipeline.
3. Instances of our class really do "fit" just as much as classes like [sklearn.preprocessing.Normalizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
4. Our class fits in conceptually with the general idea of a "pipeline" regardless of `sklearn`s implementation that assumes pipelines only chain together transformers.

Let's take a look at an example using some dummy data. In this example we'll see how a Transformer class that validates the min/maxes and categorical values works. If you want to see an example of implementing a transformer class see the file [monitoring.py]() or see my other post [A fast one hot encoder with sklearn and pandas](https://dantegates.github.io/A-Fast-One-Hot-Encoder/). I won't cover the details here because the implementation is trivial and boring.


```python
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from monitoring import DataMonitorTransformer
```


```python
SIZE = 100
df1 = pd.DataFrame({
    # categorical
    'feature1': np.random.randint(0, 5, size=SIZE),
    'feature2': np.random.randint(0, 5, size=SIZE),
    'feature3': np.random.randint(0, 5, size=SIZE),
    # numeric
    'feature4': np.random.uniform(0, 100, size=SIZE),
    'feature5': np.random.uniform(50, 100, size=SIZE),
    'response': np.random.randint(1, size=SIZE)
})
df2 = pd.DataFrame({
    # categorical
    'feature1': np.random.randint(0, 7, size=SIZE),  # extra values
    'feature2': np.random.randint(0, 5, size=SIZE),
    'feature3': np.random.randint(-1, 5, size=SIZE), # extra values
    # numeric
    'feature4': np.random.uniform(-10, 120, size=SIZE), # violate max/min
    'feature5': np.random.uniform(50, 110, size=SIZE),  # violate max
})
df1['feature1'] = df1.feature1.astype(str)
df2['feature1'] = df2.feature1.astype(str)
df1['feature2'] = df1.feature2.astype(str)
df2['feature2'] = df2.feature2.astype(str)
df1['feature3'] = df1.feature3.astype(str)
df2['feature3'] = df2.feature3.astype(str)
```


```python
# make pipeline and train
model = make_pipeline(DataMonitorTransformer(), DecisionTreeClassifier())
logging.basicConfig()
features = [c for c in df1.columns if c.startswith('feature')]
response = 'response'
_ = model.fit(df1[features], df1[response])
```

Now that we've fit the model, the instance of `DataMonitorTransformer` is ready to validate data when we predict. Predicting on the data in `df2` we see logs warning us that we are predicting on values we haven't seen before.


```python
_ = model.predict(df2[features])
```

    WARNING:monitoring:found 15 runtime value(s) for feature "feature4" greater than trained max.
    WARNING:monitoring:found 19 runtime value(s) for feature "feature5" greater than trained max.
    WARNING:monitoring:found 7 runtime value(s) for feature "feature4" less than trained min.
    WARNING:monitoring:found 31 categorical runtime value(s) for feature "feature1" not in training data: ['5', '6'].
    WARNING:monitoring:found 17 categorical runtime value(s) for feature "feature3" not in training data: ['-1'].



```python
# inspect the data monitor's schema
data_monitor = model.named_steps['datamonitortransformer'].data_monitor
data_monitor.schema
```




    {'categoricals': {'feature1': array(['1', '2', '0', '4', '3'], dtype=object),
      'feature2': array(['3', '2', '0', '4', '1'], dtype=object),
      'feature3': array(['0', '4', '3', '1', '2'], dtype=object)},
     'numeric': {'maxes': feature4    99.939672
      feature5    99.994736
      dtype: float64, 'mins': feature4     0.880318
      feature5    50.103870
      dtype: float64}}



# Beyond sklearn

We've now covered how to implement some simple monitoring techniques into an `sklearn` workflow, but this begs the question "what do we do with all of these logs?" This answer will vary depending on your use case, in fact your production environment may already have a solution for processing logs in place. If notthen my recomendation is to deploy your models with Docker and follow the instructions [here](https://docs.docker.com/config/containers/logging/awslogs/) to forward all logs from the docker container to Cloudwatch. Lastly you should [create alarms](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ConsoleAlarms.html) to react in real time when a production model receives data that does not pass the validations. Once the alarms are in place you can take the necessary steps as they arise.