---
layout: post
title: Monitoring machine learning models in production
mathjax: true
github: https://github.com/dantegates/monitoring-ml
---

Every production machine learning system is susceptible to covariate shift, when the distribution of production data predicted on at run time "drifts" from the distribution you trained on. This phenomenon can severely degrade a model's performance and can occur for a variety of reasons - for example, your company now sells new products or no longer offers old ones, the price of inventory has increased or there is a bug in the software that sends data to your model. The effect of covariate drift can be subtle and hard to detect but nevertheless important to catch. In [Google's machine learning guide](https://developers.google.com/machine-learning/rules-of-ml/) they report that refreshing a single stale table resulted in a 2% increase in install rates for Google Play. Retraining and redeploying your model is an obvious fix, but it can be ambiguous as to how often you should do this, as soon as new data arrives, every day, every week, every month? What's more this would only work for the first two examples mentioned above and can't be used to fix integration bugs.

While covariate shift is a real problem, fortunately simple monitoring practices can help you catch when this happens in real time. In this post we discuss how to implement simple monitoring practices into an existing `sklearn` workflow. Additionally we'll describe how to take these monitoring practices beyond `sklearn` by leveraging `Docker` and `AWS`.

# Stop!

If you are training and deploying models with `tensorflow` stop reading this blog post and start reading up on [tensorflow extended](https://github.com/TensorLab/tensorfx). This post simply explores how to take the suggestions from [this paper](http://delivery.acm.org/10.1145/3100000/3098021/p1387-baylor.pdf?ip=96.227.139.21&id=3098021&acc=OPENTOC&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E054E54E275136550&__acm__=1528232488_9428c653977a1be26af908c3c5b37eeb) on `tensorflow extended` and fit them into a framework where models are trained with `sklearn`.

# Monitoring

The entire [tensorflow extended paper](https://github.com/TensorLab/tensorfx) is worth reading, but the relevant sections on data validation/monitoring can be summarized in two points.

1. Keep validations simple, that is, only check simple facts about the data. Histograms suffice for numeric features and a list of unique values along with their prevalence will do for categorical features.
- Keep insights actionable. If a categorical value is found at run time that was not in the training data the appropriate response should be clear. If the value is `state=Pennsylvania` when the model was trained on the value `state=PA` then it is clear that there is an integration issue that needs to be addressed. On the other hand if the state initials were used during training then we're seeing a new part of the population at run time that we didn't have at train time and it's likely time to retrain the model.

To further summarize, as a first pass, simply checking the mins and maxes of numeric features and checking for new values of categorical features is a good place to start.

# Implementing monitoring in an sklearn workflow

So how do we integrate these monitoring checks into an `sklearn` workflow? If you've read some of my previous posts you know that I'm a big fan of `sklearn` pipelines. My suggestion would be to implement a simple transformer that collects some of the basic statistics discussed above in a `.fit()` method and checks that the input data lines up with these facts in a `.transform()` method. Some may object to implementing such a class as a transformer, since it doesn't really transform the data, but I consider this as a valid approach for the following reasons

1. The monitoring mechanism needs to be present whenever the model is trained. This is satisfied by including an instance of our class in the pipeline.
- Similary, the monitoring mechanism must be present whenever the model is used to make predictions on new data. This also is satisfied by including an instance of our class in the pipeline.
- Instances of our class really do "fit" just as much as classes like [sklearn.preprocessing.Normalizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
- Our class fits in conceptually with the general idea of a "pipeline" regardless of `sklearn`s implementation that assumes pipelines only chain together transformers.

Let's take a look at an example on data pulled from this [kaggle competition](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data). In this example we'll see how a Transformer class that validates the min/maxes and categorical values works. If you want to see an example of implementing a transformer class see the file [monitoring.py]() or see my other post [A fast one hot encoder with sklearn and pandas](https://dantegates.github.io/A-Fast-One-Hot-Encoder/). I won't cover the details here because the implementation is trivial and boring.


```python
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline

from monitoring import DataMonitorTransformer
```


```python
# read our data in
df = pd.read_csv('./sales_train.csv').sample(50000, random_state=0)
df.head()
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
      <th>date</th>
      <th>date_block_num</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>item_cnt_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2633591</th>
      <td>08.05.2015</td>
      <td>28</td>
      <td>31</td>
      <td>13688</td>
      <td>999.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1077216</th>
      <td>22.11.2013</td>
      <td>10</td>
      <td>25</td>
      <td>7856</td>
      <td>799.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2673709</th>
      <td>19.06.2015</td>
      <td>29</td>
      <td>42</td>
      <td>20091</td>
      <td>169.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>880503</th>
      <td>14.09.2013</td>
      <td>8</td>
      <td>15</td>
      <td>18344</td>
      <td>199.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>893057</th>
      <td>21.09.2013</td>
      <td>8</td>
      <td>31</td>
      <td>7882</td>
      <td>1390.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# encode some values as categoricals
df['shop_id'] = df.shop_id.astype(str)
df['item_id'] = df.item_id.astype(str)
df['date'] = pd.to_datetime(df.date)
df['month'] = df.date.dt.month.astype(str)
```


```python
# Do a train test split. We'll fit our data monitor class on the train
# set and validate the test set at predict time.
X_train, X_test = train_test_split(df, test_size=0.05, random_state=0)
X_train.shape, X_test.shape
```




    ((47500, 7), (2500, 7))




```python
# let's put some crazy values in the test set so we know we'll trigger
# the validator
where1 = np.random.choice([True, False], p=[0.99, 0.01], size=len(X_test))
where2 = np.random.choice([True, False], p=[0.995, 0.005], size=len(X_test))
X_test['item_cnt_day'] = np.where(where1, -1e10, X_test.item_cnt_day)
X_test['item_cnt_day'] = np.where(where2, 1e10, X_test.item_cnt_day)
```

    /Users/dgates/venvs/py3/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    /Users/dgates/venvs/py3/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      



```python
# make pipeline and train
model = make_pipeline(DataMonitorTransformer(), DecisionTreeRegressor())
logging.basicConfig()
features = ['shop_id', 'item_id', 'month', 'item_cnt_day']
response = 'item_price'
model.fit(X_train[features], X_train[response])
```




    Pipeline(memory=None,
         steps=[('datamonitortransformer', <monitoring.DataMonitorTransformer object at 0x112718d68>), ('decisiontreeregressor', DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best'))])




```python
_ = model.predict(X_test[features])
```

    WARNING:monitoring:found 2488 runtime value(s) for feature "item_cnt_day" greater than trained max.
    WARNING:monitoring:found 12 runtime value(s) for feature "item_cnt_day" less than trained min.
    WARNING:monitoring:found 192 categorical runtime values for feature "item_id" not in training data: ['3863', '1652', '16277', '10447', '10285', '2912', '18774', '17303', '15309', '8777', '8476', '19385', '15288', '2493', '28', '9211', '13019', '8953', '20419', '2572', '6160', '9742', '18047', '17579', '10995', '14669', '4007', '18930', '18782', '2124', '13458', '4627', '3389', '10036', '757', '13726', '18433', '19983', '16260', '265', '15247', '3980', '9833', '20325', '6489', '7797', '21465', '15685', '8369', '21799', '667', '10087', '6657', '19454', '12767', '5457', '11229', '16342', '1106', '19792', '2655', '7989', '8595', '8674', '8544', '19951', '21009', '7435', '3187', '18205', '11328', '1684', '11916', '16035', '18783', '17243', '2524', '4217', '10643', '15076', '22006', '20479', '9575', '1641', '17554', '16018', '2631', '18503', '8279', '16426', '14958', '1510', '7775', '17024', '6629', '845', '4530', '9879', '12357', '15606', '5876', '6207', '12661', '7939', '4063', '16929', '20536', '13640', '18266', '17157', '2576', '2101', '7392', '15103', '652', '17972', '19431', '21605', '9083', '11987', '2050', '1521', '3503', '8634', '19820', '6019', '21610', '18705', '17507', '18527', '19970', '2301', '17343', '1768', '19967', '4309', '9358', '12595', '21345', '56', '4764', '20370', '7292', '14321', '10714', '17125', '7934', '8991', '4612', '16202', '11554', '16061', '7352', '14524', '5867', '9763', '17018', '9501', '12654', '2012', '9598', '9141', '17172', '1128', '9270', '9872', '13993', '5522', '9459', '7073', '15244', '10912', '4550', '4266', '10089', '8886', '9674', '17054', '21332', '2209', '977', '17606', '21586', '619'].



```python
# inspect the data monitor's schema
data_monitor = model.named_steps['datamonitortransformer'].data_monitor
data_monitor.schema
```




    {'categoricals': {'item_id': array(['11767', '11812', '10502', ..., '3574', '6688', '2594'],
            dtype=object),
      'month': array(['12', '11', '10', '5', '1', '4', '6', '3', '7', '9', '8', '2'],
            dtype=object),
      'shop_id': array(['24', '54', '14', '7', '43', '19', '42', '29', '27', '50', '25',
             '31', '26', '22', '59', '6', '38', '30', '41', '44', '51', '46',
             '56', '28', '16', '2', '47', '57', '53', '0', '21', '52', '23',
             '35', '58', '8', '45', '17', '15', '18', '49', '55', '32', '20',
             '5', '37', '4', '40', '10', '12', '3', '13', '34', '48', '33',
             '39', '9', '11', '1', '36'], dtype=object)},
     'numeric': {'maxes': item_cnt_day    147.0
      dtype: float64, 'mins': item_cnt_day   -2.0
      dtype: float64}}



# Beyond sklearn

We've now covered how to implement some simple monitoring techniques into an `sklearn` workflow, but this begs the question "what do we do with all of these logs?" This answer will vary depending on your use case, in fact your production environment may already have a solution for processing logs in place. If notthen my recomendation is to deploy your models with Docker and follow the instructions [here](https://docs.docker.com/config/containers/logging/awslogs/) to forward all logs from the docker container to Cloudwatch. Lastly you should [create alarms](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ConsoleAlarms.html) to react in real time when a production model receives data that does not pass the validations. Once the alarms are in place you can take the necessary steps as they arise.
