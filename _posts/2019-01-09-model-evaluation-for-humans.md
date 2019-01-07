---
layout: post
mathjax: true
title: Model Evaluation For Humans
github: https://github.com/dantegates/model-evaluation-for-humans
creation_date: 2019-01-09
last_modified: 2019-01-06 19:08:39
tags: 
  - data science
  - machine learning
  - industry
  - model evaluation
---


# All models are wrong

There is a well known saying in machine learning: ["all models are wrong, some are useful"](https://en.wikipedia.org/wiki/All_models_are_wrong). An expression of something very familiar to us as data scientists - that even though models are (by definition) wrong, some yield huge benefits in practice while others fail spectacularly. Our goal is to deliver the former while avoiding [the latter](https://medium.com/syncedreview/2017-in-review-10-ai-failures-4a88da1bdf01) and toward this end robust model evaluation can make all the difference.

Observing a model's error on a test set is only a small part of our evaluations. The rest consists of a proper understanding of the broader context of the problem at hand and determining if the results we find on a held out set will generalize to production (where we realize the usefulness of a model). This post aims to discuss generic guidelines and practices for model evaluation that can help deliver useful models quickly and confidently.

# Model evaluation throughout the product lifecycle

For the purpose of this post, we'll discuss model evaluation as it relates throughout three stages of a project's life cycle: *before development*, *during development* and *deployment*.

As a reference point we can relate these categories to [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining), for those who are familiar, as follows.

**Before development**: Business and data understanding  
**Development**: Data preparation, modeling, evaluation  
**Deployment**: Model deployment, monitoring, evaluation

# Before development

Model evaluation should be part of the conversation even before development begins. With respsect to model evaluation, the objective of this phase should be to emerge with a clearly defined goal that can be measured once a model has been released. This phase is usually the most context dependent phase and thus the most difficult to offer concrete advice for. Nevertheless, some suggestions to keep in mind during the early stages of a project are below. Keep in mind these are just a few examples not an exhaustive list.

## Think about framing the problem

How a problem is framed will determine how a model can and/or should be evaluated as well as how it cannot and/or should not be evaluated. Getting this step right is crucial to a project's success.

Simply asking questions like "How will this model be evaluated?" and "How will we determine it is successful?" are obvious, but effective and necessary, places to start.

If you can't give a specific answer to these questions, such as "improve $x$ (some business objective) by $y$ (some measurable quantity)", then the problem hasn't been properly defined. Reframe the problem until you have a clear answer. A carefully framed problem will help avoid common pitfalls such as

- Working on a model that doesn't solve or only indirectly solves the actual problem of interest.
- Making lots of minor changes or adding complexity to a model that yield negligible results in terms of business value.
- Attempting to solve a problem that is not appropriate for data science.

In the past I've noticed that one of the biggest values a data science team can add to a company is helping to frame problems in ways suitable for machine learning and such that the model's value can be directly measured.

### Choose your metrics carefully

When thinking about how you will ultimately evaluate the usefulness of a model, choose metrics with care. While you may want to evaluate your model against several metrics, whenever possible, see to it that metrics chosen during this phase as a KPI or acceptance criteria can easily be converted into business value. This, and other considerations, are discussed further in the appendix at the end of this post. Additionally the section [Your First Objective](https://developers.google.com/machine-learning/guides/rules-of-ml/#your_first_objective) in Google's Rules of Machine Learning guide contains helpful recommendations.

## Think about deployment and future iterations

Later on in this post are recommendations of methods for evaluating deployed models such as monitoring and A/B testing. In addition to these, you may have a few ideas of your own as well. Before development begins, start thinking about the requirements you will need for these post deployment evaluations.

For example, suppose you want to perform A/B tests to evaluate whether future iterations of your model should be deployed and that model predictions are exposed behind a REST API. This will have implications on infrastructure and the API itself (for example a model ID may need to be returned to the user in the response alongside the prediction). These infrastructure requirements, in turn, may have their own implications on latency or memory requirements. Accounting for this during the initial iteration will make it easier to introduce A/B tests during future iterations.

## Think about the feedback loop

Lastly, start thinking about a feedback loop during the business and data understanding phases. How will outcomes from the model be collected and evalauted? Will the model's use case affect how it will be evaluated and/or the training data available in future iterations?

This second point is incredibly important to consider and has been true of a large number of projects I've personally worked on. For illustration, examples of this interplay between model use case and evaluations follow.

Consider a model that recommends which ad to show a user. At run time ads which the user is most likely to click are predicted but only one can be shown. Thus how the model is used within the product creates a bias in the data available in the future with implications for training and evaluation in subsequent iterations.

Now suppose a hospital deploys a model that predicts the likelihood a patient will be readmitted to the hospital within 60 days. The hospital plans to implement some sort of intervention for the patients that the model predicts a high risk of readmittance. The key thing to notice here is that the very nature of the intervention is to change the predicted label ("will be readmitted" to "will not"). If a patient receives the intervention because the model identified them as high risk and the patient is not readmitted does this mean the model was incorrect? Additionally, because of the ethical issues involved simple approaches that might work with ad recommendations (such as randomly holding out data) might not be acceptable here. This doesn't mean the model can't be evaluated, but it will be very difficult to evaluate it directly. Evaluating the model indirectly by looking for a decline in readmission rates, for example, might be the only option available.

# During Development

## Test your assumptions

Before ever calling `model.fit()` you should test your assumptions. This is especially true when working with an unfamiliar data source. More than once I've gotten lost down the rabbit hole of trying to get a model to fit to a data set only to realize later that there was a fundamental issue in the training data.

Check the distributions in your data. Plot, plot, plot. Look at ranges and means. Validate the data against what subject matter experts have to say. If the data disagrees with conventional wisdom account for the discrepencies. Inspect outliers. Etc.

## Establish a baseline

Once the data is collected and you have tested your assumptions, establishing some sort of baseline, is a great next step - especially if this is the first iteration of your model. The idea is to quickly establish a minimum criteria that a model must exceed to be considered useful. A baseline should be easy to understand and easy to implement. Baselines that take a long time to implement or that yield results you are unsure have less value.

One of the most useful advantages I've found of having a baseline is that it helps keep me on track and prevents wasting cycles unnecessarily fine tuning a model. Once a model surpasses the baseline you can begin finalizing it and moving forward to deployment.

Here are some suggestions of what to use as a baseline.

### A simple heuristic

A simple heuristic can serve as an excellent baseline and there's a good chance if you're trying to improve an existing product that one is already being used. Integrating a model into your product comes with development and maintenance costs. If your model can't beat the heuristic you may need to revisit the business and data understanding steps. On the other hand, if your model outperforms the heuristic you know you are on the right track and as a bonus you have a clear comparison to use when explaining results to business stakeholders.

### An interpretable model

Sure, deep learning is all of the rage right now and it's fun to build deep neural networks. However, starting with a simple model with parameters you can interpret (e.g. linear regression, random forest, a probabalistic model) will help give you confidence that you are on the right track. Inspecting the parameters a model is learning also helps you test your assumptions and serves as a good sanity check (see sections above and below).

### The existing model or methodology

If you already have a model, or some other method for making predictions in production, this is an obvious baseline. You wouldn't want to roll out a new model that isn't any better than what is already in place.

As an aside, I would recommend, from my own experience that if the model is old, hasn't been updated or was implemented by another team or someone no longer working for the company not to assume the production model is an adequate baseline. In these cases the model very well might not beat a simple heuristic or model and you should use whichever method is best as your baseline.

### Acceptance criteria

Perhaps you know that any model with a precision less than $x$ won't be useful. Then this metric should be used as a baseline. If another baseline mentioned above outperforms this criteria than obviously that baseline's performance supercedes this.

## Sanity checks

Incorporating sanity checks as part of your modeling workflow can help catch issues in your data or training code.

One particularly useful and simple sanity check to incorporate is making a habit of inspecting the learned coefficients. Has anyone else ever discovered that response variable was left in the training set as a feature this way? I've also seen a cases where features were unexpectedly ranked at the top of the [feature importances](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html). After looking further into why this was the case it was realized that these features were leaking information about the target variable.

Another important sanity check is to validate any algorithm specific assumptions that your model might have. For example if you are using a probabalistic model you need to check it's error on a test set per ML-101. However you should also perform [posterior predictive checks](https://docs.pymc.io/notebooks/posterior_predictive.html) to be sure that the model is logically consistent with its own assumptions. If this assumption fails, you should be dubious of counting on the performance observed on the test set generalizing into production.

These checks also might be useful later on during deployment (see section below) which is another reason to begin using them early in the development cycle.

## Cross validation

Cross validation is a ubuquitous technique for evaluating how well a model will generalize to new data. Typically cross validation is introduced as the [$K$-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html), leave one out pattern.

In industry however, an approach more similar to [backtesting](https://en.wikipedia.org/wiki/Backtesting) may represent the production workflow more closely.

To give a concrete example, suppose you intend to train a model on the most recent month of data and use that model to predict during the next month. Your evaluations should simulate how this would work in production. Pull your historical data, train on January and predict on February. Then train on February predict on March, etc. This prevents "leakage" of future information in your training data and is a more accurate representation of how your deployment strategy will do in production. The following code snippet demonstrates how you might implement this simple case with a [pandas](https://pandas.pydata.org/) DataFrame.

```python
months = list(df.groupby('month'))
for (m1, df_train), (m2, df_test) in zip(months, months[1:]):
    X_train, y_train = df_train[features], df_train[response]
    X_test, y_test = df_test[features], df_test[response]
```

## Train on a cleaner data set

Eventually you may want to attempt solving a problem with a more complex model or perhaps an algorithm that you don't have much experience with. Suppose you begin down this path and experience difficulty getting the model to fit. At this point you may be wondering if the issue is with the model or the data.

In this case it may be helpful to validate your approach on a cleaner data set if you have one available. It will be easier to trouble shoot why the model is not fitting when you have more confidence that the data itself is not the source of the problem. Once the model fits to the cleaner data you can try your approach on the original data set.

# Deployment and beyond

After deploying a model to production its standard practice to evalaute the model by comparing predictions to outcomes. Below I'd like to discuss two other types of model evaluation you can incorporate post-deployment to help ensure that your model is (and stays) useful.

## A/B Testing

When you have a new update or iteration of a model, how can you be confident that it will outperform the existing model? [A/B testing](https://en.wikipedia.org/wiki/A/B_testing) is a great practice to help you deploy with confidence.

One particularly useful pattern is to deploy the new iteration alongside the current method (this could be a model or heuristic) and begin directing some small percentage of traffic to the new model. After obseving a number of predictions and outcomes an A/B test can be performed to compare the new iteration to the old. If the result favors the new model then the amount of traffic handled by the new model can be gradually ramped up until it handles all of the traffic.

A/B tests, and more generally hypothesis tests, are helpful for addressing other questions as well such as

- How does production performance compare to performance during training?
- Is this model's production performance declining?

For more information on hypothesis testing with Python, take a look at one of my previous posts [hypothesis testing for humans](https://dantegates.github.io/2018/09/17/hypothesis-testing-for-humans-do-the-umps-really-want-to-go-home.html).

For extra credit, you can consider similar, but more advanced, methods such as [multi armed bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit).

## Monitoring

As this post has already acknowledged all models are wrong. Thus it's simply not possible to prevent bad predictions from ever being exposed to users. However, the frequency at which this happens can be minimized and prevented as much as possible.

Monitoring, which I take to mean as any process which inspects a production environment for potential "hazards" that can cause bad predictions to leak through to the user, is a necessary component for any robust system. Below are a few things to keep in mind when choosing or implementing a platform, framework, practices, etc. for deployment. Again, as with the rest of this post, keep in mind that these are not comprehensive lists.

### Covariate shift

One of the most common preventable causes of predictions going bad in production is [covariate shift](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4921&rep=rep1&type=pdf) - when the distribution of data at run time and the distribution of data during training differ enough that the quality of a model's predictions suffers.

The most obvious place to start is to monitoring the data at run time for deviations from what was observed during training. There is [a paper on tensorflow extended](http://stevenwhang.com/tfx_paper.pdf) that contains a lot of useful ideas on this topic. My two favorites are to keep monitoring simple and alerts actionable. This is described nicely in the following quote

    Each anomaly should have a simple description that helps
    the user understand how to debug and fix the data. One
    such example is an anomaly that says a feature’s value is
    out of a certain range. As an antithetical example, it is
    much harder to understand and debug an anomaly that
    says the KL divergence between the expected and actual
    distributions has exceeded some threshold.
    
One of my [previous blog posts](https://dantegates.github.io/2018/06/12/Monitoring-machine-learning-models-in-production.html) discusses some simple ways to get started with monitoring with [sklearn](https://scikit-learn.org/stable/).

### Sanity checks

Another key item to monitor are sanity checks. This is especially true if you are doing automated deployments. Some of the sanity checks described in the Development section above should be included in your automated training code. Other simple checks could be checking evaluations on a held out set or checking for anomalous predictions (for example very large or very small values).

# Appendix

## The zen of model evaluation

This post was the basis of a talk I gave at [PyData Miami](https://pydata.org/miami2019/). Since I didn't have any code to share during the talk I felt the need to incorporate some other Pythonism into the talk. The result was bestowing the label "The Zen of Model Evlaution" to the George Box quote above:  "All models are wrong, some are useful." I summarized the quote into several bullet points listed below.

- All models are wrong
- Some are useful
- Others are not
- Production matters

## Metrics

In practice not all metrics are equally useful. Some guidelines I like to follow are described below.

(Keep in mind this discussion is of metrics *not* loss functions. Cross-entropy and gini importance are loss functions - bleu and precision are metrics. Some functions, such as mean squared error, can be both.)

### Conversion to business value

As briefly mentioned above, one guideline worth following is that metrics chosen as KPIs should easily translate into business value.

Take a regression model that predicts bid rates as an example. Mean absolute percent error can be converted to business value with a statement such as "our predicted bid prices are within $\pm x$% of the winning bid price" while R-squared cannot.

Similarly consider a model that classifies transactions as fraudulent or not. Along with the known costs of allowing a fraudulent transaction to happen and investigating a transaction for fraudulency, precision and recall can be expressed in terms of business value with the statement "By identifying 10% of the fraudulent transactions on our platform (recall) and being correct 90% of the time (precision) we reduce cost from fraudulent transactions by $x$ dollars." Making a statement like this is much more difficult, if not impossible, with metrics such as accuracy or AUC.

### Comparability

Useful metrics should be campatible across time, data sets and iterations.

As an example of a metric that fails to meet this criteria consider [R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination), defined as

$$R^{2}=1-\frac{\sum_{i}^{}{(f(x_{i}) - y_{i})^{2}}}{\sum_{i}^{}{(y_{i} - \bar{y}})^{2}}$$

We recognize the numerator as mean squared error and the denominator as the variance of the response variable. It clearly follows that as $\text{Var}(Y)$ increases $R^{2}$ decreases. But is this what we want?

Consider the following table which contains hypothetical calculations of $R^{2}$ from production data in May and June.

Month|MSE|Var(Y)|R2
-----|---|------|-------
March|10 | 20   | 0.5
June |10 | 30   | 0.66

Note that mean squared error is the same in both months but the variance of the response increased in June. Although we can imagine plenty of perfectly valid, ordinary reasons as to why the variance is larger in June, the result is that it appears the model performed much better in May. Even though the business value from the model could very well have been the same in both months, as suggested by the observed MSE, $R^{2}$ muddles directly comparing the two.