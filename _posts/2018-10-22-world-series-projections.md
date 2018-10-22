---
layout: post
mathjax: true
title: World Series Projections
github: https://github.com/dantegates/mlb-statcast
creation_date: 2018-10-22
last_modified: 2018-10-22 15:13:43
tags: 
  - pymc3
  - MLB
  - statcast
  - World Series
  - forecasting
  - projections
---


In this post we'll use the model from my previous post [Hierarchical Bayesian Ranking](https://dantegates.github.io/2018/09/20/hierarchical-bayesian-ranking.html) to project the world series winner.

# Likelihood of RedSox winning world series

In my previous post I demonstrated how to use Bayesian inference to rank professional baseball teams. The gist of the approach to include parameters in our model that represented each team's ability to win baseball games. For the purpose of ranking the teams we were only concerned with these learned abilities. However, the model also included parameters representing the probability that one team will win over another which can be used to forecast the likelihood that the Red Sox will win the world series.

Since the model is already fit obtaining the projection is as simple as sampling from the model posterior in a monte carlo simulation. The process remarkably simple. As single simulation looks like

1. For each game in the series sample from the model posterior whether Boston will win.
2. Break when either team reaches four wins.
3. Record the outcome and repeat.

Performing the experiment we obtain a 95% [credible interval](https://en.wikipedia.org/wiki/Credible_interval) indicating that the probability the Red Sox will win the world series is between 0.54 and 0.73 with the median probability 0.63.

Remeber the definition of the posterior $P(\theta \ \vert \ D)\sim P(D \ \vert \ \theta)P(\theta)$ means that the credible actually represents a probability.

The outcomes of the simulation are shown in the histogram below.


![png]({{ "/assets/world-series-projections/output_2_2.png" | asbolute_url }})


If you are interested in how to generate the predictions from the fit model the entire notebook behind this post can be found [here](https://github.com/dantegates/mlb-statcast/blob/master/bayesian-ranking-full.ipynb).