---
layout: post
mathjax: false
katex: true
title: What is the posterior and why does it matter?
github: https://github.com/dantegates/what-is-the-posterior
creation_date: 2026-08-31
last_modified: 2026-08-31 10:36:08.333709
tags: 
  - Bayesian Inference
  - pymc
---


_This post is an adaptation of a [talk](https://dantegates.github.io/slides/what-is-the-posterior.html#/title-slide) I gave at Data Philly a couple of years ago._

# Introduction

We've all seen it. [That neon sign](https://www.flickr.com/photos/mattbuck007/3676624894), hanging on some wall of some math department 
somewhere. Bayes' theorem:

<div id="eq:bayes">

$$
P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)} \,.
\tag{1}
$$
</div>

I would wager that Bayes' Theorem is best known for calculating simple conditional probabilities, like [_the chance that a prize is behind door two,_ **given** _a goat is behind door one_](https://en.wikipedia.org/wiki/Monty_Hall_problem#Bayes'_theorem). However, in my opinion, its best application is when it serves as the foundation of Bayesian Inference.

In practice, inference tends to be more interesting, because it accommodates a richer class of questions. Contrast the classic Monty Hall problem with the following: what is _the chance that a prize is behind door two, given a goat is behind door one_ **and the outcomes of past games**?

This slight change in phrasing suddenly allows us to use data from previous games to consider questions like perhaps, every so often the game never had a prize, just to tilt the long-term odds in Monty's favor.

In Bayesian Inference, <a href="#eq:bayes">(1)</a> can be broken down into pieces that each play a specific role. In our discussion, I want to explore the possibilities of the left-hand side, which is known as the posterior distribution.[^theta-d]

<div id="eq:posterior">

$$
\underbrace{P(\theta\mid D)}_{\mathrm{posterior}} = \color{grey}{\frac{\overbrace{P(D\mid \theta)}^{\mathrm{likelihood}}\ \overbrace{P(\theta)}^{\mathrm{priors}}}{\underbrace{P(D)}_{\mathrm{marginal\ likelihood}}}}
\tag{2}
$$
</div>

The posterior makes Bayesian Inference truly unique. It doesn't give you just one predicted value. Instead, it produces _a probability distribution_ over all possible values, with each probability directly informed by data.[^prediction-technicality]

In my own experience, it took me quite a while to recognize the full potential of this property. For years I would reduce Bayesian models to a mean plus or minus a confidence interval,[^credible-interval-technicality] and, voilà, problem solved.

While this can be a reasonable approach at times, more often, it ends up leaving value on the table, or it would simply be more prudent to use techniques that are designed to produce point estimates in the first place if that's what the job calls for.

In this post, I want to illustrate some of the unique opportunities the  posterior distribution makes possible. To do so, we'll work through an investing strategy described by Ed Thorp in [an expository note published in the MAA](https://www.edwardothorp.com/wp-content/uploads/2016/11/TheKellyCriterionAndTheStockMarket.pdf), but with an added "Bayesian twist" to help drive our point home.

# An investing strategy

As a physicist-turned-mathematician, Thorp turned his talents to Vegas, using an [ingenious device invented with Claude Shannon](https://exhibits.lib.uci.edu/thorp/spin), an original [card-counting theory](https://www.goodreads.com/en/book/show/891883.Beat_the_Dealer), and the [Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion) to beat the house. After overstaying his welcome in the city of lost wages, he returned to academia, only to change careers once again. And, once again, he found success in his new venture as a hedge fund manager by leaning on his past experiences - in particular, probability theory, applied mathematics, and a working knowledge of the Kelly criterion.[^thorp-biography]

It turns out that the Kelly criterion is central to our conversation. Developed by John Kelly at the legendary Bell Labs, the Kelly criterion seeks to maximize long-term wealth by repeatedly employing a "fixed-fractional" betting strategy that optimizes for geometric growth.

In the note referenced above, Thorp demonstrated that Kelly's framework could be extended from a betting strategy for events with discrete outcomes to an investment strategy for events with continuous ones, like the S&P 500. This investment strategy will form the basis of our discussion.

To get started, let's begin by taking a look at those returns between 1928 and 2024.[^tbill-technicality]

![]({{ "/assets/what-is-the-posterior/historical-returns.png" | absolute_url }})

Following the Kelly criterion, Thorp too sought a fixed-fractional investment strategy, where an investor consistently reinvests their total wealth at a rate of $f$. The challenge is to avoid rates of $f$ that are too high and lead to eventual ruin (complete financial loss), or too low and diminish the potential of long-term gains.

Thorp showed that given an adequate probability model, let's call it $p(s)$, which describes the (relative) probability of a return $s$, one should invest at the rate $f^{*}$ which solves the optimization problem

<div id="eq:expected growth rate">

$$
\begin{equation}
\operatorname*{argmax}_{f}{\int_{a}^{b}{p(s)\, \log(1+fs)\, ds}}\,.
\end{equation}
\tag{3}
$$
</div>

Let's break down the integrand.

* If $\log(1 + fs)$ is the _geometric growth rate_ of one's wealth, when invested at a rate of $f$ for a return rate of $s$,
* then $p(s)\log(1+fs)$ is the _probability-weighted_ geometric growth rate. 
* Thus, integrating over $s$ gives us the _expected geometric growth rate_ for our investment strategy $f$.

We can visualize this optimization problem by plotting each investment strategy, $f$, against the corresponding expected geometric growth rate.

![]({{ "/assets/what-is-the-posterior/visualizing-kelly-criterion.png" | absolute_url }})

Thorp spent a lot of time carefully building the mathematical justification for several attributes about this problem that can be seen in the chart above:

* There is a unique, optimal investment rate $f^{*}$.
* Investing at rates beyond $f^{*}$ eventually leads to ruin (the ["gambler's dilemma"](https://en.wikipedia.org/wiki/Gambler%27s_ruin)).
* The conditions under which the objective function will admit a meaningful solution $f^{*}$.
* How to find $f^{*}$ and how to avoid ruin.

Wading into those details is beyond the scope of this post, but it is crucial that we emphasize that those details exist. Going forward, it's important to keep in mind that Thorp's strategy, though perhaps somewhat intuitive when you think about it, is more than a heuristic - it is a method with theoretical grounding.

But, of course, all of this is true _supposing that you have access to an adequate probability model_, **$p(s)$**, in the first place.


# Modeling the returns

Let's use Thorp's model as a starting point, with some choice modifications. 

Keep in mind we're not trying to one-up Thorp (who acknowledged his model as "somewhat artificially constructed" with limitations) or produce an oracle for S&P investments. Rather, we're trying to develop an example that illustrates the unique properties of the posterior in Bayesian Inference.

Thorp took $p(s)$ to be a truncated normal distribution, and derived its mean and standard deviation empirically from the returns in his data set (the same as ours, but from the years 1926-1984) using [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

For our illustration, it's sufficient to keep the normality assumption (which admittedly requires some hand-waving) but question the advisability of using a fixed pair $(\mu,\sigma)$, derived via maximum likelihood or otherwise.

Our main concern with choosing a fixed pair ($\mu, \sigma$) is that we risk bias due to our relatively small sample size. While this type of bias is always a concern when working on a model, it is especially so in our case where what may seem like small changes in these parameters can produce drastically different investment results.

You can see this play out in the following chart. On the right, we show several values of $(\mu,\sigma)$ that plausibly fit our data. On the left are the corresponding investment strategies, which yield drastically different financial advice, ranging from investing less than 100% of one's wealth to greater than 300% (implying heavy leverage).

Thus, the risk of producing an estimate that is _slightly_ off could be the difference between massive over-investment (leading to ruin) or massive under-investment (leading to lost opportunity).

![]({{ "/assets/what-is-the-posterior/motivating-bayesian-approach2.png" | absolute_url }})

On the other hand, a Bayesian model produces a nice compromise. By treating $\mu$ and $\sigma$ as random variables _while_ conditioning on past data, our estimates are informed by the data without being unduly committed to a single pair of model parameters.

Revisiting <a href="#eq:posterior">(2)</a>, the posterior expresses a probability distribution over $\mu$ and $\sigma$ conditional on past returns.

<div id="eq:posterior specific">

$$
p(\mu,\sigma\vert s) = \frac{p(s\vert \mu,\sigma)p(\mu,\sigma)}{p(s)}\,.
\tag{4}
$$
</div>


# From posterior to posterior predictive

With the posterior on hand, we're ready to tackle Thorp's investing problem, but we'll need to replace $p(\hat{s})$ with $p(\hat{s}\vert s)$ and adjust our notation, letting $\hat{s}$ and $s$ differentiate between arbitrary and historical returns to avoid ambiguity:

<div id="eq:expected conditional growth rate">

$$
\operatorname*{argmax}_{f}\int_{}^{}{p(\hat{s}\vert s) \log(1+f\hat{s})\, d\hat{s}}\,.
\tag{5}
$$
</div>

While it might look like a minor change, $p(\hat{s}\vert s)$ is a fundamental object in Bayesian Inference known as the [posterior predictive distribution](https://en.wikipedia.org/wiki/Posterior_predictive_distribution) which positions us to introduce the posterior into the objective function.

Using the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability), we can expand $p(\hat{s}\vert s)$ so that the expected growth rate is weighted over the posterior distribution in <a href="#eq:expected conditional growth rate">(5)</a>, becoming

<div id="eq:objective with posterior">

$$
\operatorname*{argmax}_{f}\iint_{}^{}{\overset{\mathrm{posterior\ predictive}}{\overbrace{p(\hat{s}\vert\mu,\sigma)\ \underset{\mathrm{posterior}}{\underbrace{p(\mu,\sigma\vert s)}}}} \log(1+f\hat{s})\,  d\mu\, d\sigma}\,.
\tag{6}
$$
</div>

This is our "Bayesian twist" on Thorp's problem. By conditioning on the historical excess returns, we produce an objective function that is _explicitly_ data-driven, without losing the theoretical support from Thorp's work.

Now that our objective function is redefined in terms of the posterior, we simply need to solve it.


# Putting it to use

Would it come as a surprise if I told you that equations, such as <a href="#eq:objective with posterior">(6)</a>, that involve the posterior or posterior predictive range from difficult to impossible to calculate analytically? While this is true for all but certain [special cases](https://en.wikipedia.org/wiki/Conjugate_prior), there's never been a better time to practice Bayesian Inference. That's because numeric methods have never been faster or more reliable and are easier than ever to use with libraries like [stan](https://mc-stan.org/) and [pymc](https://www.pymc.io/welcome.html).

These [numeric methods](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) cleverly avoid the cumbersome integration required by expressions like $p(\hat{s}\vert s)$ altogether and instead _draw samples_ from the posterior and posterior predictive:

$$
\begin{align*}
&\text{posterior predictive samples}&\hat{s}_{i}\sim p(\hat{s}\vert \mu_{i},\sigma_{i})\ \ &\\
&\text{posterior samples}&\mu_{i},\sigma_{i}\sim p(\mu,\sigma\vert s)\,.
\end{align*}
$$

How these methods work warrants its own discussion[^mcmc-references], but note that this idea leans into the recognition that the posterior is a _distribution_.

With enough posterior samples, we can approximate the analytic solutions through simulation. For example, in our case, we can trade the integral in <a href="#eq:expected conditional growth rate">(5)</a> for a weighted average over the posterior predictive samples $\hat{s}_{i}$,

$$
\int_{a}^{b}{p(\hat{s}\vert s)\log(1+f\hat{s})\, d\hat{s}}
\approx
\frac{1}{N}\sum_{i=1}^{N}{}\log(1+f\hat{s}_{i})\,.
$$

From there, the right-hand side can be optimized in a number of ways.

With tools like `pymc`, the sampling procedure is rather straightforward.

1. First, we define the model parameters and data-generating process[^data-generating-process]
2. Then we generate samples that approximate draws from the posterior, $(\mu_{i},\sigma_{i})$
3. Each $(\mu_{i}, \sigma_{i})$ is used in turn to simulate a value from the posterior predictive, $\hat{s}_{i}$
4. Last, we simulate growth rates with $\log(1+f\hat{s}_{i})$

Steps 3 and 4 are sometimes called "forward sampling", because each is determined from values in the prior step.

```python
import numpy as np
import pymc as pm

with pm.Model() as model:
    # ↓↓↓ 1. define the model ↓↓↓
    μ = pm.Gamma('μ', mu=.04, sigma=.005)
    σ = pm.Gamma('σ', mu=.2, sigma=.05)
    likelihood = pm.Normal('likelihood', mu=μ, sigma=σ, observed=s)

    # ↓↓↓ 2. generate posterior samples: p(μ,σ ∣ s) ↓↓↓
    idata = pm.sample()

    # ↓↓↓ 3. generate posterior predictive samples  ↓↓↓
    # ↓↓↓    p(ŝ ∣ s) = p(ŝ ∣ μ, σ) × p(μ, σ ∣ s)   ↓↓↓
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# ↓↓↓ 4. simulate the growth rate, for a given value `f` ↓↓↓
p_ŝ_given_s = idata.posterior_predictive.likelihood.values.ravel()

def expected_growth_rate(f):
    return -np.mean(np.log(1 + p_ŝ_given_s * f))

f_star = minimize(expected_growth_rate, ...)
```

Applying this process to our data set results in the following model fit and investment criteria:[^footnote-on-finding-f_star]

![]({{ "/assets/what-is-the-posterior/posterior-predictive.png" | absolute_url }})


# Discussion

About a year ago I came across a quote from [Allen Downey](https://allendowney.blogspot.com/2016/06/bayesian-statistics-for-undergrads.html) which summarizes our discussion well:

> Bayesian methods don't do the same things better; they do different things, which are better.

What is it that gives Bayesian methods these unique capabilities? Quoting Downey again, the posterior distribution is "exactly the information that makes Bayesian results more useful."

However, I've taken this statement slightly out of context[^downey-context], and perhaps it would be more fair to write

> Bayesian methods don't do the same things better; they do different things, which are [sometimes] better.

Conveniently, that single word, "sometimes", is a reminder of this post's primary goal: to provide a concrete illustration of the line between the _sometimes_ Bayesian methods are better and the _sometimes_ they are not.

If you're making thousands of investment decisions a day, a week, or a month, a Bayesian model may be overkill. You might even be able to simplify Thorp's strategy, using something like

$$
\operatorname*{argmax}_{f} \log(1+fs^{*})
$$

where $s^{*}$ is a point estimate produced by a more traditional ML model. In fact, Google claimed to do [something not too dissimilar](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) to this in their real-time ad auctions.

And if you have googols of data, that's probably fine. The variance on $s^{*}$ may be negligible, and the mean can provide a lot of mileage. But, if all you have are 106 data points, or you feel like you're dealing with some other meaningful source of uncertainty, Bayesian methods like we've described here can provide a robust data-driven solution.

I hope that this recognition helps you resist the temptation to simply reduce the posterior to a mean $\pm$ some confidence interval and call it a day.[^doing-the-same-thing-differently] Once this sinks in, and you've had an ["I know kung fu"](https://www.youtube.com/watch?v=6vMO3XmNXe4) moment, you'll begin to see that this post only scratches the surface.

In this example alone, it's not hard to imagine future extensions of what we put together:

* What if we [modeled the returns as a time series](https://www.pymc.io/projects/examples/en/latest/time_series/stochastic_volatility.html) instead of a static distribution?
* What if we fit the model to a portfolio of funds, accounting for the covariance between them?
* How could we use [Bayesian decision theory](https://www.cambridge.org/core/books/bayesian-optimization/introduction/B855B6A81FA1DF897C389F8B017AE891) to learn a long-term strategy over time?
* $\ldots$

# Footnotes

[^theta-d]: In Bayesian Inference, it is customary to use $\theta$ to represent the parameter(s) of your model and $D$ for observed data.

[^prediction-technicality]: This statement makes it seem as though the posterior distribution / Bayesian methods are exclusively focused on prediction. For reasons explained later, this isn't necessarily true and ignores applications like Bayesian Data Analysis. That said, we're getting ahead of ourselves, and at this stage I favored a naive statement for the sake of simplicity.

[^credible-interval-technicality]: Err... technically a credible interval. But that doesn't quite roll off the tongue the same way.

[^thorp-biography]: All of these details and more (apparently the mob cut his brakes to chase him out of town) are covered in his [autobiography](https://www.goodreads.com/en/book/show/25733505-a-man-for-all-markets), which I highly recommend.

[^tbill-technicality]: Technically we're talking about excess returns minus treasury bills, because the time value of money matters, but that's a mouthful. For our discussion, calling these "the returns" satisfies the conceptual need of our discussion.

[^footnote-on-finding-f_star]: For some pseudocode for optimizing over the posterior, see [this slide](https://dantegates.github.io/slides/what-is-the-posterior.html#/solving-for-f-better) from my Data Philly talk.

[^mcmc-references]: For more information on this topic, I like Thomas Wiecki's post [MCMC sampling for dummies](https://twiecki.io/blog/2015/11/10/mcmc-sampling/). For a more advanced treatment you can't do better than Michael Betancourt's [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434), although it's not for the faint of heart.

[^downey-context]: Downey was contrasting Bayesian methods with classical statistics in particular, and without that context his quote could be interpreted as a much broader critique.

[^doing-the-same-thing-differently]: What we might call "doing the same things differently" in the language of Downey's post.

[^data-generating-process]: I covered this topic at length in the past for [PyMCon](https://www.youtube.com/watch?v=7KrspD1TZNU).
