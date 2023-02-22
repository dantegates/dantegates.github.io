---
layout: post
mathjax: true
title: PyMCon Afterword
github: https://github.com/dantegates/the-power-of-bayes-in-industry
creation_date: 2023-02-22
last_modified: 2023-02-22 07:11:14.186111
tags: 
  - Bayesian Inference
  - pymc
---


A couple of weeks ago I gave a [talk](https://www.youtube.com/watch?v=7KrspD1TZNU&ab_channel=PyMCDevelopers) for the 2023 [PyMCon Web Series](https://pymcon.com/events/). The aim of the talk was to advocate the advantages of a _first principles_ approach to modeling, i.e. one that places focus on modeling the data generating process (DGP) and not simply outcomes alone. This post covers two slides, _Tips_ and _Resources_, that didn't make the final cut. Below, each would-have-been bullet point is covered briefly in two sentences or less.

## Tips

#### Remember: there's no "right" answer

By nature, the best of these _first principles_ models can feel so natural that it becomes tempting to think of them as _the right_ model, but any one approach is likely chosen from among other reasonable alternatives. 
It's [_approximations all the way down_](https://en.wikipedia.org/wiki/All_models_are_wrong) so take liberty to reimagine your models and exercise creativity.

#### Don’t rush to your favorite modeling package

Take time to sketch, whiteboard, brainstorm with colleagues, do EDA, [simulate from your assumptions](https://colab.research.google.com/drive/1luKMwARuZPuu-CVqV_OU5kJTvC5e5Ogs#scrollTo=NULj5Wi7jEKn) or work things out on paper before model building.

#### Start simple and add assumptions incrementally

Examine your progress between iterations and stop when you exhaust the ["information available to estimate any additional parameters"](https://mc-stan.org/users/documentation/case-studies/golf.html). Increase complexity judiciously.

#### Think graphically

See [this excellent case study](https://betanalpha.github.io/assets/case_studies/generative_modeling.html).

#### Learn to love priors

This methodology unashamedly places priors front-and-center, so lean into it. Use informative priors and avoid the temptation to limit yourself with flat priors.

#### Prior predictive samples

Do it.

#### Get comfortable with basics of probability

A basic vocabulary is needed to express your mental models mathematically. A few non-exhaustive examples of things you should know include the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability), the rules of [conditional probability](https://en.wikipedia.org/wiki/Conditional_probability), [joint probability](https://en.wikipedia.org/wiki/Joint_probability_distribution) and  [expectations](https://en.wikipedia.org/wiki/Expected_value).

#### Challenge yourself to see these problems everywhere

Learning to recognize when a _first principles_ approach is or isn't appropriate is key. This skill comes with practice and you can get reps in by looking for examples in everyday life: as an example, check out [the post on my quarantine playlist](https://dantegates.github.io/2020/04/20/my-quarantine-playlist.html).

## Resources

Other industry examples. Each illustrates a nice "middle ground" between traditional ML and the model I demonstrated in my talk which was very domain specific. I.e. both examples model the DGP, but are not specific to any one business model.

- [PyMC Labs x Hello Fresh](https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/) Mixed Media Marketing
- [Pareto/NBD](http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf)

Additional resources on the putting case study (the motivating example of the talk):
- [The original case study](https://mc-stan.org/users/documentation/case-studies/golf.html)
- [My Colab notebook, companion to the talk](https://colab.research.google.com/drive/1luKMwARuZPuu-CVqV_OU5kJTvC5e5Ogs)
- [Example from the PyMC Gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html)

Other resources I was reading/listening to while preparing the talk:
- [What's the Probabilistic Story?](https://betanalpha.github.io/assets/case_studies/generative_modeling.html)
- [The Prior Can Often Only Be Understood in the Context of the Likelihood](https://www.mdpi.com/1099-4300/19/10/555)
- Learning Bayesian Statistics interview with [Michael Betancourt](https://learnbayesstats.com/episode/6-a-principled-bayesian-workflow-with-michael-betancourt/) and [the authors of Regression and Other Stories](https://learnbayesstats.com/episode/20-regression-and-other-stories-with-andrew-gelman-jennifer-hill-aki-vehtari/)
- [Andrew Gelman's Keynote at PyData NYC](https://www.youtube.com/watch?v=veiLCvcLIg8&t=774s&ab_channel=PyData) - my first introduction to the putting example.