---
layout: post
mathjax: true
title: Neural Networks Explained
github: https://github.com/dantegates/neural-networks-explained
creation_date: 2019-12-12
last_modified: 2019-12-12 16:59:50
tags: 
  - Deep Learning
  - Beginner
  - Opinion
---


Last night - at the first ever [PyData Philly](https://www.meetup.com/PyData-PHL/) meetup! (thanks to all of the organizers for taking the initiative to get this started) - I gave a lightening talk titled "Neural Networks Explained". This was an idea I had been thinking about for about a week and the talk was a great opportunity to get my thoughts together before writing them down in a post. Since I received a lot of positive feedback from the talk I'm going to try and keep that "lightening talk feel" in this post.

# Setup

When I was first starting out in this industry, I found myself at a point where I felt very comfortable, at both a conceptual and working level, with a variety of machine learning models like logistic regression and random forests. This was in 2014, so it naturally wasn't long before I began hearing about neural networks. They sounded cool. They sounded interesting. They sounded mysterious. They sounded intimidating. They sounded like deep magic.

I started reading up on neural networks and consistently came across material that attempted to explain them away with graphical images or biological analogies. All of that is fine, I suppose, but none of it gave me a concrete idea of what a neural network *is*.

Eventually I had a conversation with a coworker that was a real "a ha!" moment. The phrase that stuck with me was "neural networks are matrix multiplication and stochastic gradient descent, nothing more, nothing less", which coming from a background in math was incredibly helpful.

It's this idea that I would like to try and break down in this post for those who find themselves in a similar position.

# Building blocks

Let's start with [linear regression](https://en.wikipedia.org/wiki/Linear_regression). This is a great starting point because it is ubiquitous across so many disciplines and industries and at the same time will lead us to the basic building blocks of neural networks.

When I first encountered linear regression it was expressed in an algebraic form such as

$$
y=\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+\cdots+\beta_{n}x_{n}
$$

where each $x_{i}$ is understood as the value of a feature in the data set and the $\beta_{i}$ are "coefficients" we want to learn. Thus we obtain our predictions by multiplying each feature by its corresponding coefficient and summing these products (plus a constant commonly referred to as an intercept or error term).

Another way we could choose think about this expression is the dot product of two vectors $X\in{R^{n}}$, $\beta\in{R^{n}}$. That is

$$
y=X^{T}\beta + \alpha
$$

where $\alpha$ is a real number replacing $\beta_{0}$ for notational convenience.

We can add one more piece to this equation to obtain a well known model used for classification. By applying a simple non-linear transformation to the scalar output of the linear regression model and thereby obtaining a value between 0 and 1 we now have the well known logistic regression [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model

$$
y=\sigma(X^{T}\beta+\alpha)
$$

where $\sigma(x)=\frac{1}{1+e^{-x}}$.

# Taking the leap to neural networks

To take the leap from the regression models above to a neural network we simply add a dimmension to each of our model parameters above. That is, we replace the coefficient vector $\beta\in{R^{n}}$ with a "weight" matrix $W\in{R^{n\times m}}$. Similary, the scalar $\alpha$ is replaced with a "bias" vector $b\in{R^{m}}$.

Putting it all together we have

$
H=f(X^{T}W+b)
$

where $f$ is a differentiable function, known as the "activation", is typically non-linear and applied "element-wise" along its input. Commonly used are the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

If you've been following the math you will have noticed that $H$ is not a scalar, but in fact a vector of dimension $m$. Since we (usually) wish to predict a scalar output we'll need to do one more matrix multiplication with another "weight" matrix, add another "bias" vector, and apply one more "activation" function to get our final output. In this final transformation the activation function may or may not be non-linear depending on the domain of the target variable. 

Note that $H$ is what is known as a "hidden layer" with $m$ "units".

What we have just described a "single (hidden) layer neural network" which we could write more explicitly as

$$
\begin{align}
&H=f_{1}(X^{T}W_{1})+b_{1}\\
&y = f_{2}(HW_{2}+b_{2})
\end{align}
$$

There we have it, matrix multiplication (with a few twists), "nothing more, nothing less".

# Deep neural networks

The model described above is often referred to as a "feed forward neural network" in the literature. This name refers to the fact that we often have multiple hidden layers that "feed forward" into one another - in other words composition of functions that [Khan academy teaches in its precalculus track](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:composite/x9e81a4f98389efdf:composing/v/function-composition).

A "deep neural network" is just this "feed forward" definition with many hidden layers.

# CNNs, RNNs, Oh My!

What about other "architectures" like [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) or [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network)?

These too, are matrices [all the way down](https://en.wikipedia.org/wiki/Turtles_all_the_way_down) so to speak.

Note that I didn't say "matrix multiplication all the way down." This was intentional because some of these models, [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) for instance, make use of matrix multiplication and *other* matrix operations such as element wise products. In any event, at the end of the day, it's matrix math with non-linear, differentiable functions sprinkled in.

# Motivating neural networks

Once we've established the basic mechanisms that make up a neural network we can motivate why these models are so successful without appealing to [pictures](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/1280px-Colored_neural_network.svg.png) or anecdotes of how people thought the brain worked 60 years ago.

Though I hesitate to use this description, let's call neural networks linear regression models on steroids. They are both capable of learning many, many more parameters than a simple linear model *as well* as learning non-linear relationships that we know exist in much of our data. Considering the usefulness these much simpler linear models lend to many applications, it should come as no surprise that these more advanced models can work so well.

# See also

As I mentioned at the beginning of this post, I'm not adding anything original to the conversation here, I'm just saying it in my own words. If you are interested in learning more here are some other resources to check out.

[An Attempt At Demystifying Bayesian Deep Learning](https://youtu.be/s0S6HFdPtlA): The substance of the first 5-10 minutes of this talk is nearly identical to the conversation we've had here and the speaker goes even further.

[Visualizing the Learning of a Neural Network](http://srome.github.io/Visualizing-the-Learning-of-a-Neural-Network-Geometrically/): This is a post from my friend, Scott Rome (and indeed the coworker mentioned at the beginning of this post), who describes what exactly a neural network is learning with some really cool visualizations.

[Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem): If you are really interested in the math, here is a theorem you should know about. The actual practicality of this theorem is debatable, but interesting all at the same time and nevertheless relevant to this discussion.

# A final note on SGD

I suppose I should conclude this post admitting that, although stochastic gradient descent was mentioned in the beginning of this post I never had any real intent of addressing it again.

I'll simply say that when you read "backpropogation" in the literature it is really referring to [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) or extensions thereof. Additionally, where I've explicitly pointed out in this post that a function should be differentiable it's because this is a requirement for leveraging SGD. The rest, at this time, I will leave as an excecise to the reader :).
