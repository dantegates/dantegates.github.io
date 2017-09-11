
# Seven must know tools for a Python developer

This year I spent six months at startup leading the development of a large scale analytics platform written in Python. This post lists some of the tools I found particularly useful for writing high quality software. I won't discuss any of these in great detail here (although some topics are likely to be covered at greater length in a future post). Rather, this post simply documents some of the great tools out there that a Python developer should be aware of.

As an aside, after writing this post I noticed that all of these tools have one thing in common: minimal barrier to entry. However that's no surprise since I favor elegant, simplistic and easy-to-use solutions.

## Redis

**Why you should know about it**: Redis is an open source Key-Value store that plays *really* nicely with Python. It's lightweight and has a very small learning curve. Queues, caches and pub-sub are only the beginning.

**Commentary**: Redis was certainly one of my favorite tools while working on this project and became my go to solution for just about every need that surfaced due to distributed or parallel processing.

What really got me hooked on Redis was the that I was able to hook it into Python's built in logging library to allow logging accross many Python instances at once in under a day with no prior experience using the library. It's simplicity are its strongest features in my book.

Furthermore, though incredibly simple, Redis is an incredibly flexible data structure, essentially a hash map that you can access from any application over a network. In addition to using its message broker capabilities as a logging solution, some other things I have used it for include building a [functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)-like decorator that works accross separate Python instances, a cache for API requests and the backend of a service used for coordinating the assignment of uniue IDs accross processes.

## Docker

**Why you should know about it**: You need your apps to be portable. You need to be sure that your team is developing in the same environment. You're in charge of deployment even though you aren't devops (work in a startup?). You would like to just be able to *pull* an environment with all of your codes dependencies (java, C, odbc, etc.) pre-installed for you.

**Commentary**: I feel like there is so much talk about docker right now that I don't need to say much about it here. I'll only add that from my experience as a lead software engineer some of its biggest benefits were:

- Starting a usable Redis instance with a *single* command.
- A replacement for virtualenv.
- Portability. When it looked one of our VMs in the cloud needed to be completely wiped and restarted i didn't sweat a drop. If your app is Dockerized there's no need to worry about the pains of the installation process of your software's dependencies.

## Flask

**Why you should know about it**: Flask is the *simplest* framework out there (in Python) for building webapps. It's perfect for small applications, like microservices, and prototypes.

**Commentary**: Python developers shouldn't think of the well known [Flask](http://flask.pocoo.org/) library as for web developers only. In my opinion Flask is the best solution for writing REST APIs in Python. It's simplicity makes the task of turning your code into apps incredibly simple.

Pair with [Gunicorn](http://gunicorn.org/) and package with Docker for a tried and true pattern for deploying your apps.

## coverage.py

**Why you should know about it**: coverage.py is a straightforward resource for generating reports on the test coverage of your code base.

**Commentary**: [coverage](https://coverage.readthedocs.io/en/coverage-4.4.1/) is a great library for analyzing your applications test coverage. It plays very nicely with pythons built-in [unittest](https://docs.python.org/3/library/unittest.html) library so if you already have tests for your application its very easy to get started with coverage.

Of course, this library can't tell you anything of the quality of your tests but it builds really nice reports giving high-level information on your code that has test coverage as well as very low-level information (down to the line level). This library was great for identifying which modules needed tests and more than once it showed me that I was missing a test for a certain case (as in an `if-else` clause).

## Makefiles

**Why you should know about it**: But we're writing in Python right? Yes, but it turns out Makefiles are a great tool for building quick CLI tools in a Python repository. Link below says it all in 40 words.

**Commentary**: I came accross this gem of an idea in this section of the [Hitchhiker's guide to Python](http://docs.python-guide.org/en/latest/writing/structure/#makefile). Want to have the ability to initialize your repository's filesystem, execute the entrypoint of your code or run your tests suite with simple commands at the command line with a *minimum* amount of effort? If so, then check out the link above.

## git's pre-commit hook

**Why you should know about it**: git's pre-commit hook is a nice little tool you can use to customize your commits.

**Commentary**: If you are using [git](https://git-scm.com/) as the VCS for your project then the [pre-commit hook](https://git-scm.com/docs/githooks#_pre_commit) is a feature you should definitely know of. The pre-commit hook simply executes a script every time you call `git commit` and if the script returns a non-zero exit code the commit gets aborted.

One of the most common use cases for this hook is to run a test suite each time you commit. If the test suite fails, so does the commit. Even if you have your test suite running via some CI tool when you push your code to the master repository it's helpful to have these tests run automatically when developing on your local machine.

## mock.patch

**Why you should know about it**. The [unittest.mock](https://docs.python.org/3/library/unittest.mock.html) package is a great built-in (in Python3.3+, for Python2 you need to `pip` install `mock`) for overriding dependencies you don't want to run in your tests.

**Commentary**: Before I used `unitest.mock` when I needed to override some dependency (say a class that accesses a database) my test code looked something like this:

```python
import unittest
from mymod import ClassWithDependency

class SomeFakeClass:
    ...

class PseudoClassWithDependency(ClassWithDependency):
    attribute_to_override = SomeFakeClass
    
class TestClassWithDependency(unittest.TestCase):
    def test(self):
       # do something uses calls PseudoClassWithDependency
       ...
```

Now often this sort of design pattern for dependency injection is actually the right choice. However, the test code is rather verbose as it requires creating new classes. Furthermore, suppose you need to test several behaviors you might experience when using `ClassWithDependency.attribute_to_override`. If this is so, you now may end up with classes such as `PseudoClassWithDependencySuccess` and `PseudoClassWithDependencyFail` and your test code is even more verbose than before.

It's essential for developers to be confident in there tests and this means that test code sould be as simple as possible. However the pattern above violates this principle.

Fortunately, we can use `unittest.mock` to give us the same functionality with a standard interface supported by the PSF. Here's an example of what the code above would look like using `unittest.mock.patch`.

```python
import unittest
from unittest import mock
from mymod import ClassWithDependency

class TestClassWithDependency(unittest.TestCase):
    @mock.patch('mymod.ClassWithDependency.attribute_to_override')
    def test_success(self, mock_attr_to_override):
        mock_attr_to_override.return_value = True
        # do something with the *real* class ClassWithDependency
        ...

    @mock.patch('mymod.ClassWithDependency.attribute_to_override')
    def test_success(self, mock_attr_to_override):
        mock_attr_to_override.return_value = False
        # do something with the *real* class ClassWithDependency
        ...
```
