---
layout: post
mathjax: true
title: From Docker to Kubernetes
github: https://github.com/dantegates/from-docker-to-kubernetes
creation_date: 2018-05-24
last_modified: 2018-06-02 16:34:39
tags: 
  - docker
  - kubernetes
  - helm
  - Software
  - Industry
---


Docker has been a huge win for deploying software across the board and in particular for deploying machine learning models from my own experience. Perhaps you've already adopted this practice but wonder how to take the next step and deploy your image at scale via Kubernetes. If so this post is for you.

## What this post covers

In this post we'll build a container around a simple [flask](flask.pocoo.org) app and write a Helm chart that enables us to run the app on Kubernetes. If you've followed the [bitnami tutorial](https://docs.bitnami.com/kubernetes/how-to/create-your-first-helm-chart/) this post is a concrete example with a bit more detail of [step 3.](https://docs.bitnami.com/kubernetes/how-to/create-your-first-helm-chart/#step-3-modify-chart-to-deploy-a-custom-service) in the tutorial.

Things this post will not cover are:

- The ins and outs of all that Helm is and what it can do
- How to stand up a real (not local) Kubernetes cluster

# Why Helm?

Helm was introduced to standardize how Kubernetes applications are "packaged." The anology that the project uses is that Helm is to `yum` or `apt-get` as Helm charts (the data about your project that the `helm` executable consumes) are analogous to `rpm` or `deb` files.

For the developer who wants to quickly get started with Kubernetes Helm is a great solution. There are many moving pieces and configurations to a Kubernetes app and Helm provides a great way of encapsulating and version controlling all of these details. In this way it is a bit like `docker-compose`, i.e. a single place where you can define your configurations and coordinate services.

# Prerequisites

If you want to follow along with the code in this post

1. Install [minikube](https://github.com/kubernetes/minikube) (an app that lets your run a local Kubernetes cluster) and start it up with `minikube start`.
2. Install [helm](https://github.com/kubernetes/helm) and initialize it on your minikube cluster. I recommend using `helm init --upgrade --service-account default`.
3. Clone this [GitHub repository](https://github.com/dantegates/from-docker-to-kubernetes). This repo contains the flask app, Dockerfile and Helm chart that we'll be using.
4. If at any point you experience errors and can't follow along, skip to the Troubleshooting  section at the bottom of this page for a link to an excellent blog post on diagnosing errors when deploying to Kubernetes.

# To Kubernetes!

## The Docker image

First we'll build the Docker image (the container needs to exist on our local machine before installing the helm chart. You can pull images from DockerHub or a private registry, but for this example we'll keep it simple and build it locally).

```shell
docker build -t from-docker-to-kubernetes:stable .
```

Now this container isn't anything special. It's simply a flask app that runs on port 5000 inside the container and returns a randomly generated number every time you visit the root (`/`) endpoint. When we customize the Helm chart we'll see that it's important that this image has a meaningful tag (i.e., not latest).

## Writing the Helm chart

`helm create` is the simplest way to create a Helm chart for beginners. It is also recommended as a best practice by the Helm developers as it guarantees you'll be starting your project off with the correct chart structure.

```shell
helm create app-chart
```

If you just ran this command you should now have a directory called `app-chart`. This directory is the Helm chart for your project. The chart is also functional out of the box (though not very interesting yet), it chart describes a simple `nginx` server and can be installed to the cluster with

```shell
helm install --name example ./app-chart --set service.type=NodePort
```

You can verify that the installation is working by running the following command in your terminal

```shell
minikube service example
```

The "Welcome to nginx!" page should have opened in your browser.


To recap, in only 4 lines of code we

1. We fired up a local Kubernetes cluster.
2. We used the built in `helm create` tool to give us a basic Helm chart template.
3. We installed the nginx Helm chart to the cluster with `helm install`. The service type was set to `NodePort` to expose the service outside of the cluster. The Bitnami tutorial contains for details on this.
4. We verified that we could interact with the service.

So far, this was pretty easy.

## Customizing the chart to your project

So how do we customize this default chart to run our flask app. It turns out there are only 2 things we need to add to a single file to complete the chart.

1. Specify our custom docker image.
2. Specify which port inside the container Kubernetes should listen to.

To make the first change we edit the lines in `app-chart/values.yml` from

```yml
image:
  repository: nginx
  tag: stable
```

to

```yml
image:
  repository: from-docker-to-kubernetes
  tag: stable
```

To make the second two changes we add the lines to `values.yml`

```yml
container:
  # changed from 80
  port: 5000
```

Now we can install our app with

```shell
helm install --name from-docker-to-kubernetes ./from-docker-to-kubernetes-chart --set service.type=NodePort
```

The following `git diff` shows all changes made to the default chart created with `helm create`.


```python
!git diff --no-color app-chart
```

    diff --git a/app-chart/values.yaml b/app-chart/values.yaml
    index 6d0f0c7..9fd69ab 100644
    --- a/app-chart/values.yaml
    +++ b/app-chart/values.yaml
    @@ -5,13 +5,16 @@
     replicaCount: 1
     
     image:
    -  repository: nginx
    +  repository: from-docker-to-kubernetes
       tag: stable
       pullPolicy: IfNotPresent
     
     service:
       type: ClusterIP
    -  port: 80
    +  port: 5000
    +
    +container:
    +  port: 5000
     
     ingress:
       enabled: false


And that's it! We can now install and run this Helm chart using the same commands as above and see our app working in the browser.

## Troubleshooting

In this case we have properly installed our flask app. However we overlooked a key element that tripped me up when I was writing my first helm chart.

In `templates/deployments.yml` "liveness" and "readiness" probes are defined. These are descriptions of how to ping the service to know if it is "alive." By default the example in `helm create` simply makes an HTTP GET request to the root route - '/'. If it returns 200 the checks are considered successful. If not, the service will continually be restarted until the check passes. It happened to be that the image I used when first trying out `helm` doesn't define a root endpoint which caused both probes to fail.

Fortunately I was able to figure out the issue by reading [this excellent blog post](https://kukulinski.com/10-most-common-reasons-kubernetes-deployments-fail-part-1/) which has great trouble shooting advice for working with Kubernetes.