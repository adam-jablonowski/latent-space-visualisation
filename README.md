# Latent Space Visualisation

## About
The tool visualises latent spaces of user defined models.
For given latent space, dataset and twice differentiable mapping of latent space into dataset space it displayes dataset points with different features based on calculation of Riemannian metric dependent on the mapping.
Grafical user interface consists of components presenting different functionalities and configuration options.

Computations of Riemannian metric are based on 

Georgios Arvanitidis, Lars Kai Hansen, Søren Hauberg, Latent Space Oddity: on the
Curvature of Deep Generative Models, arXiv:1710.11379v3, 2021

## Installation
`pip install -r requirements.txt`

## Running
Run `python app.py`

Go to http://localhost:8080

## Setting manifold for visualisation
`ManifoldVae` class is the default manifold.
It loads a variational autoencoder model with softplus activations and MNIST dataset.

To change manifold used by the tool create a class extending `Manifold` and set `manifold` variable in `app.py` to an instance of your class.

## File structure
`./` - app server and backend computations

`view/` - gui elements

`view/components/` - gui elements templates

`models/` - example variational autoencoder models with training code

## Images
Global and local view of latent space of the default model.
![](image.png?raw=true "")
