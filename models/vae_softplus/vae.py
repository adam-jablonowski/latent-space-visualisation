import time

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import functional
from torchvision import datasets, transforms

from manifold import Manifold


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        # use softplus instead of ReLU for double differentiability
        h = F.softplus(self.fc1(x))
        h = F.softplus(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.softplus(self.fc4(z))
        h = F.softplus(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class ManifoldVae(Manifold):
    def __init__(self, dim: int, dataset, device=torch.device("cpu")):
        self.name = "mnist"
        self.model = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=dim).to(device)
        model_sub_dir = {
            datasets.MNIST: "mnist",
            datasets.FashionMNIST: "fashion_mnist",
        }[dataset]
        self.model.load_state_dict(
            torch.load(
                f"models/vae_softplus/{model_sub_dir}/vae_dim{dim}.ckpt",
                map_location=device,
            ),
            strict=False,
        )
        self.dataset = dataset

    def loss_function(self, recon_x, x, mu, log_var):
        ### Computes loss of model
        def batch_loss(recon_x, x, mu, log_var):
            x = torch.tensor(x)
            x = x.view(784)
            x = x.float()
            mu = torch.tensor(mu)
            mu = mu.float()
            BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            return (BCE + KLD).item()

        return [batch_loss(*e) for e in zip(recon_x, x, mu, log_var)]

    def encode(self, x):
        ### Returns vectorized encoding of x as mu and log var
        x = torch.tensor(x)
        x = x.view(-1, 784)
        x = x.float()
        mu, log_var = self.model.encoder(x)
        return mu.detach(), log_var.detach()
        # return self.model.encoder(x)[0].squeeze(1).detach()

    def decode(self, z):
        ### Returns vectorized encoding of sampled z
        z = torch.tensor(z)
        z = z.float()
        return self.model.decoder(z)

    def get_datasets(self, batch_size=100):
        ### Returns tuple of datasets train and test loader
        # MNIST Dataset
        train_dataset = self.dataset(
            root="./datasets/",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        test_dataset = self.dataset(
            root="./datasets/",
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def point_info(self, point):
        ### Returns info about latent point in form of go.Image
        decoded = self.decode(torch.Tensor(point))
        decoded = decoded.reshape((28, 28)).detach()
        color_range = 255
        decoded *= color_range
        decoded = torch.stack([decoded] * 3).movedim(0, -1)
        return go.Image(z=decoded.detach())

    def metric_tensor(self, z, nargout=1):
        ### Returns vectorized metric tensor at latent point z (and its jacobian if nargout=2)
        time.time()

        def get_metrics(latents):
            metrics = []
            # batchsize x D x batchsize x d
            jacobians = functional.jacobian(
                self.model.decoder, latents, vectorize=True, strategy="forward-mode"
            )
            for i, jacobian in enumerate(jacobians):
                jacobian = jacobian[:, i, :]
                metric = jacobian.T @ jacobian
                # jacobian vector product check
                metrics.append(metric)
            return torch.stack(metrics)

        def get_dmetrics(latents):
            dmetrics = []
            # batchsize x d x d  x batchsize x d
            metric_jacobians = functional.jacobian(get_metrics, latents, vectorize=True)
            # , strategy="forward-mode")
            for i, jacobian in enumerate(metric_jacobians):
                jacobian = jacobian[:, :, i, :]
                dmetrics.append(jacobian)
            return dmetrics

        metrics = []
        if nargout == 2:
            dmetrics = []

        if len(z.shape) == 1:
            z = [z]

        for latents in z:
            if len(latents.shape) == 1:
                latents = torch.tensor(latents, dtype=torch.float32).unsqueeze(0)
            metrics += list(get_metrics(latents))

            if nargout == 2:
                dmetrics += get_dmetrics(latents)

        M = torch.stack(metrics)
        if nargout == 2:
            dMdz = torch.stack(dmetrics)
            return M.detach().cpu().numpy(), dMdz.detach().cpu().numpy()

        return M.detach().cpu().numpy()
