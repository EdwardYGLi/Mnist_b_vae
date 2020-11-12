"""
Created by Edward Li at 11/11/20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()

    def forward(self,x):
        return x* F.sigmoid(x)


class BetaVAE(nn.Module):

    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "swish": Swish,
        "tanh": nn.Tanh,
        "lrelu": nn.LeakyReLU,
        "Elu" : nn.ELU,
        "PReLU":nn.PReLU
    }

    def __init__(self, layers, activation = "relu"):
        super(BetaVAE, self).__init__()

        encoders = []
        decoders = []
        layers_len = len(layers)
        self.fc1 = nn.Linear(784, layers[0])

        for i in range(1, layers_len // 2):
            encoders.append(nn.Sequential(nn.Linear(layers[i - 1], layers[i]), self.activations[activation]()))
        self.encoder = nn.Sequential(*encoders)

        self.mu = nn.Linear(layers[layers_len // 2 - 1], layers[layers_len // 2])
        self.sig = nn.Linear(layers[layers_len // 2 - 1], layers[layers_len // 2])

        for i in range(layers_len // 2 + 1, layers_len):
            decoders.append(nn.Sequential(nn.Linear(layers[i - 1], layers[i]), self.activations[activation]()))

        self.decoder = nn.Sequential(*decoders)

        self.fc_end = nn.Linear(layers[-1], 784)

    def encode(self, x):
        x = self.fc1(x)
        x = self.encoder(x)
        return self.mu(x), self.sig(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        x = self.fc_end(x)

        return x

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var