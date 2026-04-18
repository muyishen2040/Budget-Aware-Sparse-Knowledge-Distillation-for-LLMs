import torch
import torch.nn as nn


VOCAB_SIZE = 50304
LATENT_DIM = 8


class KDAautoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.VOCAB_SIZE = VOCAB_SIZE # self.VOCAB_SIZE = 50304
        self.LATENT_DIM = LATENT_DIM # self.LATENT_DIM = {4, 8, 16, 32, 64} = {K}
        self.encoder = nn.Sequential(
            nn.Linear(in_features = self.VOCAB_SIZE, out_features= 6288),
            nn.ReLU(),
            nn.Linear(in_features = 6288, out_features= 1024),
            nn.ReLU(),
            nn.Linear(in_features = 1024, out_features= 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features= self.LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features = self.LATENT_DIM, out_features= 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features= 1024),
            nn.ReLU(),
            nn.Linear(in_features = 1024, out_features= 6288),
            nn.ReLU(),
            nn.Linear(in_features = 6288, out_features= self.VOCAB_SIZE),
        )
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        x = self.encoder(x)
        x_latent = self.softmax(x) # *** This SM ensures that the latent space is a probability distribution over the compressed vocabulary dimension!
        x = self.decoder(x_latent)
        return self.softmax(x), x_latent