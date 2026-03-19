import torch
from jaxtyping import Float
from torch import Tensor, nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)

    def encode(self, x: Float[Tensor, "batch d_in"]) -> Float[Tensor, "batch d_sae"]:
        return torch.relu(self.encoder(x))

    def forward(
        self, x: Float[Tensor, "batch d_in"]
    ) -> tuple[Float[Tensor, "batch d_in"], Float[Tensor, "batch d_sae"]]:
        codes = self.encode(x)
        recon = self.decoder(codes)
        return recon, codes
