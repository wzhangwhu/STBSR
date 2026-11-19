"""Three-view graph encoder and expression-mean decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import GraphConvolution


def _advance_linear_draws(in_features, out_features, count=1):
    """Maintain the published run's initialization order for active parameters."""
    torch.rand(count * (in_features + 1) * out_features, device="cpu")


class GCN(nn.Module):
    def __init__(self, genes, hidden, dimensions):
        super().__init__()
        self.gc1 = GraphConvolution(genes, hidden)
        self.gc2 = GraphConvolution(hidden, dimensions)

    def forward(self, expression, adjacency):
        return self.gc2(F.relu(self.gc1(expression, adjacency)), adjacency)


class ExpressionDecoder(nn.Module):
    """Map latent features to a positive reconstructed expression mean."""

    def __init__(self, genes, hidden, dimensions):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(dimensions, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        _advance_linear_draws(hidden, genes, count=2)
        self.mean = nn.Linear(hidden, genes)

    def forward(self, embedding):
        return torch.exp(self.mean(self.hidden(embedding))).clamp(1e-5, 1e6)


class STBSR(nn.Module):
    def __init__(self, genes, spatial_attention_gain):
        super().__init__()
        hidden, dimensions = 128, 64
        self.SGCN = GCN(genes, hidden, dimensions)
        self.FGCN = GCN(genes, hidden, dimensions)
        self.CGCN = GCN(genes, hidden, dimensions)
        self.decoder = ExpressionDecoder(genes, hidden, dimensions)
        self.meta = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.attention = nn.Linear(dimensions, dimensions)
        _advance_linear_draws(dimensions, dimensions)
        _advance_linear_draws(2 * dimensions, dimensions)
        self.fusion = nn.Linear(3 * dimensions, dimensions)
        self.spatial_attention_gain = spatial_attention_gain

    def forward(self, expression, spatial_graph, feature_graph):
        spatial = self.SGCN(expression, spatial_graph)
        feature = self.FGCN(expression, feature_graph)
        mix = self.meta.clamp(0, 1)
        mixed_graph = mix * feature_graph + (1 - mix) * spatial_graph
        mixed = self.CGCN(expression, mixed_graph)
        branches = torch.stack((spatial, mixed, feature), dim=1)
        gains = branches.new_tensor([self.spatial_attention_gain, 1., 1.]).reshape(1, 3, 1)
        attention = F.normalize(self.attention(branches) * gains, p=2, dim=1)
        fused = torch.cat(
            [attention[:, index] * branch
             for index, branch in enumerate((spatial, mixed, feature))],
            dim=1,
        )
        embedding = self.fusion(fused)
        return embedding, self.decoder(embedding), spatial, feature
