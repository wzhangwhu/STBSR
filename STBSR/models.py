import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

try:
    from .layers import GraphConvolution
except ImportError:
    from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class Decoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.BatchNorm1d(nhid1),
            nn.ReLU()
        )
        self.pi = nn.Linear(nhid1, nfeat)
        self.disp = nn.Linear(nhid1, nfeat)
        self.mean = nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super().__init__()
        self.wl = Linear(n_mlp, 64)

    def forward(self, mlp_in):
        return self.wl(mlp_in)


class STBSR(nn.Module):
    """Three-branch STBSR encoder with adaptive representation fusion."""

    def __init__(
        self,
        nfeat,
        nhid1,
        nhid2,
        dropout,
    ):
        super().__init__()
        self.dropout = dropout

        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = Decoder(nfeat, nhid1, nhid2)

        self.meta = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        self.MLP_L = MLP_L(64)
        # Preserve the published seeded initialization sequence.
        self.proj1 = nn.Linear(64, 64)
        self.proj2 = nn.Linear(128, 64)
        self.proj3 = nn.Linear(192, 64)

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj)
        emb2 = self.FGCN(x, fadj)
        meta = torch.clamp(self.meta, 0.0, 1.0).to(x.device)
        fusion_adjacency = meta * fadj + (1.0 - meta) * sadj
        fusion_embedding = self.CGCN(
            x,
            fusion_adjacency,
        )

        branches = torch.stack(
            [emb1, fusion_embedding, emb2],
            dim=1,
        )
        attention = F.normalize(self.MLP_L(branches), p=2, dim=1)
        fused = torch.cat(
            [
                attention[:, 0] * emb1,
                attention[:, 1] * fusion_embedding,
                attention[:, 2] * emb2,
            ],
            dim=1,
        )
        emb = self.proj3(fused)

        pi, disp, mean = self.ZINB(emb)
        return (
            emb,
            pi,
            disp,
            mean,
            emb1,
            emb2,
            fusion_embedding,
        )
