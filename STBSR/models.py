import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from .layers import GraphConvolution


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
    """
    Unified STBSR for ablation experiments.
    Supports:
      - branch removal (S/F/C)
      - fixed or learnable meta
      - CAM-like adaptive fusion or direct average
    """
    def __init__(
        self,
        nfeat,
        nhid1,
        nhid2,
        dropout,
        *,
        use_s_branch=True,
        use_f_branch=True,
        use_c_branch=True,
        use_amf=True,
        fixed_meta=None,
    ):
        super().__init__()
        self.use_s_branch = use_s_branch
        self.use_f_branch = use_f_branch
        self.use_c_branch = use_c_branch
        self.use_amf = use_amf
        self.fixed_meta = fixed_meta
        self.dropout = dropout

        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout) if use_s_branch else None
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout) if use_f_branch else None
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout) if use_c_branch else None
        self.ZINB = Decoder(nfeat, nhid1, nhid2)

        if fixed_meta is None:
            self.meta = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        else:
            self.register_buffer("meta_const", torch.tensor([float(fixed_meta)], dtype=torch.float32))
            self.meta = None

        self.MLP_L = MLP_L(64)
        self.proj1 = nn.Linear(64, 64)
        self.proj2 = nn.Linear(128, 64)
        self.proj3 = nn.Linear(192, 64)

    def _get_meta(self, device):
        if self.fixed_meta is None:
            return torch.clamp(self.meta, 0.0, 1.0).to(device)
        return self.meta_const.to(device)

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj) if self.use_s_branch else None
        emb2 = self.FGCN(x, fadj) if self.use_f_branch else None

        conadj = None
        com = None
        if self.use_c_branch:
            meta = self._get_meta(x.device)
            conadj = meta * fadj + (1.0 - meta) * sadj
            com = self.CGCN(x, conadj)

        active = [b for b in [emb1, com, emb2] if b is not None]
        if len(active) == 0:
            raise RuntimeError("At least one branch must be enabled.")

        if len(active) == 1:
            emb = active[0]
        elif self.use_amf:
            stacked = torch.stack(active, dim=1)   # [N, B, 64]
            attn = self.MLP_L(stacked)
            attn = F.normalize(attn, p=2, dim=1)
            weighted = []
            for i, branch in enumerate(active):
                weighted.append(attn[:, i] * branch)
            fused = torch.cat(weighted, dim=1)
            if len(active) == 2:
                emb = self.proj2(fused)
            else:
                emb = self.proj3(fused)
        else:
            emb = torch.stack(active, dim=0).mean(dim=0)
            emb = self.proj1(emb)

        pi, disp, mean = self.ZINB(emb)
        return emb, pi, disp, mean, emb1, emb2, com
