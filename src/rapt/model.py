from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.15):
        super().__init__()
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attn = nn.Linear(int(hidden_dim), 1)

    def forward(self, x):
        h, _ = self.lstm(x)             # [B,T,H]
        a = self.attn(h).squeeze(-1)    # [B,T]
        w = torch.softmax(a, dim=1)
        z = torch.sum(h * w.unsqueeze(-1), dim=1)
        return z, w


class ModalityGatedFusion(nn.Module):
    def __init__(self, input_dims: Sequence[int], gating_dim=96, dropout=0.15):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(d), int(gating_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
            for d in input_dims
        ])
        self.gate = nn.Sequential(
            nn.Linear(int(gating_dim) * len(input_dims), int(gating_dim)),
            nn.ReLU(),
            nn.Linear(int(gating_dim), len(input_dims)),
        )
        self.output_dim = int(gating_dim)

    def forward(self, zs):
        p = [layer(z) for layer, z in zip(self.proj, zs)]
        cat = torch.cat(p, dim=-1)
        weights = torch.softmax(self.gate(cat), dim=-1)
        stack = torch.stack(p, dim=1)
        fused = torch.sum(stack * weights.unsqueeze(-1), dim=1)
        return fused, weights


class MultiSourceRAPT(nn.Module):
    def __init__(
        self,
        veg_dim: int,
        prod_dim: int,
        cli_dim: int,
        aux_dim: int,
        n_tasks: int,
        n_units: int,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_tasks = int(n_tasks)

        self.veg_enc = TemporalAttentionEncoder(
            veg_dim, cfg.veg_hidden, cfg.num_layers, cfg.dropout
        )
        self.prod_enc = TemporalAttentionEncoder(
            prod_dim, cfg.prod_hidden, cfg.num_layers, cfg.dropout
        )
        self.cli_enc = TemporalAttentionEncoder(
            cli_dim, cfg.cli_hidden, cfg.num_layers, cfg.dropout
        )

        self.aux_enc = nn.Sequential(
            nn.Linear(int(aux_dim), int(cfg.aux_hidden)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
        )

        self.use_unit_embedding = bool(cfg.use_unit_embedding)
        if self.use_unit_embedding:
            self.unit_embedding = nn.Embedding(
                int(n_units), int(cfg.unit_embedding_dim)
            )
            unit_dim = int(cfg.unit_embedding_dim)
        else:
            self.unit_embedding = None
            unit_dim = 0

        modality_dims = [
            int(cfg.veg_hidden),
            int(cfg.prod_hidden),
            int(cfg.cli_hidden),
            int(cfg.aux_hidden),
        ]

        self.use_modality_gating = bool(cfg.use_modality_gating)
        if self.use_modality_gating:
            self.gating = ModalityGatedFusion(
                modality_dims, cfg.gating_dim, cfg.dropout
            )
            fusion_in = int(cfg.gating_dim) + unit_dim
        else:
            self.gating = None
            fusion_in = sum(modality_dims) + unit_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, int(cfg.fusion_hidden)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.fusion_hidden), int(cfg.fusion_hidden)),
            nn.ReLU(),
        )

        def make_head():
            return nn.Sequential(
                nn.Linear(int(cfg.fusion_hidden), int(cfg.head_hidden)),
                nn.ReLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(int(cfg.head_hidden), 1),
            )

        self.shared_head = bool(cfg.shared_head)
        if self.shared_head:
            self.head = make_head()
            self.heads = None
        else:
            self.head = None
            self.heads = nn.ModuleList([make_head() for _ in range(self.n_tasks)])

    def forward(self, veg, prod, climate, aux, task_idx, unit_idx):
        z_veg, att_veg = self.veg_enc(veg)
        z_prod, att_prod = self.prod_enc(prod)
        z_cli, att_cli = self.cli_enc(climate)
        z_aux = self.aux_enc(aux)

        zs = [z_veg, z_prod, z_cli, z_aux]

        gate_weights = None
        if self.use_modality_gating:
            z, gate_weights = self.gating(zs)
        else:
            z = torch.cat(zs, dim=-1)

        if self.use_unit_embedding:
            z_unit = self.unit_embedding(unit_idx.long())
            z = torch.cat([z, z_unit], dim=-1)

        shared = self.fusion(z)

        if self.shared_head:
            pred = self.head(shared).squeeze(1)
        else:
            # Vectorized sample-specific task-head routing.
            all_pred = torch.cat([head(shared) for head in self.heads], dim=1)
            pred = all_pred.gather(1, task_idx.long().view(-1, 1)).squeeze(1)

        extras = {
            "att_veg": att_veg,
            "att_prod": att_prod,
            "att_cli": att_cli,
            "gate_weights": gate_weights,
            "shared": shared,
        }
        return pred, extras
