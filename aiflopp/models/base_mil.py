import argparse
from typing import List, Tuple

import torch
from torch import nn


class AttentionMILBase(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, hidden_dim: int, dropout: float, output_dim: int = 1):

        # E' la versione piu basic del MIL
        # W*tan(V*h) come attenzione, quindi una matrice V di peso per ogni patch che porta da una dimension attention_dim
        # poi W che la porta a 1 (seguita poi dalla softmax nella forward)

        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim), # V
            nn.Tanh(),
            nn.Linear(attention_dim, 1), # W
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, bags: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits: list[torch.Tensor] = []
        attn_weights: list[torch.Tensor] = []

        for bag in bags:
            scores = self.attention(bag).squeeze(-1) # (num_patches,)
            weights = torch.softmax(scores, dim=0)
            bag_repr = torch.sum(weights.unsqueeze(-1) * bag, dim=0)
            logit = self.classifier(bag_repr)

            logits.append(logit.squeeze(-1) if logit.shape[-1] == 1 else logit)
            attn_weights.append(weights)

        return torch.stack(logits), attn_weights


def validate_base_mil_args(args: argparse.Namespace) -> list[str]:
    if args.attention_dim <= 0:
        raise ValueError("--attention-dim must be > 0.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be > 0.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if getattr(args, "num_classes", 1) <= 0:
        raise ValueError("--num-classes must be > 0.")
    if getattr(args, "output_dim", 1) <= 0:
        raise ValueError("Model output_dim must be > 0.")
    
    # Return the required args names
    return ["input_dim", "attention_dim", "hidden_dim", "dropout", "output_dim"]


def build_base_mil(args: argparse.Namespace,) -> nn.Module:
    return AttentionMILBase(
        input_dim=args.input_dim,
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        output_dim=getattr(args, "output_dim", getattr(args, "num_classes", 1)),
    )
