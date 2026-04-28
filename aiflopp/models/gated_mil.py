import argparse
from typing import List, Tuple

import torch
from torch import nn


class AttentionMILGated(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, hidden_dim: int, dropout: float):

        # E' la versione avanzata del BaseMIL, con un gate che pesa l'attenzione.
        # La formula diventa W*tan(V*h)*sigmoid(U*h) dove U e V portano entrambi a attention_dim.
        # poi W che porta da attention_dim a 1 (seguita poi dalla softmax nella forward)

        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, attention_dim), # V
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, attention_dim), # U
            nn.Sigmoid(),
        )
        self.attention_score = nn.Linear(attention_dim, 1) # W

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bags: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits: list[torch.Tensor] = []
        attn_weights: list[torch.Tensor] = []

        for bag in bags:
            v = self.attention_v(bag)
            u = self.attention_u(bag)

            # weight each v patch score by u patch gate value, returns (num_patches,)
            scores = self.attention_score(v * u).squeeze(-1)
            weights = torch.softmax(scores, dim=0)

            bag_repr = torch.sum(weights.unsqueeze(-1) * bag, dim=0)
            logit = self.classifier(bag_repr)

            logits.append(logit.squeeze(-1))
            attn_weights.append(weights)

        return torch.stack(logits), attn_weights


def validate_gated_mil_args(args: argparse.Namespace) -> None:
    if args.attention_dim <= 0:
        raise ValueError("--attention-dim must be > 0.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be > 0.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


def build_gated_mil(args: argparse.Namespace) -> nn.Module:
    return AttentionMILGated(
        input_dim=args.input_dim,
        attention_dim=args.attention_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def get_gated_mil_config(args: argparse.Namespace) -> dict:
    return {
        "input_dim": args.input_dim,
        "attention_dim": args.attention_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
