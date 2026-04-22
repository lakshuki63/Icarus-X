"""
ICARUS-X — M2 Predictor: BiGRU Seq2Seq Model with Bahdanau Attention

Architecture:
  - Bidirectional GRU encoder (2 layers, 128 hidden)
  - Bahdanau additive attention mechanism
  - Multi-horizon decoder → 8 Kp forecasts + confidence intervals

Inputs:  (batch, seq_len=24, n_features=19) solar wind + AR features
Outputs: (batch, 8) predicted Kp at horizons [3,6,9,12,15,18,21,24]h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """Additive attention mechanism for sequence-to-one mapping."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, hidden) — decoder state
            keys:  (batch, seq_len, hidden) — encoder outputs
        Returns:
            context: (batch, hidden)
            weights: (batch, seq_len)
        """
        query_expanded = self.W_q(query).unsqueeze(1)  # (B, 1, H)
        keys_proj = self.W_k(keys)                      # (B, T, H)
        energy = self.v(torch.tanh(query_expanded + keys_proj))  # (B, T, 1)
        weights = F.softmax(energy.squeeze(-1), dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, H)
        return context, weights


class BiGRUPredictor(nn.Module):
    """Bidirectional GRU with attention for multi-horizon Kp forecasting."""

    def __init__(
        self,
        input_size: int = 19,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_horizons: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_horizons = n_horizons

        # Encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Project bidirectional output to hidden_size
        self.enc_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Attention
        self.attention = BahdanauAttention(hidden_size)

        # Decoder head — one per horizon for specialization
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3),  # [mean, log_var_low, log_var_high]
            )
            for _ in range(n_horizons)
        ])

        # MC Dropout for uncertainty (kept active during inference)
        self.mc_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> dict:
        """
        Args:
            x: (batch, seq_len, input_size)
            return_attention: if True, include attention weights
        Returns:
            dict with kp_pred, kp_ci_low, kp_ci_high, attention_weights
        """
        batch_size = x.size(0)

        # Encode
        enc_out, h_n = self.encoder(x)  # enc_out: (B, T, 2*H)
        enc_proj = self.enc_proj(enc_out)  # (B, T, H)

        # Final hidden state (concat forward + backward last layers)
        h_fwd = h_n[-2]  # (B, H)
        h_bwd = h_n[-1]  # (B, H)
        decoder_state = h_fwd + h_bwd  # (B, H)

        # Attention
        context, attn_weights = self.attention(decoder_state, enc_proj)

        # Concat context + decoder state
        combined = torch.cat([context, decoder_state], dim=-1)  # (B, 2*H)
        combined = self.mc_dropout(combined)

        # Decode each horizon
        kp_preds = []
        kp_ci_lows = []
        kp_ci_highs = []

        for head in self.horizon_heads:
            out = head(combined)  # (B, 3)
            mean = out[:, 0]
            ci_low = mean - F.softplus(out[:, 1])
            ci_high = mean + F.softplus(out[:, 2])

            kp_preds.append(mean)
            kp_ci_lows.append(ci_low)
            kp_ci_highs.append(ci_high)

        result = {
            "kp_pred": torch.stack(kp_preds, dim=1),        # (B, 8)
            "kp_ci_low": torch.stack(kp_ci_lows, dim=1),    # (B, 8)
            "kp_ci_high": torch.stack(kp_ci_highs, dim=1),  # (B, 8)
        }

        if return_attention:
            result["attention_weights"] = attn_weights  # (B, T)

        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> dict:
        """MC Dropout inference for Bayesian uncertainty estimation."""
        self.train()  # Keep dropout active

        all_preds = []
        all_attn = []

        for _ in range(n_samples):
            out = self.forward(x, return_attention=True)
            all_preds.append(out["kp_pred"])
            all_attn.append(out["attention_weights"])

        preds = torch.stack(all_preds)  # (S, B, 8)
        attns = torch.stack(all_attn)   # (S, B, T)

        result = {
            "kp_pred": preds.mean(0),          # (B, 8)
            "kp_std": preds.std(0),            # (B, 8)
            "kp_ci_low": preds.quantile(0.05, dim=0),
            "kp_ci_high": preds.quantile(0.95, dim=0),
            "attention_weights": attns.mean(0),  # (B, T)
        }

        self.eval()
        return result
