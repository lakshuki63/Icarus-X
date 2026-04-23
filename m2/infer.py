
"""
ICARUS-X M2 — infer.py
Single-call inference for the M5 Architect module.
M5 calls: from infer import run_forecast
"""

import sys, os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# ── Model definition (self-contained, no external import) ────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1,  bias=False)
    def forward(self, enc):
        energy  = torch.tanh(self.W_h(enc))
        scores  = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc).squeeze(1)
        return context, weights

class BiGRUPredictor(nn.Module):
    def __init__(self, sw_dim=4, ar_dim=12, hidden=256,
                 n_layers=2, dropout=0.15, n_horizons=8):
        super().__init__()
        self.gru = nn.GRU(sw_dim, hidden, n_layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden)
        self.drop      = nn.Dropout(dropout)
        self.fusion    = nn.Sequential(
            nn.Linear(hidden * 2 + ar_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(inplace=True)
        )
        self.decoder   = nn.Linear(hidden, n_horizons)
    def forward(self, x_sw, x_ar):
        enc, _ = self.gru(x_sw)
        enc    = self.drop(enc)
        ctx, attn = self.attention(enc)
        fused  = self.drop(torch.cat([ctx, x_ar], -1))
        return self.decoder(self.fusion(fused)), attn
    @torch.no_grad()
    def mc_dropout_predict(self, x_sw, x_ar, n_passes=50):
        self.train()
        preds = [self.forward(x_sw, x_ar)[0].unsqueeze(0) for _ in range(n_passes)]
        preds = torch.cat(preds, dim=0)
        self.eval()
        _, attn = self.forward(x_sw, x_ar)
        return (preds.mean(0), preds.std(0),
                torch.quantile(preds, 0.05, dim=0),
                torch.quantile(preds, 0.95, dim=0), attn)

# ── Constants ─────────────────────────────────────────────────────────────
HORIZONS = [3, 6, 9, 12, 15, 18, 21, 24]
SW_COLS  = ['Bz', 'Vsw', 'Np', 'Pdyn']

# ── Main inference function ────────────────────────────────────────────────
def run_forecast(
    solar_wind_df,
    ar_features,
    ckpt_path   = 'models/best.pt',
    scalers_path = 'models/scalers.pkl',
    n_passes    = 50,
    device      = None
):
    """
    Main inference function — called by M5 model_runner.py

    Args:
        solar_wind_df : pd.DataFrame with columns [Bz,Vsw,Np,Pdyn], >= 24 rows
        ar_features   : list or np.ndarray of shape [12] — from M1
        ckpt_path     : path to best.pt checkpoint
        scalers_path  : path to scalers.pkl
        n_passes      : MC Dropout passes for uncertainty (default 50)
        device        : torch device (auto-detected if None)

    Returns:
        dict: run_timestamp + list of 8 horizon dicts
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Normalise solar wind ──────────────────────────────────────────────
    scalers = joblib.load(scalers_path)
    sw = solar_wind_df[SW_COLS].copy().astype(float)
    for col, s in scalers.items():
        sw[col] = (sw[col] - s['mean']) / max(s['std'], 1e-8)

    # Ensure exactly 24 rows
    if len(sw) < 24:
        pad = pd.DataFrame(np.zeros((24 - len(sw), 4)), columns=SW_COLS)
        sw  = pd.concat([pad, sw], ignore_index=True)
    sw_arr = sw.tail(24).values.astype(np.float32)
    sw_arr = np.nan_to_num(sw_arr, nan=0.0)

    # ── Build tensors ────────────────────────────────────────────────────
    x_sw = torch.tensor(sw_arr).unsqueeze(0).to(device)                   # [1, 24, 4]
    x_ar = torch.tensor(np.array(ar_features, dtype=np.float32)
                        ).unsqueeze(0).to(device)                          # [1, 12]

    # ── Load model ───────────────────────────────────────────────────────
    model = BiGRUPredictor().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # ── MC Dropout inference ─────────────────────────────────────────────
    mean, std, p5, p95, attn = model.mc_dropout_predict(x_sw, x_ar, n_passes)

    mean = np.clip(mean.squeeze(0).cpu().numpy(), 0, 9)
    std  = std.squeeze(0).cpu().numpy()
    p5   = np.clip(p5.squeeze(0).cpu().numpy(),  0, 9)
    p95  = np.clip(p95.squeeze(0).cpu().numpy(), 0, 9)
    attn = attn.squeeze(0).cpu().numpy().tolist()  # [24] hourly attention weights

    return {
        'run_timestamp': datetime.utcnow().strftime(' %Y-%m-%dT%H:%M:%S').strip(),
        'horizons': [
            {
                'horizon_hr'       : int(HORIZONS[i]),
                'kp_predicted'    : round(float(mean[i]), 3),
                'kp_ci_low'       : round(float(p5[i]),   3),
                'kp_ci_high'      : round(float(p95[i]),  3),
                'kp_std'          : round(float(std[i]),  3),
                'attention_weights': attn,
            }
            for i in range(8)
        ]
    }


if __name__ == '__main__':
    import json
    demo_sw = pd.DataFrame({
        'Bz'  : np.full(24, -10.0),
        'Vsw' : np.full(24,  600.0),
        'Np'  : np.full(24,   10.0),
        'Pdyn': np.full(24,    2.5),
    })
    demo_ar = [0.1] * 12
    out = run_forecast(demo_sw, demo_ar)
    # Print without attention weights
    for h in out['horizons']:
        h['attention_weights'] = '[... 24 floats ...]'   # truncate for display
    print(json.dumps(out, indent=2))
