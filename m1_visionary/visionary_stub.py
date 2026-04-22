"""
ICARUS-X — M1 Visionary: Stub Module

Generates realistic mock Active Region (AR) feature vectors
when the YOLOv10 model is still training on Kaggle.

The stub returns the EXACT same dict structure as the real visionary
module, with values based on September 2017 storm observations.

Inputs:  None (generates from noise + realistic baselines)
Outputs: M1 output contract dict (12-dim AR feature vector)
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from loguru import logger


# ── Sept 2017 X9.3 flare reference values ───────────────
# These represent an extremely active AR (NOAA 12673)
SEPT_2017_BASELINE = {
    "n_regions_detected": 2,
    "f0": 1.24,   # total unsigned magnetic flux (normalized)
    "f1": 0.08,   # flux imbalance
    "f2": 0.31,   # polarity inversion line length
    "f3": 0.19,   # gradient along PIL
    "f4": 0.44,   # shear angle
    "f5": 0.52,   # free magnetic energy proxy
    "f6": 0.73,   # twist parameter
    "f7": 1.20,   # current helicity
    "f8": 2.91,   # R-value (schrijver)
    "f9": 0.67,   # WL_SG (gradient-weighted PIL)
    "f10": 0.88,  # effective connected magnetic field
    "f11": 0.71,  # complexity index
}

# ── Quiet Sun baseline (no significant AR) ──────────────
QUIET_BASELINE = {
    "n_regions_detected": 0,
    "f0": 0.05, "f1": 0.01, "f2": 0.02, "f3": 0.01,
    "f4": 0.03, "f5": 0.02, "f6": 0.04, "f7": 0.03,
    "f8": 0.10, "f9": 0.02, "f10": 0.05, "f11": 0.03,
}


def generate_stub_features(
    timestamp: Optional[str] = None,
    scenario: str = "storm",
    noise_std: float = 0.05,
) -> Dict[str, Any]:
    """
    Generate realistic mock AR feature vector.

    Args:
        timestamp: ISO timestamp (defaults to now)
        scenario: 'storm' for Sept 2017-like, 'quiet' for no AR
        noise_std: Standard deviation of random noise added

    Returns:
        M1 output contract dict
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    baseline = SEPT_2017_BASELINE if scenario == "storm" else QUIET_BASELINE
    rng = np.random.RandomState()

    features = {"timestamp": timestamp}

    # Add noise to make it look "live"
    features["n_regions_detected"] = max(0, baseline["n_regions_detected"] + rng.choice([-1, 0, 0, 1]))

    for i in range(12):
        key = f"f{i}"
        base_val = baseline[key]
        noisy_val = base_val + rng.normal(0, noise_std * abs(base_val) + 0.01)
        features[key] = round(max(0, float(noisy_val)), 4)

    logger.debug(f"[!] STUB M1 -- AR features generated (scenario={scenario})")
    return features


def get_ar_feature_vector(
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for M1 — called by model_runner.
    In stub mode, returns mock features. When real YOLO is loaded,
    this function is replaced by visionary.py.
    """
    logger.warning("[!] STUB MODE -- M1 Visionary (YOLOv10 not loaded)")
    return generate_stub_features(timestamp, scenario="storm")


if __name__ == "__main__":
    # Demo both scenarios
    print("=== Storm Scenario (Sept 2017) ===")
    storm = generate_stub_features(scenario="storm")
    for k, v in storm.items():
        print(f"  {k}: {v}")

    print("\n=== Quiet Scenario ===")
    quiet = generate_stub_features(scenario="quiet")
    for k, v in quiet.items():
        print(f"  {k}: {v}")
