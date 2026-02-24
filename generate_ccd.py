#!/usr/bin/env python3
"""
Generate Central Composite Design (CCD) experiment configurations
for fall detection parameter optimization.

Generates a CCD for each of the 4 YOLO models with 3 factors:
  1. Detection confidence threshold (--conf)
  2. Fall detection sensitivity threshold (--fall-threshold)
  3. Angle threshold (--angle-threshold)

Output: ccd_experiments.json — consumed by run_experiments.py
"""

import json
import itertools
import math
import os

# ============================================================
# ADJUSTABLE PARAMETERS — Modify these to suit your needs
# ============================================================

# Alpha factor for CCD (distance of axial/star points from center)
# Common choices:
#   sqrt(k)    — rotatable design  (k = number of factors)
#   1.0        — face-centered design (CCF)
#   2^(k/4)    — orthogonal design
ALPHA = 1.1  # ~1.7321 for rotatable design with k=3

# Number of center-point replicates
N_CENTER = 3

# Factor definitions
# Adjust "low" and "high" to change the ±1 coded range.
# The center point is automatically (low + high) / 2.
FACTORS = {
    "conf": {
        "cli_arg": "--conf",
        "low": 0.3,          # coded level -1
        "high": 0.7,         # coded level +1
        "description": "Detection confidence threshold (0–1)",
    },
    "fall_threshold": {
        "cli_arg": "--fall-threshold",
        "low": 0.2,          # coded level -1
        "high": 0.6,         # coded level +1
        "description": "Fall detection sensitivity threshold (0–1)",
    },
    "angle_threshold": {
        "cli_arg": "--angle-threshold",
        "low": 30,           # coded level -1
        "high": 60,          # coded level +1
        "description": "Angle threshold in degrees",
    },
}

# YOLO model files in the project root
MODELS = [
    #"yolov12m.pt",
    "yolov12n.pt",
    "yolov12n1.pt",
    "yolov8n.pt",
]

# ============================================================
# CCD generation logic
# ============================================================

def coded_to_actual(coded_value, low, high):
    """Convert a coded CCD level to an actual parameter value."""
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0
    return center + coded_value * half_range


def generate_ccd_coded(n_factors, alpha, n_center):
    """Return a list of CCD design points in coded coordinates.

    Each point is a dict with keys 'type' and 'coded' (list of floats).
    """
    points = []

    # 1. Full-factorial cube  (2^k points)
    for combo in itertools.product([-1, 1], repeat=n_factors):
        points.append({"type": "factorial", "coded": list(combo)})

    # 2. Axial / star points  (2k points)
    for i in range(n_factors):
        for sign in [-1, 1]:
            coded = [0.0] * n_factors
            coded[i] = sign * alpha
            points.append({"type": "axial", "coded": coded})

    # 3. Center points
    for _ in range(n_center):
        points.append({"type": "center", "coded": [0.0] * n_factors})

    return points


def clamp_factor(name, value):
    """Apply reasonable physical bounds to a factor value."""
    if name == "conf":
        return max(0.05, min(0.99, value))
    elif name == "fall_threshold":
        return max(0.01, min(0.99, value))
    elif name == "angle_threshold":
        return max(5.0, min(85.0, value))
    return value


def main():
    factor_names = list(FACTORS.keys())
    n_factors = len(factor_names)

    # Generate coded design matrix
    design = generate_ccd_coded(n_factors, ALPHA, N_CENTER)

    # Build per-model experiment lists
    experiments = {}
    for model in MODELS:
        model_experiments = []
        for idx, point in enumerate(design):
            exp = {
                "experiment_id": idx + 1,
                "point_type": point["type"],
                "coded_levels": {},
                "actual_levels": {},
                "cli_args": {},
            }
            for j, fname in enumerate(factor_names):
                f = FACTORS[fname]
                coded = point["coded"][j]
                actual = coded_to_actual(coded, f["low"], f["high"])
                actual = clamp_factor(fname, actual)
                actual = round(actual, 4)

                exp["coded_levels"][fname] = round(coded, 4)
                exp["actual_levels"][fname] = actual
                exp["cli_args"][f["cli_arg"]] = actual

            model_experiments.append(exp)
        experiments[model] = model_experiments

    # Assemble output
    output = {
        "design_info": {
            "type": "Central Composite Design (CCD)",
            "n_factors": n_factors,
            "alpha": ALPHA,
            "n_center_points": N_CENTER,
            "n_factorial_points": 2 ** n_factors,
            "n_axial_points": 2 * n_factors,
            "total_runs_per_model": len(design),
            "total_models": len(MODELS),
            "total_runs": len(design) * len(MODELS),
            "factors": FACTORS,
            "models": MODELS,
        },
        "experiments": experiments,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccd_experiments.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Summary ----
    print("=" * 60)
    print("  Central Composite Design Generated")
    print("=" * 60)
    print(f"  Factors          : {n_factors}")
    print(f"  Alpha            : {ALPHA}")
    print(f"  Factorial points : {2 ** n_factors}")
    print(f"  Axial points     : {2 * n_factors}")
    print(f"  Center points    : {N_CENTER}")
    print(f"  Runs per model   : {len(design)}")
    print(f"  Models           : {len(MODELS)}")
    print(f"  Total experiments: {len(design) * len(MODELS)}")
    print(f"\n  Saved to: {output_path}")
    print()

    # Factor ranges including axial extremes
    print("  Factor Ranges (axial_low … low … center … high … axial_high):")
    for fname, f in FACTORS.items():
        center = (f["low"] + f["high"]) / 2.0
        half   = (f["high"] - f["low"]) / 2.0
        ax_lo  = clamp_factor(fname, center - ALPHA * half)
        ax_hi  = clamp_factor(fname, center + ALPHA * half)
        print(f"    {fname:20s}: {ax_lo:.4f}  {f['low']:.4f}  {center:.4f}  {f['high']:.4f}  {ax_hi:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
