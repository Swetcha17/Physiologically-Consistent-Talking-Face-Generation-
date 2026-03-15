"""
bvp_generator.py — Synthetic BVP Waveform Generator

OWNER: Ankit
Interface created by Swetcha. Ankit replaces the placeholder with
realistic waveform (systolic peak, dicrotic notch, HRV, RSA).
See docs/ for parameter specs from George.
"""

import numpy as np


def generate_bvp(heart_rate=72, duration=10, fps=25, hrv_std=0.025,
                 respiratory_rate=15):
    """
    Generate a synthetic BVP (blood volume pulse) waveform.

    Returns:
        bvp_signal: np.ndarray shape (duration*fps,) normalized [0, 1]
        timestamps: np.ndarray of timestamps in seconds
    """
    num_samples = int(duration * fps)
    timestamps = np.linspace(0, duration, num_samples, endpoint=False)

    # TODO (Ankit): Replace with realistic BVP including:
    #   - Systolic peak + dicrotic notch
    #   - HRV (beat-to-beat RR jitter)
    #   - Respiratory sinus arrhythmia
    # Refs: Allen 2007, pyHRV (github.com/PGomes92/pyhrv)

    hr_hz = heart_rate / 60.0
    bvp = (
        0.6 * np.sin(2 * np.pi * hr_hz * timestamps) +
        0.3 * np.sin(2 * np.pi * 2 * hr_hz * timestamps - np.pi / 4) +
        0.1 * np.sin(2 * np.pi * 3 * hr_hz * timestamps - np.pi / 3)
    )
    bvp = (bvp - bvp.min()) / (bvp.max() - bvp.min() + 1e-8)
    return bvp, timestamps


if __name__ == "__main__":
    bvp, t = generate_bvp(heart_rate=72, duration=5, fps=25)
    print(f"BVP: shape={bvp.shape}, range=[{bvp.min():.3f}, {bvp.max():.3f}]")
