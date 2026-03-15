"""
inject_rppg.py — rPPG Signal Injection into Video Frames

Takes video frames + skin ROI masks + BVP signal and modulates
the green channel intensity in skin regions to embed a heartbeat signal.

The modulation is subtle (~1-2% intensity change) and spatially localized
to skin regions (forehead, cheeks, nose) — mimicking how real rPPG signals
appear in camera-captured facial video.

Usage:
    python src/inject_rppg.py --video results/test_video.mp4 --heart_rate 72
    python src/inject_rppg.py --test

Author: Swetcha
"""

import os
import sys
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from generate_video import extract_frames, frames_to_video, create_test_video, load_config
from face_parsing import FaceParser, detect_skin_hsv
from bvp_generator import generate_bvp


# =============================================================================
# Signal Injection
# =============================================================================

class RPPGInjector:
    """
    Inject rPPG signal into video frames by modulating pixel intensity
    in skin regions according to a BVP waveform.

    The modulation targets the green channel primarily, as green light
    penetrates skin most effectively and carries the strongest blood
    volume pulse signal (Verkruysse et al., 2008).
    """

    def __init__(self, amplitude=0.015, channel="green",
                 spatial_smoothing=True, blur_ksize=15):
        """
        Args:
            amplitude: modulation strength as fraction of pixel value.
                       Real rPPG signals are ~0.5-2% of intensity.
                       Default 0.015 = 1.5%
            channel: which channel(s) to modulate
                     'green' — green channel only (strongest rPPG response)
                     'all_chrom' — chrominance model (green+red, inverse weights)
            spatial_smoothing: smooth mask edges for natural blending
            blur_ksize: Gaussian blur kernel for mask smoothing
        """
        self.amplitude = amplitude
        self.channel = channel
        self.spatial_smoothing = spatial_smoothing
        self.blur_ksize = blur_ksize

    def inject(self, frames, masks, bvp_signal):
        """
        Inject BVP signal into video frames.

        Args:
            frames: list of np.ndarray (H, W, 3) BGR uint8
            masks: list of np.ndarray (H, W) float32 [0, 1] skin masks
            bvp_signal: np.ndarray (N,) normalized BVP signal [0, 1]

        Returns:
            injected_frames: list of modified frames with rPPG signal
            injection_log: dict with injection metadata
        """
        n_frames = len(frames)
        n_masks = len(masks)
        n_bvp = len(bvp_signal)

        # Ensure BVP signal length matches frame count
        if n_bvp != n_frames:
            # Resample BVP to match frame count
            x_old = np.linspace(0, 1, n_bvp)
            x_new = np.linspace(0, 1, n_frames)
            bvp_signal = np.interp(x_new, x_old, bvp_signal)

        # Ensure mask count matches frame count
        if n_masks < n_frames:
            # Repeat last mask for remaining frames
            masks = masks + [masks[-1]] * (n_frames - n_masks)

        # Center BVP around zero for bidirectional modulation
        # Original BVP is [0,1], center to [-0.5, 0.5] then scale by amplitude
        bvp_centered = bvp_signal - 0.5  # now in [-0.5, 0.5]
        modulation = bvp_centered * 2.0 * self.amplitude  # scale to [-amp, +amp]

        injected_frames = []
        actual_changes = []

        for i in tqdm(range(n_frames), desc="Injecting rPPG", unit="f"):
            frame = frames[i].copy().astype(np.float32)
            mask = masks[i]

            if self.spatial_smoothing and mask.max() > 0:
                mask = cv2.GaussianBlur(mask, (self.blur_ksize, self.blur_ksize), 0)

            # Current modulation value for this frame
            mod = modulation[i]

            if self.channel == "green":
                # Modulate green channel only (BGR index 1)
                # delta = pixel_value * modulation * mask
                delta = frame[:, :, 1] * mod * mask
                frame[:, :, 1] = np.clip(frame[:, :, 1] + delta, 0, 255)

            elif self.channel == "all_chrom":
                # Chrominance-based modulation (De Haan & Jeanne, 2013)
                # Green and red modulated with opposite signs
                # This better matches the physiological model
                delta_g = frame[:, :, 1] * mod * mask
                delta_r = frame[:, :, 2] * (-0.5 * mod) * mask
                frame[:, :, 1] = np.clip(frame[:, :, 1] + delta_g, 0, 255)
                frame[:, :, 2] = np.clip(frame[:, :, 2] + delta_r, 0, 255)

            injected_frames.append(frame.astype(np.uint8))

            # Track actual pixel change for logging
            if mask.max() > 0:
                skin_pixels = mask > 0.5
                if skin_pixels.any():
                    orig_mean = frames[i][:, :, 1][skin_pixels].mean()
                    new_mean = frame[:, :, 1][skin_pixels].mean()
                    actual_changes.append(abs(new_mean - orig_mean))

        # Injection metadata
        injection_log = {
            "n_frames": n_frames,
            "amplitude": self.amplitude,
            "channel": self.channel,
            "bvp_min": float(bvp_signal.min()),
            "bvp_max": float(bvp_signal.max()),
            "modulation_range": [float(modulation.min()), float(modulation.max())],
            "avg_pixel_change": float(np.mean(actual_changes)) if actual_changes else 0,
            "max_pixel_change": float(np.max(actual_changes)) if actual_changes else 0,
        }

        return injected_frames, injection_log

    def compute_frame_diff(self, original_frames, injected_frames, mask_idx=0):
        """
        Compute and visualize the difference between original and injected frames.
        Useful for debugging and verifying injection.

        Args:
            original_frames: list of original frames
            injected_frames: list of injected frames
            mask_idx: frame index to visualize

        Returns:
            diff_amplified: amplified difference image for visualization
        """
        orig = original_frames[mask_idx].astype(np.float32)
        inj = injected_frames[mask_idx].astype(np.float32)

        diff = inj - orig

        # Amplify for visualization (real diff is very subtle)
        diff_amplified = np.clip(diff * 50 + 128, 0, 255).astype(np.uint8)

        return diff_amplified


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inject rPPG signal into video")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--output", type=str, default="results/injected_video.mp4")
    parser.add_argument("--heart_rate", type=int, default=72, help="Target HR in BPM")
    parser.add_argument("--amplitude", type=float, default=0.015,
                        help="Modulation amplitude (fraction of pixel intensity)")
    parser.add_argument("--channel", type=str, default="green",
                        choices=["green", "all_chrom"])
    parser.add_argument("--use_hsv", action="store_true",
                        help="Use HSV skin detection instead of BiSeNet")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # --- Get video ---
    if args.test:
        print("\n[TEST] Signal injection on synthetic video...")
        video_path = create_test_video("results/test_video.mp4")
    elif args.video:
        video_path = args.video
    else:
        parser.error("--video required (or use --test)")

    # --- Extract frames ---
    print("\n[1/4] Extracting frames...")
    frames, fps, meta = extract_frames(video_path)

    # --- Face parsing ---
    print("\n[2/4] Generating skin masks...")
    if args.use_hsv:
        masks = [detect_skin_hsv(f) for f in tqdm(frames, desc="HSV detection", unit="f")]
    else:
        try:
            fp = FaceParser()
            masks = fp.parse_frames(frames)
        except Exception as e:
            print(f"  BiSeNet failed ({e}), falling back to HSV detection...")
            masks = [detect_skin_hsv(f) for f in tqdm(frames, desc="HSV fallback", unit="f")]

    # --- Generate BVP signal ---
    print(f"\n[3/4] Generating BVP signal (HR={args.heart_rate} bpm)...")
    duration = meta["duration_seconds"]
    bvp, timestamps = generate_bvp(
        heart_rate=args.heart_rate,
        duration=duration,
        fps=fps,
    )
    print(f"  BVP: {len(bvp)} samples, duration={duration}s")

    # --- Inject ---
    print(f"\n[4/4] Injecting rPPG signal (amplitude={args.amplitude})...")
    injector = RPPGInjector(
        amplitude=args.amplitude,
        channel=args.channel,
    )
    injected_frames, log = injector.inject(frames, masks, bvp)

    print(f"\n  Injection log:")
    for k, v in log.items():
        print(f"    {k}: {v}")

    # --- Save output ---
    print(f"\n  Saving injected video...")
    frames_to_video(injected_frames, args.output, fps=fps)

    # Save difference visualization
    diff_dir = os.path.join(os.path.dirname(args.output), "injection_debug")
    os.makedirs(diff_dir, exist_ok=True)

    for i in range(min(5, len(frames))):
        diff = injector.compute_frame_diff(frames, injected_frames, i)
        cv2.imwrite(os.path.join(diff_dir, f"diff_{i:05d}.png"), diff)

    print(f"  Debug diffs saved to {diff_dir}/")

    # Save ground truth BVP signal for later evaluation
    gt_path = args.output.replace(".mp4", "_gt_bvp.npy")
    np.save(gt_path, bvp)
    print(f"  Ground truth BVP saved: {gt_path}")

    print(f"\n[DONE] Injected video: {args.output}")


if __name__ == "__main__":
    main()
