"""
generate_video.py — SadTalker Wrapper for Talking-Face Video Generation

Generates talking-face videos from source image + audio using SadTalker.
Also provides frame extraction/reassembly utilities used by downstream modules.

Usage:
    python src/generate_video.py --source_image img.png --audio audio.wav --output out.mp4
    python src/generate_video.py --test   # synthetic test video (no SadTalker needed)

Author: Swetcha
"""

import os
import sys
import yaml
import argparse
import subprocess
import glob
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# SadTalker Generator
# =============================================================================

class SadTalkerGenerator:
    """Wrapper around SadTalker inference for talking-face video generation."""

    def __init__(self, config=None):
        if config is None:
            config = load_config().get("sadtalker", {})

        self.repo_path = os.path.abspath(config.get("repo_path", "./SadTalker"))
        self.checkpoint_dir = config.get("checkpoint_dir",
                                          os.path.join(self.repo_path, "checkpoints"))
        self.enhancer = config.get("enhancer", "gfpgan")
        self.preprocess = config.get("preprocess", "crop")
        self.still_mode = config.get("still_mode", False)
        self.output_size = config.get("output_size", 256)
        self.expression_scale = config.get("expression_scale", 1.0)
        self.pose_style = config.get("pose_style", 0)
        self._validate()

    def _validate(self):
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                f"SadTalker not found at {self.repo_path}. "
                f"Run: bash scripts/setup_models.sh"
            )

    def generate(self, source_image, audio_path, output_dir="./results",
                 result_name=None):
        """
        Generate talking-face video.

        Args:
            source_image: path to portrait image
            audio_path: path to audio file (.wav)
            output_dir: output directory
            result_name: optional output filename

        Returns:
            path to generated .mp4 video
        """
        source_image = os.path.abspath(source_image)
        audio_path = os.path.abspath(audio_path)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "inference.py",
            "--driven_audio", audio_path,
            "--source_image", source_image,
            "--result_dir", output_dir,
            "--enhancer", self.enhancer,
            "--preprocess", self.preprocess,
            "--expression_scale", str(self.expression_scale),
            "--pose_style", str(self.pose_style),
        ]
        if self.still_mode:
            cmd.append("--still")
        if self.output_size == 512:
            cmd.extend(["--size", "512"])

        print(f"\n[SadTalker] Generating video...")
        print(f"  Image: {source_image}")
        print(f"  Audio: {audio_path}")

        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] SadTalker failed:\n{result.stderr}")
            raise RuntimeError("SadTalker inference failed.")

        output_video = self._find_latest_output(output_dir)
        if result_name and output_video:
            final_path = os.path.join(output_dir, result_name)
            shutil.move(output_video, final_path)
            output_video = final_path

        print(f"  [OK] Output: {output_video}")
        return output_video

    def _find_latest_output(self, output_dir):
        videos = []
        for pat in ["**/*.mp4", "*.mp4"]:
            videos.extend(glob.glob(os.path.join(output_dir, pat), recursive=True))
        return max(videos, key=os.path.getmtime) if videos else None


# =============================================================================
# Frame Utilities
# =============================================================================

def extract_frames(video_path, output_dir=None):
    """
    Extract all frames from video.

    Returns:
        frames: list of np.ndarray (H, W, 3) BGR
        fps: float
        metadata: dict with width, height, total_frames, duration_seconds
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    metadata = {
        "width": w, "height": h, "fps": fps,
        "total_frames": total,
        "duration_seconds": round(total / fps, 2) if fps > 0 else 0,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames = []
    idx = 0
    for _ in tqdm(range(total), desc="Extracting frames", unit="f"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"frame_{idx:05d}.png"), frame)
        idx += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames | {w}x{h} @ {fps}fps | {metadata['duration_seconds']}s")
    return frames, fps, metadata


def frames_to_video(frames, output_path, fps=25):
    """Reassemble frames into an mp4 video."""
    if not frames:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))
    for frame in tqdm(frames, desc="Writing video", unit="f"):
        writer.write(frame)
    writer.release()

    # Re-encode with ffmpeg for compatibility
    _reencode(output_path, fps)
    print(f"  [OK] Video saved: {output_path}")


def _reencode(path, fps):
    """Re-encode with H.264 via ffmpeg if available."""
    try:
        tmp = path.replace(".mp4", "_tmp.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "18", "-r", str(fps), "-pix_fmt", "yuv420p", tmp],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            os.replace(tmp, path)
        elif os.path.exists(tmp):
            os.remove(tmp)
    except FileNotFoundError:
        pass  # ffmpeg not installed


def create_test_video(output_path="results/test_video.mp4", num_frames=75, fps=25):
    """
    Create a synthetic test video with a face-like ellipse.
    Use this to test the full pipeline without SadTalker.
    """
    h, w = 256, 256
    frames = []
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (60, 50, 40)

        # Skin-colored face ellipse
        cv2.ellipse(frame, (w // 2, h // 2), (70, 90), 0, 0, 360,
                     (140, 175, 210), -1)
        # Forehead highlight region (important for rPPG)
        cv2.ellipse(frame, (w // 2, h // 2 - 40), (50, 25), 0, 0, 360,
                     (150, 185, 220), -1)
        # Cheek regions
        cv2.circle(frame, (w // 2 - 35, h // 2 + 10), 20, (145, 180, 215), -1)
        cv2.circle(frame, (w // 2 + 35, h // 2 + 10), 20, (145, 180, 215), -1)
        # Eyes
        cv2.circle(frame, (w // 2 - 25, h // 2 - 15), 8, (80, 60, 50), -1)
        cv2.circle(frame, (w // 2 + 25, h // 2 - 15), 8, (80, 60, 50), -1)
        # Mouth (animated)
        mouth_open = int(3 + 5 * abs(np.sin(2 * np.pi * i / num_frames * 3)))
        cv2.ellipse(frame, (w // 2, h // 2 + 30), (20, mouth_open), 0, 0, 360,
                     (100, 80, 120), -1)

        frames.append(frame)

    frames_to_video(frames, output_path, fps=fps)
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate talking-face video")
    parser.add_argument("--source_image", type=str, help="Source portrait image")
    parser.add_argument("--audio", type=str, help="Driving audio file")
    parser.add_argument("--output", type=str, default="results/generated_video.mp4")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--extract_frames", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="Create synthetic test video (no SadTalker needed)")
    args = parser.parse_args()

    if args.test:
        print("\n[TEST] Creating synthetic test video...")
        path = create_test_video()
        frames, fps, meta = extract_frames(path)
        print(f"  Metadata: {meta}")
        print(f"[DONE] {path}")
        return

    if not args.source_image or not args.audio:
        parser.error("--source_image and --audio required (or use --test)")

    config = load_config(args.config)
    gen = SadTalkerGenerator(config.get("sadtalker"))
    video_path = gen.generate(args.source_image, args.audio,
                               output_dir=os.path.dirname(args.output) or "results",
                               result_name=os.path.basename(args.output))

    if args.extract_frames and video_path:
        frames, fps, meta = extract_frames(video_path,
                                            output_dir="results/frames")
        print(f"  {meta}")

    print("[DONE]")


if __name__ == "__main__":
    main()
