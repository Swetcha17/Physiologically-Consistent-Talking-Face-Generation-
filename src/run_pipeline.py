"""
run_pipeline.py — End-to-End rPPG Talking-Face Pipeline

Chains all stages:
  1. Video generation (SadTalker or test video)
  2. Face parsing (BiSeNet skin ROI extraction)
  3. BVP signal generation (synthetic pulse waveform)
  4. Signal injection (green-channel modulation in skin regions)
  5. [Optional] rPPG evaluation (extract + compare against ground truth)

Usage:
    # Full pipeline with SadTalker
    python src/run_pipeline.py \
        --source_image data/source_images/face.png \
        --audio data/audio/speech.wav \
        --heart_rate 72 \
        --output results/final_output.mp4

    # Test mode (no external models needed)
    python src/run_pipeline.py --test

    # Use existing video (skip SadTalker)
    python src/run_pipeline.py --video results/my_video.mp4 --heart_rate 80

Author: Swetcha
"""

import os
import sys
import time
import argparse
import json

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_video import (
    SadTalkerGenerator, extract_frames, frames_to_video,
    create_test_video, load_config
)
from face_parsing import FaceParser, detect_skin_hsv
from inject_rppg import RPPGInjector
from bvp_generator import generate_bvp
from rppg_evaluator import evaluate_rppg


def run_pipeline(
    source_image=None,
    audio_path=None,
    video_path=None,
    heart_rate=72,
    amplitude=0.015,
    channel="green",
    output_path="results/pipeline_output.mp4",
    use_hsv_fallback=False,
    run_evaluation=False,
    config_path="config.yaml",
    test_mode=False,
):
    """
    Run the full rPPG talking-face pipeline.

    Provide EITHER (source_image + audio_path) for SadTalker generation,
    OR video_path to use an existing video, OR test_mode=True for synthetic test.

    Args:
        source_image: portrait image for SadTalker
        audio_path: audio file for SadTalker
        video_path: existing video to inject into (skips SadTalker)
        heart_rate: target heart rate in BPM
        amplitude: injection amplitude (fraction of pixel intensity)
        channel: 'green' or 'all_chrom'
        output_path: final output video path
        use_hsv_fallback: use HSV skin detection instead of BiSeNet
        run_evaluation: run rPPG extraction evaluation after injection
        config_path: path to config.yaml
        test_mode: use synthetic test video

    Returns:
        result: dict with paths, metadata, and optional evaluation metrics
    """
    start_time = time.time()
    result = {"stages": {}}

    output_dir = os.path.dirname(output_path) or "results"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  rPPG Talking-Face Pipeline")
    print("=" * 60)

    # =========================================================================
    # STAGE 1: Video Generation
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Video Generation")
    print(f"{'='*60}")

    if test_mode:
        print("  Mode: Synthetic test video")
        raw_video = os.path.join(output_dir, "raw_video.mp4")
        create_test_video(raw_video, num_frames=75, fps=25)
        result["stages"]["generation"] = {"mode": "test", "video": raw_video}

    elif video_path:
        print(f"  Mode: Using existing video")
        raw_video = video_path
        result["stages"]["generation"] = {"mode": "existing", "video": raw_video}

    elif source_image and audio_path:
        print(f"  Mode: SadTalker generation")
        try:
            config = load_config(config_path)
            gen = SadTalkerGenerator(config.get("sadtalker"))
            raw_video = gen.generate(source_image, audio_path,
                                      output_dir=output_dir)
            result["stages"]["generation"] = {"mode": "sadtalker", "video": raw_video}
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            print(f"  Falling back to test video...")
            raw_video = os.path.join(output_dir, "raw_video.mp4")
            create_test_video(raw_video)
            result["stages"]["generation"] = {"mode": "test_fallback", "video": raw_video}
    else:
        raise ValueError("Provide (source_image + audio), video_path, or use test_mode")

    # =========================================================================
    # STAGE 2: Frame Extraction
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Frame Extraction")
    print(f"{'='*60}")

    frames, fps, meta = extract_frames(raw_video)
    result["stages"]["extraction"] = meta

    # =========================================================================
    # STAGE 3: Face Parsing
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 3: Face Parsing (Skin ROI Masks)")
    print(f"{'='*60}")

    if use_hsv_fallback:
        print("  Method: HSV color-based skin detection")
        from tqdm import tqdm
        masks = [detect_skin_hsv(f) for f in tqdm(frames, desc="  HSV", unit="f")]
        parse_method = "hsv"
    else:
        try:
            fp = FaceParser()
            masks = fp.parse_frames(frames)
            parse_method = "bisenet"
        except Exception as e:
            print(f"  BiSeNet failed: {e}")
            print(f"  Falling back to HSV detection...")
            from tqdm import tqdm
            masks = [detect_skin_hsv(f) for f in tqdm(frames, desc="  HSV", unit="f")]
            parse_method = "hsv_fallback"

    skin_coverage = np.mean([m.mean() for m in masks])
    result["stages"]["parsing"] = {
        "method": parse_method,
        "n_masks": len(masks),
        "avg_skin_coverage": round(float(skin_coverage), 4),
    }
    print(f"  Generated {len(masks)} masks | Avg skin coverage: {skin_coverage:.1%}")

    # =========================================================================
    # STAGE 4: BVP Signal Generation
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 4: BVP Signal Generation (HR={heart_rate} bpm)")
    print(f"{'='*60}")

    duration = meta["duration_seconds"]
    bvp, timestamps = generate_bvp(
        heart_rate=heart_rate,
        duration=duration,
        fps=fps,
    )
    print(f"  BVP signal: {len(bvp)} samples over {duration}s")
    print(f"  Range: [{bvp.min():.3f}, {bvp.max():.3f}]")

    result["stages"]["bvp"] = {
        "heart_rate_bpm": heart_rate,
        "n_samples": len(bvp),
        "duration_s": duration,
    }

    # =========================================================================
    # STAGE 5: Signal Injection
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 5: rPPG Signal Injection")
    print(f"{'='*60}")

    injector = RPPGInjector(amplitude=amplitude, channel=channel)
    injected_frames, injection_log = injector.inject(frames, masks, bvp)

    print(f"\n  Injection stats:")
    print(f"    Amplitude: {amplitude}")
    print(f"    Channel: {channel}")
    print(f"    Avg pixel change: {injection_log['avg_pixel_change']:.3f}")
    print(f"    Max pixel change: {injection_log['max_pixel_change']:.3f}")

    result["stages"]["injection"] = injection_log

    # =========================================================================
    # STAGE 6: Output Video Assembly
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 6: Output Video Assembly")
    print(f"{'='*60}")

    frames_to_video(injected_frames, output_path, fps=fps)

    # Save ground truth BVP
    gt_path = output_path.replace(".mp4", "_gt_bvp.npy")
    np.save(gt_path, bvp)
    print(f"  Ground truth BVP: {gt_path}")

    result["output_video"] = output_path
    result["ground_truth_bvp"] = gt_path

    # =========================================================================
    # STAGE 7 (Optional): rPPG Evaluation
    # =========================================================================
    if run_evaluation:
        print(f"\n{'='*60}")
        print(f"  STAGE 7: rPPG Evaluation")
        print(f"{'='*60}")

        metrics, extracted = evaluate_rppg(output_path, bvp, fps=fps)
        result["stages"]["evaluation"] = metrics
        print(f"  Metrics: {metrics}")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    result["elapsed_seconds"] = round(elapsed, 2)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output video  : {output_path}")
    print(f"  Ground truth  : {gt_path}")
    print(f"  Total time    : {elapsed:.1f}s")
    print(f"  Frames        : {meta['total_frames']}")
    print(f"  Heart rate    : {heart_rate} bpm")
    print(f"  Amplitude     : {amplitude}")
    print(f"  Skin coverage : {skin_coverage:.1%}")
    print(f"{'='*60}\n")

    # Save pipeline log
    log_path = output_path.replace(".mp4", "_pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Pipeline log: {log_path}")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end rPPG talking-face pipeline"
    )

    # Input options (mutually semi-exclusive)
    parser.add_argument("--source_image", type=str,
                        help="Portrait image for SadTalker")
    parser.add_argument("--audio", type=str,
                        help="Audio file for SadTalker")
    parser.add_argument("--video", type=str,
                        help="Existing video (skip SadTalker)")
    parser.add_argument("--test", action="store_true",
                        help="Use synthetic test video")

    # rPPG parameters
    parser.add_argument("--heart_rate", type=int, default=72,
                        help="Target heart rate in BPM (default: 72)")
    parser.add_argument("--amplitude", type=float, default=0.015,
                        help="Injection amplitude (default: 0.015 = 1.5%%)")
    parser.add_argument("--channel", type=str, default="green",
                        choices=["green", "all_chrom"])

    # Output
    parser.add_argument("--output", type=str,
                        default="results/pipeline_output.mp4")

    # Options
    parser.add_argument("--use_hsv", action="store_true",
                        help="Use HSV skin detection (no BiSeNet needed)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run rPPG evaluation after injection")
    parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()

    result = run_pipeline(
        source_image=args.source_image,
        audio_path=args.audio,
        video_path=args.video,
        heart_rate=args.heart_rate,
        amplitude=args.amplitude,
        channel=args.channel,
        output_path=args.output,
        use_hsv_fallback=args.use_hsv,
        run_evaluation=args.evaluate,
        config_path=args.config,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
