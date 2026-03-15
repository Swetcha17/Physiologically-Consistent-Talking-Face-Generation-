# rPPG Talking-Face: Physiologically Consistent Video Generation

Generating talking-face videos with embedded remote photoplethysmography (rPPG) signals for enhanced physiological realism.

## Overview

This project generates talking-face videos from source images and audio using SadTalker, then injects realistic blood volume pulse (BVP) signals as subtle skin color modulations. The injected signals can be recovered using standard rPPG extraction methods, enabling applications in deepfake research, synthetic dataset generation, and digital twin simulation.

## Pipeline

```
Source Image + Audio
        |
        v
  SadTalker (talking-face generation)
        |
        v
  BiSeNet (face parsing -> skin ROI masks)
        |
        v
  BVP Generator (synthetic pulse waveform)
        |
        v
  Signal Injection (green-channel modulation)
        |
        v
  Output Video with rPPG Signal
        |
        v
  rPPG Extraction & Evaluation
```

## Setup

```bash
git clone https://github.com/<your-username>/rppg-talking-face.git
cd rppg-talking-face
conda create -n rppg-face python=3.8
conda activate rppg-face
pip install -r requirements.txt
bash scripts/setup_models.sh
```

## Usage

```bash
# Full pipeline: generate video + inject rPPG signal
python src/run_pipeline.py \
    --source_image data/source_images/sample.png \
    --audio data/audio/sample.wav \
    --heart_rate 72 \
    --output results/output.mp4

# Test mode (no SadTalker needed): synthetic face video
python src/run_pipeline.py --test
```

## Project Structure

```
src/
  generate_video.py      # SadTalker wrapper (Swetcha)
  face_parsing.py        # BiSeNet skin ROI extraction (Swetcha)
  inject_rppg.py         # rPPG signal injection (Swetcha)
  bvp_generator.py       # Synthetic BVP waveform (Ankit)
  rppg_evaluator.py      # rPPG extraction & metrics (Ankit)
  run_pipeline.py        # End-to-end pipeline (Swetcha)
```

## Team

- **Swetcha** — Pipeline integration, face parsing, signal injection
- **Ankit** — BVP waveform generation, rPPG evaluation
- **George** — Literature review, parameter specification, documentation

## References

- SadTalker: Zhang et al., CVPR 2023 ([paper](https://arxiv.org/abs/2211.12194))
- BiSeNet face parsing ([repo](https://github.com/zllrunning/face-parsing.PyTorch))
- pyVHR: Boccignone et al., 2022 ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207/))
