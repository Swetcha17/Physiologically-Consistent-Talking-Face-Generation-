"""
face_parsing.py — BiSeNet Face Parsing for Skin ROI Extraction

Extracts per-frame skin region masks (forehead, cheeks, nose) from
video frames using a pretrained BiSeNet model (CelebAMask-HQ).
These masks define WHERE the rPPG signal gets injected.

Includes HSV color-based fallback that works WITHOUT torch installed.

Usage:
    python src/face_parsing.py --video results/test_video.mp4
    python src/face_parsing.py --test --use_hsv  # no model/torch needed

Author: Swetcha
"""

import os
import sys
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from generate_video import extract_frames, create_test_video, load_config


# =============================================================================
# HSV Skin Detection (no dependencies — always available)
# =============================================================================

def detect_skin_hsv(frame_bgr):
    """
    Skin detection using HSV color thresholding.
    Works without any model weights or torch.

    Args:
        frame_bgr: np.ndarray (H, W, 3) BGR uint8

    Returns:
        mask: np.ndarray (H, W) float32 in [0, 1]
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Skin tone range in HSV
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 170, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = ((mask1 | mask2) / 255.0).astype(np.float32)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    return mask


# =============================================================================
# BiSeNet Face Parser (requires torch — lazy loaded)
# =============================================================================

class FaceParser:
    """
    Extract skin ROI masks using BiSeNet.
    Requires torch + pretrained weights.

    CelebAMask-HQ labels:
        0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
        6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose,
        11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=necklace,
        16=cloth, 17=hair, 18=hat
    """

    DEFAULT_SKIN_LABELS = [1, 10, 14]

    def __init__(self, model_path=None, skin_labels=None, device=None):
        # Lazy import torch only when BiSeNet is actually used
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torchvision import transforms
        except ImportError:
            raise ImportError(
                "torch/torchvision not installed. Use --use_hsv flag for "
                "HSV-based skin detection, or install: pip install torch torchvision"
            )

        self._torch = torch
        self._transforms = transforms
        self._F = F

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.skin_labels = skin_labels or self.DEFAULT_SKIN_LABELS

        if model_path is None:
            model_path = "models/face_parsing_79999_iter.pth"

        # Build model
        self.model = self._build_bisenet(torch, nn, F)

        if os.path.isfile(model_path):
            print(f"  Loading BiSeNet weights: {model_path}")
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            print(f"  [WARNING] Weights not found: {model_path}")
            print(f"  Using random weights — masks will be meaningless.")
            print(f"  Download: gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O {model_path}")

        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225)),
        ])

    def _build_bisenet(self, torch, nn, F):
        """Build BiSeNet architecture inline."""

        class ConvBNReLU(nn.Module):
            def __init__(self, in_c, out_c, ks=3, stride=1, padding=1):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, ks, stride, padding, bias=False)
                self.bn = nn.BatchNorm2d(out_c)
            def forward(self, x):
                return F.relu(self.bn(self.conv(x)))

        class BiSeNetOutput(nn.Module):
            def __init__(self, in_c, mid_c, n_classes):
                super().__init__()
                self.conv = ConvBNReLU(in_c, mid_c, ks=3, stride=1, padding=1)
                self.conv_out = nn.Conv2d(mid_c, n_classes, 1, 1, 0)
            def forward(self, x):
                return self.conv_out(self.conv(x))

        class AttentionRefinementModule(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = ConvBNReLU(in_c, out_c, ks=3, stride=1, padding=1)
                self.conv_atten = nn.Conv2d(out_c, out_c, 1, 1, 0, bias=False)
                self.bn_atten = nn.BatchNorm2d(out_c)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                feat = self.conv(x)
                atten = torch.mean(feat, dim=(2, 3), keepdim=True)
                atten = self.sigmoid(self.bn_atten(self.conv_atten(atten)))
                return feat * atten

        class FeatureFusionModule(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.convblk = ConvBNReLU(in_c, out_c, ks=1, stride=1, padding=0)
                self.conv1 = nn.Conv2d(out_c, out_c // 4, 1, 1, 0, bias=False)
                self.conv2 = nn.Conv2d(out_c // 4, out_c, 1, 1, 0, bias=False)
                self.relu = nn.ReLU(inplace=True)
                self.sigmoid = nn.Sigmoid()
            def forward(self, fsp, fcp):
                fcat = torch.cat([fsp, fcp], dim=1)
                feat = self.convblk(fcat)
                atten = torch.mean(feat, dim=(2, 3), keepdim=True)
                atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
                return feat + feat * atten

        class ContextPath(nn.Module):
            def __init__(self):
                super().__init__()
                import torchvision
                resnet = torchvision.models.resnet18(pretrained=False)
                self.conv1 = resnet.conv1
                self.bn1 = resnet.bn1
                self.relu = resnet.relu
                self.maxpool = resnet.maxpool
                self.layer1 = resnet.layer1
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4
                self.arm16 = AttentionRefinementModule(256, 128)
                self.arm32 = AttentionRefinementModule(512, 128)
                self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
                self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
                self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                feat8 = self.layer2(self.layer1(x))
                feat16 = self.layer3(feat8)
                feat32 = self.layer4(feat16)
                avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
                avg = self.conv_avg(avg)
                avg_up = F.interpolate(avg, feat32.shape[2:], mode="nearest")
                feat32_arm = self.arm32(feat32)
                feat32_sum = feat32_arm + avg_up
                feat32_up = F.interpolate(feat32_sum, feat16.shape[2:], mode="nearest")
                feat32_up = self.conv_head32(feat32_up)
                feat16_arm = self.arm16(feat16)
                feat16_sum = feat16_arm + feat32_up
                feat16_up = F.interpolate(feat16_sum, feat8.shape[2:], mode="nearest")
                feat16_up = self.conv_head16(feat16_up)
                return feat8, feat16_up, feat32_up

        class SpatialPath(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
                self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
                self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
                self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
            def forward(self, x):
                return self.conv_out(self.conv3(self.conv2(self.conv1(x))))

        class BiSeNet(nn.Module):
            def __init__(self, n_classes=19):
                super().__init__()
                self.cp = ContextPath()
                self.sp = SpatialPath()
                self.ffm = FeatureFusionModule(256, 256)
                self.conv_out = BiSeNetOutput(256, 256, n_classes)
                self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
                self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
            def forward(self, x):
                H, W = x.shape[2:]
                feat_res8, feat_cp8, feat_cp16 = self.cp(x)
                feat_sp = self.sp(x)
                feat_fuse = self.ffm(feat_sp, feat_cp8)
                feat_out = self.conv_out(feat_fuse)
                feat_out = F.interpolate(feat_out, (H, W), mode="bilinear",
                                          align_corners=True)
                feat_out16 = self.conv_out16(feat_cp8)
                feat_out32 = self.conv_out32(feat_cp16)
                return feat_out, feat_out16, feat_out32

        return BiSeNet(n_classes=19)

    @property
    def _torch_no_grad(self):
        return self._torch.no_grad()

    def parse_frame(self, frame_bgr):
        """
        Parse a single frame and return skin ROI mask.

        Args:
            frame_bgr: np.ndarray (H, W, 3) BGR uint8
        Returns:
            mask: np.ndarray (H, W) float32 [0, 1]
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (512, 512))

        tensor = self.transform(frame_resized).unsqueeze(0).to(self.device)

        with self._torch_no_grad:
            out = self.model(tensor)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

        mask = np.zeros_like(parsing, dtype=np.float32)
        for label in self.skin_labels:
            mask[parsing == label] = 1.0

        mask = cv2.resize(mask, (orig_w, orig_h))
        return mask

    def parse_frames(self, frames, smooth_edges=True, blur_ksize=15):
        """
        Parse multiple frames.

        Returns:
            masks: list of np.ndarray (H, W) float32 [0, 1]
        """
        masks = []
        for frame in tqdm(frames, desc="Face parsing", unit="f"):
            mask = self.parse_frame(frame)
            if smooth_edges:
                mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
            masks.append(mask)
        return masks


# =============================================================================
# Visualization
# =============================================================================

def visualize_mask(frame_bgr, mask, alpha=0.5):
    """Overlay green mask on frame for debugging."""
    overlay = frame_bgr.copy().astype(np.float32)
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    mask_3ch = np.stack([mask] * 3, axis=-1)
    overlay = (overlay * (1 - alpha * mask_3ch) +
               green * (alpha * mask_3ch)).astype(np.uint8)
    return overlay


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Face parsing for skin ROI")
    parser.add_argument("--video", type=str, help="Input video")
    parser.add_argument("--model", type=str, default="models/face_parsing_79999_iter.pth")
    parser.add_argument("--output_dir", type=str, default="results/masks")
    parser.add_argument("--use_hsv", action="store_true",
                        help="HSV skin detection (no model/torch needed)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        video_path = create_test_video("results/test_video.mp4")
    elif args.video:
        video_path = args.video
    else:
        parser.error("--video or --test required")

    frames, fps, meta = extract_frames(video_path)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_hsv:
        print("\n[HSV] Color-based skin detection...")
        masks = [detect_skin_hsv(f) for f in tqdm(frames, desc="HSV", unit="f")]
    else:
        print("\n[BiSeNet] Neural face parsing...")
        fp = FaceParser(model_path=args.model)
        masks = fp.parse_frames(frames)

    for i in range(min(5, len(masks))):
        cv2.imwrite(os.path.join(args.output_dir, f"mask_{i:05d}.png"),
                     (masks[i] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, f"vis_{i:05d}.png"),
                     visualize_mask(frames[i], masks[i]))

    print(f"  {len(masks)} masks | shape: {masks[0].shape} | "
          f"coverage: {np.mean([m.mean() for m in masks]):.1%}")
    print("[DONE]")


if __name__ == "__main__":
    main()
