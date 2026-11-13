"""
PyTorch implementation of TrackNetV4.

Reference: "TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps"
arXiv 2409.14543 (September 2024)

Architecture: U-Net backbone with a Motion Attention Map (MAM) module inserted in the
decoder. The MAM computes per-pixel motion saliency from consecutive frame pairs and
gates decoder feature maps so the network focuses on moving regions.

Input:  (B, 9, H, W)  — 3 RGB frames stacked channel-wise, resized to 512×288
Output: (B, 1, H, W)  — Gaussian heatmap; peak pixel = shuttlecock location
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

from app.services.tracking.data_types import ShuttleDetection


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims in case of odd input sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ──────────────────────────────────────────────
# Motion Attention Map (MAM) — key V4 innovation
# ──────────────────────────────────────────────

class MotionAttentionMap(nn.Module):
    """
    Computes a spatial attention weight map from consecutive frame pairs.
    Takes the 9-channel input (3 RGB frames) and produces a (B, 1, H, W) attention
    mask that is multiplied into decoder feature maps to highlight moving regions.
    """

    def __init__(self):
        super().__init__()
        # Frame difference → motion saliency
        self.motion_conv = nn.Sequential(
            ConvBNReLU(6, 32),   # two consecutive frame pairs → 6 channels
            ConvBNReLU(32, 16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, frames_9ch: torch.Tensor) -> torch.Tensor:
        # frames_9ch: (B, 9, H, W) — [frame0_RGB | frame1_RGB | frame2_RGB]
        f0 = frames_9ch[:, 0:3]
        f1 = frames_9ch[:, 3:6]
        f2 = frames_9ch[:, 6:9]
        # Frame differences capture motion
        diff01 = torch.abs(f1 - f0)
        diff12 = torch.abs(f2 - f1)
        motion = torch.cat([diff01, diff12], dim=1)  # (B, 6, H, W)
        return self.motion_conv(motion)              # (B, 1, H, W)


# ──────────────────────────────────────────────
# TrackNetV4 full model
# ──────────────────────────────────────────────

class TrackNetV4(nn.Module):
    INPUT_H = 288
    INPUT_W = 512

    def __init__(self):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(9, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Motion Attention Module
        self.mam = MotionAttentionMap()
        self.mam_upsample = nn.Identity()  # resized on-the-fly in forward

        # Decoder with skip connections
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512,  256, 256)
        self.up3 = Up(256,  128, 128)
        self.up4 = Up(128,   64,  64)

        # Output head — single-channel Gaussian heatmap
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9, H, W)
        mam = self.mam(x)  # (B, 1, H, W) — full resolution attention

        # Encoder
        s1 = self.inc(x)       # 64 ch
        s2 = self.down1(s1)    # 128 ch
        s3 = self.down2(s2)    # 256 ch
        s4 = self.down3(s3)    # 512 ch
        b  = self.down4(s4)    # 1024 ch — bottleneck

        # Decoder
        d1 = self.up1(b,  s4)  # 512 ch
        d2 = self.up2(d1, s3)  # 256 ch
        d3 = self.up3(d2, s2)  # 128 ch
        d4 = self.up4(d3, s1)  # 64 ch

        # Apply MAM: gate the final decoder features with motion attention
        mam_resized = F.interpolate(mam, size=d4.shape[2:], mode="bilinear", align_corners=False)
        d4 = d4 * (1.0 + mam_resized)  # additive residual boost on moving regions

        return self.out_conv(d4)  # (B, 1, H, W)

    # ── Inference helpers ──────────────────────────────────────────────────

    @classmethod
    def load_weights(cls, path: str) -> "TrackNetV4":
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Accept both bare state dicts and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    @torch.no_grad()
    def predict_frame_sequence(
        self,
        frames: list,  # list of np.ndarray (H, W, 3) BGR or RGB, any size
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> list[Optional[ShuttleDetection]]:
        """
        Process a list of video frames and return one detection per frame.
        The first two frames always return None (need 3 frames for temporal input).
        """
        results: list[Optional[ShuttleDetection]] = [None, None]
        self.to(device)
        self.eval()

        for i in range(2, len(frames)):
            triplet = [frames[i - 2], frames[i - 1], frames[i]]
            tensor = self._prepare_input(triplet, device)
            heatmap = self(tensor)[0, 0]  # (H, W)
            detection = self._decode_heatmap(heatmap, frame_index=i, threshold=confidence_threshold)
            results.append(detection)

        return results

    def _prepare_input(self, frames_rgb: list, device: str) -> torch.Tensor:
        import cv2
        channels = []
        for frame in frames_rgb:
            resized = cv2.resize(frame, (self.INPUT_W, self.INPUT_H))
            t = torch.from_numpy(resized).float() / 255.0  # (H, W, 3)
            channels.append(t.permute(2, 0, 1))            # (3, H, W)
        x = torch.cat(channels, dim=0).unsqueeze(0)        # (1, 9, H, W)
        return x.to(device)

    def _decode_heatmap(
        self, heatmap: torch.Tensor, frame_index: int, threshold: float
    ) -> Optional[ShuttleDetection]:
        conf = heatmap.max().item()
        if conf < threshold:
            return None
        flat_idx = heatmap.argmax().item()
        h, w = heatmap.shape
        y_px = (flat_idx // w)
        x_px = (flat_idx % w)
        # Scale back to original-video pixel coords — caller handles this mapping
        return ShuttleDetection(
            frame=frame_index,
            x_px=float(x_px) * (1.0 / w),   # stored as normalized 0-1
            y_px=float(y_px) * (1.0 / h),
            confidence=float(conf),
        )
