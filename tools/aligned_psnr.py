#!/usr/bin/env python3
"""Test-view render vs GT: raw PSNR + per-view global affine-aligned PSNR.

Alignment-metric companion to the cc eval — least-squares fits a global 3×4 affine
([r,g,b,1] → rgb) per view to separate out the appearance gauge component (an offline tool
for fair comparison of appearance-neutral models; a more general global 3×4 fit than the
color_correct_affine of the eval 4 metrics (per-channel scalars)).

usage: aligned_psnr.py <render_dir> <gt_dir> <names_file>
  render_dir: cam_<idx>.png (splatx render output — name-sorted camera indices)
  gt_dir:     GT image directory (on resolution mismatch, LANCZOS-resized to the render size)
  names_file: list of "idx name" lines (camera index ↔ GT filename mapping)

Dependencies: numpy, Pillow — system python3 execution convention.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    render_dir, gt_dir, names_file = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    raws, aligneds = [], []
    for line in open(names_file):
        idx, name = line.split()
        r = np.asarray(
            Image.open(render_dir / f"cam_{idx}.png").convert("RGB"), dtype=np.float64
        ) / 255.0
        g_img = Image.open(gt_dir / name).convert("RGB")
        if g_img.size != (r.shape[1], r.shape[0]):
            g_img = g_img.resize((r.shape[1], r.shape[0]), Image.LANCZOS)
        g = np.asarray(g_img, dtype=np.float64) / 255.0
        mse = np.mean((r - g) ** 2)
        raws.append(10 * np.log10(1.0 / mse))
        # Per-view global affine fit: [r,1] @ W ≈ g (the image-global version of color_correct)
        X = np.concatenate([r.reshape(-1, 3), np.ones((r.size // 3, 1))], axis=1)
        W, *_ = np.linalg.lstsq(X, g.reshape(-1, 3), rcond=None)
        corr = np.clip(X @ W, 0.0, 1.0)
        mse_c = np.mean((corr - g.reshape(-1, 3)) ** 2)
        aligneds.append(10 * np.log10(1.0 / mse_c))
        print(
            f"view {idx} {name}: raw {raws[-1]:.3f} dB"
            f" | affine-aligned {aligneds[-1]:.3f} dB"
        )
    print(
        f"\nMEAN raw {np.mean(raws):.3f} dB"
        f" | affine-aligned {np.mean(aligneds):.3f} dB ({len(raws)} views)"
    )


if __name__ == "__main__":
    main()
