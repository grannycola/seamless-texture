import argparse
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def gaussian_blur_wrap(channel: np.ndarray, sigma: float) -> np.ndarray:
    h, w = channel.shape
    tiled = np.tile(channel, (3, 3))
    img = Image.fromarray(np.clip(tiled * 255.0, 0, 255).astype(np.uint8))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    blurred = np.asarray(blurred, dtype=np.float32) / 255.0
    return blurred[h:2 * h, w:2 * w]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def auto_target_color(rgb: np.ndarray, sigma_rel: float):
    h, w, _ = rgb.shape
    sigma = max(2.0, sigma_rel * min(h, w))
    r = gaussian_blur_wrap(rgb[:, :, 0], sigma).mean()
    g = gaussian_blur_wrap(rgb[:, :, 1], sigma).mean()
    b = gaussian_blur_wrap(rgb[:, :, 2], sigma).mean()
    return float(r), float(g), float(b)


def flatten_to_single_color_with_detail(
    rgb: np.ndarray,
    sigma_rel: float = 0.5,
    detail: float = 0.15,
    clip_low_p: float = 2.0,
    clip_high_p: float = 98.0,
) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    h, w, _ = rgb.shape
    sigma = max(2.0, sigma_rel * min(h, w))
    eps = 1e-4

    illum = gaussian_blur_wrap(to_gray(rgb), sigma)
    p_lo = np.percentile(illum, clip_low_p)
    p_hi = np.percentile(illum, clip_high_p)
    illum = np.clip(illum, max(p_lo, eps), max(p_hi, eps))

    hp = rgb / np.maximum(illum[..., None], eps)
    m = np.mean(hp, axis=(0, 1), keepdims=True)
    m = np.clip(m, eps, None)
    hp = hp / m

    tr, tg, tb = auto_target_color(rgb, sigma_rel)
    target = np.stack(
        [np.full((h, w), tr, dtype=np.float32),
         np.full((h, w), tg, dtype=np.float32),
         np.full((h, w), tb, dtype=np.float32)],
        axis=-1,
    )

    detail = float(np.clip(detail, 0.0, 1.0))
    out = target * (1.0 + detail * (hp - 1.0))
    return np.clip(out, 0.0, 1.0)


def periodic_mask(
    h: int,
    w: int,
    period_px: int,
    vertical: bool,
    feather_radius: int = 2,
    seed: int = 0,
    shape: str = "square",
    duty: float = 0.5,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=np.float32)
    period_px = max(2, int(period_px))
    duty = float(np.clip(duty, 0.01, 0.99))

    if vertical:
        x = np.arange(w, dtype=np.float32)
        for y in range(h):
            phase = rng.randint(0, period_px)
            t = ((x + phase) % period_px) / float(max(1, period_px - 1))
            if shape == "square":
                row = (t < duty).astype(np.float32)
            else:
                row = 1.0 - np.abs(2.0 * t - 1.0)
            mask[y, :] = row
    else:
        y = np.arange(h, dtype=np.float32)
        for x in range(w):
            phase = rng.randint(0, period_px)
            t = ((y + phase) % period_px) / float(max(1, period_px - 1))
            if shape == "square":
                col = (t < duty).astype(np.float32)
            else:
                col = 1.0 - np.abs(2.0 * t - 1.0)
            mask[:, x] = col

    if feather_radius > 0:
        img = Image.fromarray((mask * 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
        mask = np.asarray(img, dtype=np.float32) / 255.0

    return np.clip(mask, 0.0, 1.0)


def seamless_overlap_blend(
    rgb: np.ndarray,
    overlap_px: int = 24,
    period_px: int = 10,
    feather_radius: int = 2,
) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    h, w, _ = rgb.shape
    m = int(max(1, min(overlap_px, w // 6, h // 6)))
    out = rgb.copy()

    left = out[:, :m, :]
    right = out[:, -m:, :]
    m_lr = periodic_mask(h, m, period_px, True, feather_radius, seed=42, 
                         shape="square", duty=0.5)[..., None]
    blend_lr = m_lr * right + (1.0 - m_lr) * left
    out[:, :m, :] = blend_lr
    out[:, -m:, :] = blend_lr

    top = out[:m, :, :]
    bottom = out[-m:, :, :]
    m_tb = periodic_mask(m, w, period_px, False, feather_radius, seed=42, 
                         shape="square", duty=0.5)[..., None]
    blend_tb = m_tb * bottom + (1.0 - m_tb) * top
    out[:m, :, :] = blend_tb
    out[-m:, :, :] = blend_tb

    return np.clip(out, 0.0, 1.0)


def load_rgb(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr.astype(np.float32) / 255.0


def save_rgb(path: str, rgb: np.ndarray) -> None:
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--sigma-rel", type=float, default=0.5)
    parser.add_argument("--detail", type=float, default=0.15)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--feather", type=int, default=2)
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    rgb = load_rgb(args.image)

    tile = flatten_to_single_color_with_detail(rgb, 
                                               sigma_rel=args.sigma_rel, 
                                               detail=args.detail)

    tile = seamless_overlap_blend(tile, overlap_px=args.overlap,
                                  period_px=args.period,
                                  feather_radius=args.feather)

    save_rgb(os.path.join(args.outdir, "new_texture.png"), tile)
    grid = np.tile(tile, (3, 3, 1))
    save_rgb(os.path.join(args.outdir, "grid_3x3.png"), grid)


if __name__ == "__main__":
    main()
