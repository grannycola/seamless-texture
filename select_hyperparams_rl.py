import argparse
import os
import json
import random
import shutil
import subprocess
import numpy as np
import cv2
from PIL import Image

BASE_SCRIPT = "make_seamless.py"

PARAM_BOUNDS = {
    "sigma_rel": (0.2, 0.8),
    "detail":    (0.00, 0.60),
    "overlap":   (10, 40),
    "period":    (5, 60),
    "feather":   (1, 8),
}

def random_params() -> dict:
    out = {}
    for k, (lo, hi) in PARAM_BOUNDS.items():
        if isinstance(lo, float) or isinstance(hi, float):
            out[k] = round(random.uniform(float(lo), float(hi)), 3)
        else:
            out[k] = int(random.randint(int(lo), int(hi)))
    return out

def mutate_params(center: dict, strength: float = 0.25) -> dict:
    newp = dict(center)
    for k, (lo, hi) in PARAM_BOUNDS.items():
        if random.random() < 0.6:
            span = float(hi) - float(lo)
            delta = random.uniform(-span * strength, span * strength)
            val = float(newp[k]) + delta
            if isinstance(lo, int) and isinstance(hi, int):
                val = int(round(val))
            val = max(lo, min(hi, val))
            newp[k] = round(val, 3) if not isinstance(val, int) else val
    return newp

def run_variant(image_path: str, tmpdir: str, p: dict, idx: int) -> str:
    cmd = [
        "python3", BASE_SCRIPT,
        "--image", image_path,
        "--outdir", tmpdir,
        "--sigma-rel", str(p["sigma_rel"]),
        "--detail", str(p["detail"]),
        "--overlap", str(p["overlap"]),
        "--period", str(p["period"]),
        "--feather", str(p["feather"]),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    produced = os.path.join(tmpdir, "new_texture.png")
    if not os.path.exists(produced):
        return None

    variant_tile = os.path.join(tmpdir, f"new_texture_{idx}.png")
    shutil.move(produced, variant_tile)

    arr = np.array(Image.open(variant_tile).convert("RGB"))
    tile2x2 = np.tile(arr, (2, 2, 1))
    preview_path = os.path.join(tmpdir, f"preview2x2_{idx}.png")
    Image.fromarray(tile2x2).save(preview_path)

    grid_path = os.path.join(tmpdir, "grid_3x3.png")
    if os.path.exists(grid_path):
        os.remove(grid_path)

    return preview_path

def load_bgr_resized(path: str, size: int) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size, size))

def show_grid(previews: list, grid_size: int = 3, cell_size: int = 256) -> int | None:
    h, w = cell_size, cell_size
    canvas = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)
    k = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if k >= len(previews):
                break
            img = previews[k]
            y0, y1 = r * h, (r + 1) * h
            x0, x1 = c * w, (c + 1) * w
            canvas[y0:y1, x0:x1] = img
            cv2.putText(canvas, str(k), (x0 + 10, y0 + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            k += 1
    cv2.imshow("Select best (0-8), ESC to stop", canvas)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
        return None
    if 48 <= key <= 57:
        return key - 48
    return None

def clean_tmp(tmpdir: str):
    for f in os.listdir(tmpdir):
        if f.endswith(".png"):
            os.remove(os.path.join(tmpdir, f))

def save_best(outdir: str, best: dict):
    path = os.path.join(outdir, "best_params.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")

def main():
    parser = argparse.ArgumentParser(description="Human-in-the-loop RL tuner for make_seamless.py")
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="rl")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--cell-size", type=int, default=256)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tmpdir = os.path.join(args.outdir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)

    pop_size = args.grid_size * args.grid_size
    population = [random_params() for _ in range(pop_size)]
    best = None

    for gen in range(1, args.generations + 1):
        print(f"\n=== Generation {gen}/{args.generations} ===")
        clean_tmp(tmpdir)

        previews = []
        for i, p in enumerate(population):
            print(f"Var {i}: {p}")
            prev_path = run_variant(args.image, tmpdir, p, idx=i)
            if prev_path and os.path.exists(prev_path):
                previews.append(load_bgr_resized(prev_path, args.cell_size))
            else:
                previews.append(np.zeros((args.cell_size, args.cell_size, 3), np.uint8))

        choice = show_grid(previews, args.grid_size, args.cell_size)
        if choice is None:
            print("Stopped by user.")
            break

        winner = population[choice]
        best = winner
        print(f"→ Winner: {winner}")

        population = [mutate_params(winner, strength=0.30) for _ in range(pop_size)]
        population[0] = winner

    print("\n=== Best params ===")
    print(best)
    if args.save_best and best is not None:
        save_best(args.outdir, best)

if __name__ == "__main__":
    main()