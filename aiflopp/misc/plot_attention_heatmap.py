import argparse
from pathlib import Path
import sys
import os

import numpy as np
import openslide
import pandas as pd
from matplotlib import colormaps
from PIL import Image

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))


PATCH_SIZE = 512
DISPLAY_PATCH_SIZE = 8

WSI_DIRS = [Path("/home/ubuntu/giodir/digitalPathology/data/aiFlopp/prostate_40casi_ibex"),
            Path("/home/ubuntu/giodir/digitalPathology/data/share/2025-09-22_TN_1_17"),
            Path("/home/ubuntu/giodir/digitalPathology/data/share/2025-09-23_TN_18_40")]

DEFAULT_ATTENTION_DIR = Path("aiflopp/outputs_inference/test_model_wnames/attention_scores")
DEFAULT_OUTPUT_PATH = Path("aiflopp/test_heatmap")

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Plot attention heatmaps over resized WSI thumbnails."
    )
    parser.add_argument("--bag-id", type=str, default=None)
    parser.add_argument(
        "--num-sampled-bag",
        type=int,
        default=None,
        help="Number of bags to randomly sample from attention-dir when --bag-id is not provided. If omitted, plot all bags.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling bags with --num-sampled-bag.",
    )

    parser.add_argument("--wsi-dirs", type=Path, nargs="+", default=WSI_DIRS, help="Directory (or directories) to search for WSI files.")
    parser.add_argument("--attention-dir", type=Path, default=DEFAULT_ATTENTION_DIR, help="Directory containing attention CSV files named <bag_id>.csv.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Directory where heatmap images will be saved.")

    parser.add_argument("--plot-outlines", action="store_true", help="Whether to plot patch outlines on the heatmap.")

    return parser.parse_args()


def resolve_bag_ids(attention_dir: Path, bag_id: str | None, num_sampled_bag: int | None, seed: int) -> list[str]:
    if bag_id is not None:
        return [bag_id]

    if num_sampled_bag is not None and num_sampled_bag <= 0:
        raise ValueError("--num-sampled-bag must be > 0 when provided.")

    attention_paths = sorted(attention_dir.glob("*.csv"))
    if not attention_paths:
        raise FileNotFoundError(f"No attention CSV files found in {attention_dir}")

    bag_ids = [path.stem for path in attention_paths]
    if num_sampled_bag is None:
        return bag_ids

    sample_size = min(num_sampled_bag, len(bag_ids))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(bag_ids), size=sample_size, replace=False)
    return sorted(bag_ids[idx] for idx in sampled_idx)


def find_wsi_path(wsi_dirs: list[Path], bag_id: str) -> Path:
    """
    Find the matching WSI by recursively searching for a file named <bag_id>.svs under wsi_dirs.
    """

    target_name = f"{bag_id}.svs"
    matches = []

    for wsi_dir in wsi_dirs:
        matches.extend(list(wsi_dir.rglob(target_name)))

    # Fallback: try matching by stem equality for any .svs file
    if not matches:
        for wsi_dir in wsi_dirs:
            for path in wsi_dir.rglob("*.svs"):
                if path.stem == bag_id:
                    matches.append(path)

    if not matches:
        raise FileNotFoundError(f"No WSI file found for identifier={bag_id} in {wsi_dirs}")
    if len(matches) > 1:
        match_names = ", ".join(str(path) for path in matches)
        raise ValueError(f"Multiple WSI files match identifier={bag_id}: {match_names}")

    return matches[0]


def load_attention_scores(attention_dir: Path, bag_id: str) -> pd.DataFrame:
    attention_path = attention_dir / f"{bag_id}.csv"
    if not attention_path.exists():
        raise FileNotFoundError(f"Missing attention CSV: {attention_path}")

    # Load and check the columns
    attention_df = pd.read_csv(attention_path)
    required_cols = {"bag_id", "wsi_name", "x", "y", "attention_score"}
    missing = required_cols - set(attention_df.columns)
    if missing:
        raise ValueError(f"Attention CSV missing columns: {missing}")

    # Check bag_id uniqueness
    csv_bag_ids = set(attention_df["bag_id"].astype(str).unique().tolist())
    if csv_bag_ids != {bag_id}:
        raise ValueError(f"Attention CSV contains unexpected bag_ids: {sorted(csv_bag_ids)}")

    return attention_df


def build_overlay(slide: openslide.OpenSlide, attention_df: pd.DataFrame, plot_outlines: bool) -> Image.Image:

    # Compute expected heatmap dimensions
    width, height = slide.dimensions
    thumb_width = max(1, int(np.ceil(width * DISPLAY_PATCH_SIZE / PATCH_SIZE)))
    thumb_height = max(1, int(np.ceil(height * DISPLAY_PATCH_SIZE / PATCH_SIZE)))

    # Load thumbnail and compute actual scaling factors
    thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    thumb_array = np.asarray(thumbnail, dtype=np.float32)
    actual_thumb_width, actual_thumb_height = thumbnail.size

    scale_x = actual_thumb_width / width
    scale_y = actual_thumb_height / height

    # Initialize the heatmap and the contour mask
    heatmap = np.zeros((actual_thumb_height, actual_thumb_width), dtype=np.float32)
    if plot_outlines:
        contour_map = np.zeros((actual_thumb_height, actual_thumb_width), dtype=np.float32)

    for row in attention_df.itertuples(index=False):
        # Map patch coords to thumbnail
        x0 = max(0, min(actual_thumb_width, int(np.floor(row.x * scale_x))))
        y0 = max(0, min(actual_thumb_height, int(np.floor(row.y * scale_y))))
        x1 = max(x0 + 1, min(actual_thumb_width, int(np.ceil((row.x + PATCH_SIZE) * scale_x))))
        y1 = max(y0 + 1, min(actual_thumb_height, int(np.ceil((row.y + PATCH_SIZE) * scale_y))))

        # Apply the color to the heatmap
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], float(row.attention_score))

        # Save patch outline
        if plot_outlines:
            contour_map[y0:y1, x0] = 1.0
            contour_map[y0:y1, x1 - 1] = 1.0
            contour_map[y0, x0:x1] = 1.0
            contour_map[y1 - 1, x0:x1] = 1.0

    # Normalize the heatmap to [0, 1] range
    if float(heatmap.max()) > 0.0:
        normalized_heatmap = heatmap / float(heatmap.max())
    else:
        normalized_heatmap = heatmap

    # Apply the heatmap over the thumbnail
    heatmap_rgb = colormaps["inferno"](normalized_heatmap)[..., :3].astype(np.float32) * 255.0
    alpha = (normalized_heatmap[..., None] * 0.65).astype(np.float32)
    overlay = thumb_array * (1.0 - alpha) + heatmap_rgb * alpha
    if plot_outlines:
        overlay[contour_map > 0] = 0.0
    
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def resolve_output_path(output_path: Path, bag_id: str, wsi_name: str, n_wsis: int) -> Path:
    
    safe_name = wsi_name.replace(os.sep, "_") # if wsi_name contains subdirs, replace the separators
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path / f"{bag_id}__{safe_name}.png"


def plot_bag_heatmaps(args: argparse.Namespace, bag_id: str) -> None:
    print("Loading attention scores...")
    attention_df = load_attention_scores(args.attention_dir, bag_id)
    unique_wsi_names = sorted(attention_df["wsi_name"].unique())

    print(f"Found attention scores for {len(attention_df)} patches across {len(unique_wsi_names)} WSIs")

    # Generate one heatmap per original WSI referenced
    for wsi_name in unique_wsi_names:

        print(f"Processing WSI {wsi_name}...")

        # Filter scores for the current WSI
        subset = attention_df[attention_df["wsi_name"] == wsi_name][["bag_id", "x", "y", "attention_score"]]

        # Load the WSI
        try:

            print("Loading WSI...")

            wsi_path = find_wsi_path(args.wsi_dirs, wsi_name)
            slide = openslide.OpenSlide(str(wsi_path))
        except Exception as e:
            print(f"Skipping WSI {wsi_name}: {e}")
            continue

        # Build the heatmap overlay
        try:
            print("Building heatmap overlay...")
            overlay = build_overlay(slide, subset, plot_outlines=args.plot_outlines)
        finally:
            slide.close()

        # Save the overlay image
        out_path = resolve_output_path(args.output_path, bag_id, wsi_name, len(unique_wsi_names))
        overlay.save(out_path)
        print(f"Saved heatmap to {out_path}")


def main() -> None:
    args = parse_args()
    bag_ids = resolve_bag_ids(
        args.attention_dir,
        args.bag_id,
        args.num_sampled_bag,
        args.seed,
    )

    print(f"Plotting heatmaps for {len(bag_ids)} bag(s).")
    for bag_id in bag_ids:
        print(f"\nProcessing bag {bag_id}...")
        plot_bag_heatmaps(args, bag_id)


if __name__ == "__main__":
    main()
