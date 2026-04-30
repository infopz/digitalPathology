import argparse
import re
from pathlib import Path
import sys

import numpy as np
import openslide
import pandas as pd
from matplotlib import colormaps
from PIL import Image

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))


PATCH_SIZE = 512
DISPLAY_PATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    default_wsi_dir = Path(
        "/home/ubuntu/giodir/digitalPathology/data/aiFlopp/prostate_40casi_paige"
    )
    default_attention_dir = Path("aiflopp/outputs_inference/test_model/attention_scores")
    default_output_path = Path("aiflopp/attention_heatmap.png")

    parser = argparse.ArgumentParser(
        description="Plot one attention heatmap over a resized WSI thumbnail."
    )
    parser.add_argument("--bag-id", type=str, required=True)

    parser.add_argument("--wsi-dir", type=Path, default=default_wsi_dir)
    parser.add_argument("--attention-dir", type=Path, default=default_attention_dir)
    parser.add_argument("--output-path", type=Path, default=default_output_path)
    return parser.parse_args()


def normalize_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", value) if token]


def find_wsi_path(wsi_dir: Path, bag_id: str) -> Path:
    """
    Find the matching WSI file by splitting the bag_id into "tokens" (the different name parts like RE, I, 18280, etc)
    and search for a WSI file that has the same tokens.
    Return the path of the matching WSI file.
    """

    # Split bag_id into tokens
    bag_tokens = normalize_tokens(bag_id)
    matches: list[Path] = []

    # Search for WSI files that match the bag tokens
    for path in sorted(wsi_dir.glob("*.svs")):
        # Compute WSI tokens
        stem_tokens = normalize_tokens(path.stem)
        # Check if they match
        if stem_tokens[: len(bag_tokens)] == bag_tokens:
            matches.append(path)

    if not matches:
        raise FileNotFoundError(f"No WSI file found for bag_id={bag_id} in {wsi_dir}")
    if len(matches) > 1:
        match_names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple WSI files match bag_id={bag_id}: {match_names}")
    
    return matches[0]


def load_attention_scores(attention_dir: Path, bag_id: str) -> pd.DataFrame:
    attention_path = attention_dir / f"{bag_id}.csv"
    if not attention_path.exists():
        raise FileNotFoundError(f"Missing attention CSV: {attention_path}")

    # Load and check the columns
    attention_df = pd.read_csv(attention_path)
    required_cols = {"bag_id", "x", "y", "attention_score"}
    missing = required_cols - set(attention_df.columns)
    if missing:
        raise ValueError(f"Attention CSV missing columns: {missing}")

    # Check bag_id uniqueness
    csv_bag_ids = set(attention_df["bag_id"].astype(str).unique().tolist())
    if csv_bag_ids != {bag_id}:
        raise ValueError(f"Attention CSV contains unexpected bag_ids: {sorted(csv_bag_ids)}")

    return attention_df


def build_overlay(slide: openslide.OpenSlide, attention_df: pd.DataFrame) -> Image.Image:
    width, height = slide.dimensions
    thumb_width = max(1, int(np.ceil(width * DISPLAY_PATCH_SIZE / PATCH_SIZE)))
    thumb_height = max(1, int(np.ceil(height * DISPLAY_PATCH_SIZE / PATCH_SIZE)))

    thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    thumb_array = np.asarray(thumbnail, dtype=np.float32)
    actual_thumb_width, actual_thumb_height = thumbnail.size

    scale_x = actual_thumb_width / width
    scale_y = actual_thumb_height / height

    heatmap = np.zeros((actual_thumb_height, actual_thumb_width), dtype=np.float32)

    for row in attention_df.itertuples(index=False):
        x0 = max(0, min(actual_thumb_width, int(np.floor(row.x * scale_x))))
        y0 = max(0, min(actual_thumb_height, int(np.floor(row.y * scale_y))))
        x1 = max(x0 + 1, min(actual_thumb_width, int(np.ceil((row.x + PATCH_SIZE) * scale_x))))
        y1 = max(y0 + 1, min(actual_thumb_height, int(np.ceil((row.y + PATCH_SIZE) * scale_y))))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], float(row.attention_score))

    if float(heatmap.max()) > 0.0:
        normalized_heatmap = heatmap / float(heatmap.max())
    else:
        normalized_heatmap = heatmap

    heatmap_rgb = colormaps["inferno"](normalized_heatmap)[..., :3].astype(np.float32) * 255.0
    alpha = (normalized_heatmap[..., None] * 0.65).astype(np.float32)
    overlay = thumb_array * (1.0 - alpha) + heatmap_rgb * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    attention_df = load_attention_scores(args.attention_dir, args.bag_id)
    wsi_path = find_wsi_path(args.wsi_dir, args.bag_id)

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        overlay = build_overlay(slide, attention_df)
    finally:
        slide.close()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output_path)
    print(f"Saved heatmap to {args.output_path}")


if __name__ == "__main__":
    main()
