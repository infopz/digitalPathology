# This script is used to extract the features from WSI images.
# First, it has to load the UNI2 model used as feature extractor.
# Then, process iteratively each WSI image, extract the features and save them in a single .npy file for each WSI.
# For each WSI, also save a .csv file with the coords of each patch.

from pathlib import Path

from PIL import Image
from torchvision import transforms
import numpy as np
import timm
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

def parse_patch_filename(patch_path: Path) -> Tuple[str, int, int]:
    """
    Given a patch filename like "RE-I_25_16299_1_K_1_130911_prostate_HNE_100352_10240.png"
    extract the original filename and the coordinates of the patch (x, y).
    """
    filename_parts = patch_path.stem.split("_")
    if len(filename_parts) < 3:
        raise ValueError(f"Invalid patch filename format: {patch_path.name}")

    try:
        x, y = int(filename_parts[-2]), int(filename_parts[-1])
    except ValueError:
        raise ValueError(f"Invalid coordinates in patch filename: {patch_path.name}")
    
    original_filename = patch_path.stem[:-len(f"_{x}_{y}")]
    
    return original_filename, x, y


class PatchDataset(Dataset):
    def __init__(self, bag_folder_path: Path, transform):
        self.patch_paths = list(bag_folder_path.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str, Tuple[int, int]]:
        patch_path = self.patch_paths[idx]

        # Load image patch
        patch_array = np.array(Image.open(patch_path))
        patch_image = Image.fromarray(patch_array)

        # Extract the coordinates from the patch name
        wsi_name, x, y = parse_patch_filename(patch_path)
        coords = (x, y)

        return self.transform(patch_image), wsi_name, coords
    

def load_uni_model(model_path: str, device: torch.device):

    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )

    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()

    return model, transform


def extract_features_from_wsi(model, transform, device: torch.device, bag_path: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # This function extracts the features from the WSI image using the model and transform.
    # It returns the features and the coords of each patch.

    # How this has to work:
    # 1. Load the WSI image using openslide
    # 2. Extract the patches from the WSI image using the patch size and the coordinates of the patches
    # 3. Discard the patches that are mostly background (e.g., using a threshold on the mean pixel value)
    # 4. Batch the remaining patches and pass them through the model to extract the features
    # 5. Return the features and the coords of each patch

    print(f"Processing {bag_path.stem}...")

    bag_features = []
    bag_wsi_names = []
    bag_coords = []

    dataset = PatchDataset(bag_folder_path=bag_path, transform=transform)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=3,
                            pin_memory=True,
                            prefetch_factor=4,
                            collate_fn=lambda x: (torch.stack([item[0] for item in x]), [item[1] for item in x], [item[2] for item in x]))


    with torch.no_grad():
        for batch_patches, batch_wsi_names, batch_coords in tqdm.tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            batch_patches = batch_patches.to(device)

            batch_feats = model(batch_patches).cpu().numpy()
            bag_features.append(batch_feats)
            bag_wsi_names.extend(batch_wsi_names)
            bag_coords.extend(batch_coords)

    features = np.concatenate(bag_features, axis=0)

    coords = []
    for batch_coords in bag_coords:
        coords.extend(batch_coords)
    coords = np.array(coords)

    wsi_names = []
    for batch_wsi_names in bag_wsi_names:
        wsi_names.extend(batch_wsi_names)
    wsi_names = np.array(wsi_names)

    print(f"Extracted {features.shape[0]} features from {bag_path.name}")

    return features, wsi_names, coords


def main(input_folder: Path, output_folder: Path, uni_weights_path: Path, batch_size: int = 32):

    output_folder.mkdir(parents=True, exist_ok=True)

    print("Loading UNI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_uni_model(device=device, model_path=str(uni_weights_path))
    model.to(device)
    print(f"UNI model loaded on {device}")

    for bag_folder in tqdm.tqdm(list(input_folder.glob("*/"))):

        bag_name = bag_folder.stem

        feature_output_file_name = f"{bag_name}.npz"
        feature_output_path = output_folder / feature_output_file_name

        if feature_output_path.exists():
            print(f"Skipping {bag_name}: features already exist.")
            continue

        feature_data, wsi_names, coords = extract_features_from_wsi(model=model, transform=transform, device=device, bag_path=bag_folder, batch_size=batch_size)

        np.savez(feature_output_path, features=feature_data, wsi_names=wsi_names, coords=coords)
        print(f"Saved features for {bag_name} to {feature_output_path}")



if __name__ == "__main__":

    PATCH_FOLDER_PATH = Path("/home/ubuntu/giodir/digitalPathology/data/alice/bag_remains/")
    OUTPUT_PATH = Path("/home/ubuntu/giodir/digitalPathology/data/uni_features_RE_reamins/")

    UNI_WEIGHTS_PATH = "/home/ubuntu/giodir/misc/pytorch_model.bin"

    BATCH_SIZE = 64

    main(input_folder=PATCH_FOLDER_PATH, 
         output_folder=OUTPUT_PATH,
         uni_weights_path=UNI_WEIGHTS_PATH,
         batch_size=BATCH_SIZE)