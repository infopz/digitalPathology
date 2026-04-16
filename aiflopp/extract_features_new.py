# This script is used to extract the features from WSI images.
# First, it has to load the UNI2 model used as feature extractor.
# Then, process iteratively each WSI image, extract the features and save them in a single .npy file for each WSI.
# For each WSI, also save a .csv file with the coords of each patch.

from pathlib import Path

from openslide import OpenSlide
from torchvision import transforms
import numpy as np
import timm
import torch
import tqdm


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


def extract_features_from_wsi(model, transform, wsi_path: Path, patch_size: int) -> (np.ndarray, np.ndarray):
    # This function extracts the features from the WSI image using the model and transform.
    # It returns the features and the coords of each patch.

    # How this has to work:
    # 1. Load the WSI image using openslide
    # 2. Extract the patches from the WSI image using the patch size and the coordinates of the patches
    # 3. Discard the patches that are mostly background (e.g., using a threshold on the mean pixel value)
    # 4. Batch the remaining patches and pass them through the model to extract the features
    # 5. Return the features and the coords of each patch

    print(f"Processing {wsi_path.name}...")

    slide = OpenSlide(str(wsi_path))
    img_width, img_height = slide.dimensions

    print(f"Loading full image into memory ({img_width}x{img_height})...")
    img_array = np.array(slide.read_region((0, 0), 0, (img_width, img_height)).convert("RGB"))
    print("Image loaded.")

    BATCH_SIZE = 32
    COLOR_THRESHOLD = 220  # Threshold to discard mostly background patches

    features = []
    coords = []

    batch_patches = []
    batch_coords = []

    total_patches = ((img_height + patch_size - 1) // patch_size) * ((img_width + patch_size - 1) // patch_size)

    for y, x in tqdm.tqdm(
        ((y, x) for y in range(0, img_height, patch_size) for x in range(0, img_width, patch_size)),
        total=total_patches,
        desc=wsi_path.name,
    ):
        patch_array = img_array[y:y + patch_size, x:x + patch_size]

        # Check mean color to discard mostly background patches
        if patch_array.mean() < COLOR_THRESHOLD:
            continue

        features.append(1)  # Placeholder for the actual features
        coords.append((x, y))

        #batch_patches.append(Image.fromarray(patch_array)) # perche convertirle in Image?
        #batch_coords.append((x, y))

        #if len(batch_patches) == BATCH_SIZE:
        #    batch_tensor = torch.stack([transform(p) for p in batch_patches]).to(model.device)
        #
        #    with torch.no_grad():
        #        batch_feats = model(batch_tensor).cpu().numpy()
        #        features.append(batch_feats)
        #        coords.extend(batch_coords)
        #
        #    batch_patches = []
        #    batch_coords = []

    # Process any remaining patches
    if batch_patches:
        batch_tensor = torch.stack([transform(p) for p in batch_patches]).to(model.device)

        with torch.no_grad():
            batch_feats = model(batch_tensor).cpu().numpy()
            features.append(batch_feats)
            coords.extend(batch_coords)

    features = np.concatenate(features, axis=0)
    coords = np.array(coords)

    print(f"Extracted {features.shape[0]} features from {wsi_path.name}")

    return features, coords


def main(input_folder: Path, output_folder: Path, uni_weights_path: Path, patch_size: int):

    output_folder.mkdir(parents=True, exist_ok=True)

    print("Loading UNI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_uni_model(device=device, model_path=str(uni_weights_path))
    model.to(device)
    print(f"UNI model loaded on {device}")

    for img in tqdm.tqdm(list(input_folder.glob("*.svs"))):
        # Process each WSI image

        wsi_field = img.stem.split("_")

        center_id = wsi_field[0]
        patient_id = wsi_field[1]
        subregion_id = wsi_field[2]

        feature_file_name = f"{center_id}_{patient_id}_{subregion_id}.npz"
        feature_path = output_folder / feature_file_name

        if feature_path.exists():
            print(f"Skipping {patient_id}: features already exist.")
            continue

        feature_data, coords = extract_features_from_wsi(model=model, transform=transform, wsi_path=img, patch_size=patch_size)

        

        # Extract features from the WSI image using the model and transform
        # Save the features in a .npy file and the coords in a .csv file
        # (Implementation of feature extraction and saving is not shown here)



if __name__ == "__main__":

    WSI_FOLDER_PATH = Path("/home/ubuntu/giodir/digitalPathology/data/share/OSR/")
    OUTPUT_PATH = Path("/home/ubuntu/giodir/digitalPathology/data/OSR_scanRE_uni_features")

    UNI_WEIGHTS_PATH = "/home/ubuntu/giodir/misc/pytorch_model.bin"

    PATH_SIZE = 512

    main(input_folder=WSI_FOLDER_PATH, 
         output_folder=OUTPUT_PATH,
         uni_weights_path=UNI_WEIGHTS_PATH, 
         patch_size=PATH_SIZE)