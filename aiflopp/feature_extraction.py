import os
import torch
from torchvision import transforms
import timm


def load_model(
        device: torch.device, 
        model_path: str = "/work/bolelli_synthetic/reggio_data/model_weights/uni2-h/pytorch_model.bin"
    ) -> tuple[torch.nn.Module, transforms.Compose]:
    # Load the model weights and define the image transformations

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


def extract_features(
        model: torch.nn.Module, 
        device: torch.device, 
        images: torch.Tensor
    ) -> torch.Tensor:
    # Extract features from images using the provided model

    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features
