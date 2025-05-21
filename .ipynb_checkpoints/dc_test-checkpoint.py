#!/usr/bin/env python
"""Extract 1024‑D feature embeddings from a MobileNetV3‑Small checkpoint
trained with `lightning_mobilenet_cifar_mlflow_checkpoint.py`.

The script loads the best checkpoint, strips the final classification layer,
passes the CIFAR‑10 **test set** through the network, and saves the resulting
embeddings (and labels) to disk as NumPy arrays.

Usage
-----
python extract_embeddings.py --checkpoint checkpoints/epoch=19-val_acc=0.93.ckpt \
                             --batch_size 256 --outfile cifar10_test_embeddings.npz

The NPZ file will contain two arrays:
  • `embeddings`  – shape (10_000, 1024)
  • `labels`      – shape (10_000,)
"""
import argparse
import pathlib
from typing import Tuple
import pytorch_lightning as pl
from torchvision.models import mobilenet_v3_small

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

# ------------- Model ---------------
class MobileNetCIFAR(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.save_hyperparameters()

        backbone = mobilenet_v3_small(weights="DEFAULT")
        # if freeze_backbone:
        #     for p in backbone.parameters():
        #         p.requires_grad = False

        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x): return self.model(x)


# ----------------------- Model wrapper ----------------------------------
# class MobileNetEncoder(nn.Module):
#     """Wrapper that returns the 1024‑D penultimate activations."""

#     def __init__(self, mobilenet: nn.Module):
#         super().__init__()
#         self.features = mobilenet.features            # convolutional trunk
#         self.avgpool = mobilenet.avgpool              # global avg‑pool
#         # everything except the very last Linear layer (classifier[3])
#         self.pre_fc = nn.Sequential(*list(mobilenet.classifier)[:-1])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 3, 224, 224) → (N, 1024)
#         x = self.features(x)
#         x = self.avgpool(x)            # (N, C, 1, 1)
#         x = torch.flatten(x, 1)        # (N, C)
#         x = self.pre_fc(x)             # (N, 1024)
#         return x


# ----------------------- Data loading -----------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_test_loader(root: str, batch_size: int, num_workers: int) -> DataLoader:
    tfm = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_set = CIFAR10(root, train=False, download=True, transform=tfm)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# Create a list to store the features
features = []
    
# Define the hook function
def hook_fn(module, input, output):
    features.append(output.detach())

# ----------------------- Main routine -----------------------------------

def extract_embeddings(checkpoint: str, data_dir: str, batch_size: int, num_workers: int,
                       outfile: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    """Run inference on the CIFAR‑10 test set and write embeddings + labels."""

    # Dynamically import the Lightning module (must be on PYTHONPATH)
    # from lightning_mobilenet_cifar_mlflow_checkpoint import MobileNetCIFAR

    # 1. Re‑instantiate the LightningModule *architecture* and load weights
    lit_model = MobileNetCIFAR.load_from_checkpoint(checkpoint, map_location=device)
    lit_model.eval()  # sanity

    # Choose the layer to extract features from (e.g., layer4)
    target_layer = lit_model.model.classifier[0]

    # Register the hook to the target layer
    hook = target_layer.register_forward_hook(hook_fn)
    # 2. Wrap backbone to expose 1024‑D embeddings
    # /encoder = MobileNetEncoder(lit_model.model).to(device).eval()

    # 3. Data loader
    loader = get_test_loader(data_dir, batch_size, num_workers)

    # 4. Inference loop
    all_embeds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor]  = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            feat = lit_model(imgs)          # (B, 1024)
            all_embeds.append(feat.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeds).numpy()  # (10_000, 1024)
    labels     = torch.cat(all_labels).numpy()  # (10_000,)
    
    # Remove the hook
    hook.remove()
    
    # 5. Persist to disk (npz)
    np.savez(outfile, embeddings=embeddings, labels=labels)
    print(f"Saved embeddings → {outfile}  |  shape={embeddings.shape}")

  
# ----------------------- CLI -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=False, default='mlruns/737497459240027344/f9230b854f824153814a2d4a407b4157/checkpoints/epoch=16-step=5984.ckpt', help="Path to .ckpt file from Lightning", )
    parser.add_argument("--data_dir", default="./data", help="CIFAR‑10 root dir (downloads if missing)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--outfile", default="cifar10_test_embeddings.npz", help="Output .npz filename")

    args = parser.parse_args()
    extract_embeddings(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        outfile=args.outfile,
    )
