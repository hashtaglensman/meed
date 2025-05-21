# lightning_mobilenet_cifar_mlflow.py
import argparse, os, torch, pytorch_lightning as pl
from torch import nn
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small
import mlflow
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:5005")
# mlflow.set_tracking_uri("file:./mlflow_runs")   # clean folder
# mlflow.set_experiment("TinyNet-CIFAR10")
# Create a new MLflow Experiment
# mlflow.set_experiment("TinyNet-CIFAR10")

# ------------- Data ----------------
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.train_t = T.Compose([
            T.Resize(224), T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
        ])
        self.test_t = T.Compose([
            T.Resize(224), T.ToTensor(),
            T.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
        ])

    def prepare_data(self):
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        full = CIFAR10(self.hparams.data_dir, train=True, transform=self.train_t)
        self.train_set, self.val_set = random_split(full, [45_000, 5_000])
        self.test_set = CIFAR10(self.hparams.data_dir, train=False, transform=self.test_t)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def train_dataloader(self): return self._loader(self.train_set, True)
    def val_dataloader(self):   return self._loader(self.val_set, False)
    def test_dataloader(self):  return self._loader(self.test_set, False)


"""Small CNN → embedding → classifier.

Parameters
----------
out_dim : int, default 256
    Size of the feature embedding.
num_classes : int, default 10
    CIFAR‑10 has 10 classes; set 100 for CIFAR‑100.
"""
class TinyNetCIFAR(pl.LightningModule):
    def __init__(self, lr=3e-4,num_classes: int = 10,
                 proj_dim: int = 256,
                 encoder_channels: int = 128):
        super().__init__()
        self.save_hyperparameters()
        
        # --- simple 3-block encoder (example) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 32×32 → 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 16×16 → 8×8
            nn.Conv2d(64, encoder_channels, 3, padding=1), nn.BatchNorm2d(encoder_channels), nn.ReLU(inplace=True),
        )

        # --- NEW: make feature map size invariant ---
        self.gap = nn.AdaptiveAvgPool2d(1)      # B × C × 1 × 1
        self.projector = nn.Linear(encoder_channels, proj_dim)
        self.classifier = nn.Linear(proj_dim, num_classes)

        self._init_params()
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        h = self.encoder(x)          # B × C × H × W
        h = self.gap(h).flatten(1)   # B × C
        feats = self.projector(h)    # B × proj_dim
        logits = self.classifier(nn.Tanh()(feats))
        return logits, feats
    # ---------------------------------------------------------
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            # if m.bias is not None:
            #     nn.init.zeros_(m.bias)

    # def forward(self, x): return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits, _ = self(x)
        loss   = self.criterion(logits, y)
        preds  = logits.argmax(1)
        getattr(self, f"{stage}_acc")(preds, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", getattr(self, f"{stage}_acc"), prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, b, i):  return self._shared_step(b, "train")
    def validation_step(self, b, i): self._shared_step(b, "val")
    def test_step(self, b, i):       self._shared_step(b, "test")

    def configure_optimizers(self):
        opt = torch.optim.RMSprop(
            self.parameters(),
            lr=self.hparams.lr,
            alpha=0.99, eps=1e-06, 
            weight_decay=5e-4, momentum=0.9, 
            centered = False
        ) 
        # torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [sch]


checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=1,
    monitor="val_loss",
)


# ------------- Script --------------
def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--freeze_backbone", type=bool, default=False)
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns/tinynet")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    # MLflow logger (creates experiment on first use)
    mlf_logger = MLFlowLogger(
        experiment_name="cifar10_TINYNET",
        tracking_uri=args.tracking_uri,
        log_model=True,                           # auto-logs best checkpoint
        tags={"model": "tinynet"}
    )

    dm = CIFAR10DataModule(batch_size=args.batch_size)
    model = TinyNetCIFAR(lr=args.lr)

    ckpt_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices = 2,
        strategy = 'ddp',
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=mlf_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=25
    )

    trainer.fit(model, dm, ckpt_path = None)
    trainer.test(model, datamodule=dm)
    
if __name__ == "__main__":
    cli_main()
