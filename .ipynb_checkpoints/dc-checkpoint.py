# lightning_mobilenet_cifar_mlflow.py
import argparse, os, torch, pytorch_lightning as pl
from torch import nn
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small
import mlflow
import torchmetrics
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# # Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:5005")

# # Create a new MLflow Experiment
# mlflow.set_experiment("MLflow Quickstart")

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
        CIFAR10(self.hparams.data_dir, train=True, download=False)
        CIFAR10(self.hparams.data_dir, train=False, download=False)

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


# ------------- Model ---------------
class MobileNetCIFAR(pl.LightningModule):
    def __init__(self, num_classes=10, lr=3e-4, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()

        backbone = mobilenet_v3_small(weights="DEFAULT")
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = backbone

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x): return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--freeze_backbone", type=bool, default=False)
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    # MLflow logger (creates experiment on first use)
    mlf_logger = MLFlowLogger(
        experiment_name="cifar10_mobilenet_v3",
        tracking_uri=args.tracking_uri,
        log_model=True,                           # auto-logs best checkpoint
        tags={"model": "mobilenet_v3_small"}
    )

    dm = CIFAR10DataModule(batch_size=args.batch_size)
    model = MobileNetCIFAR(lr=args.lr, freeze_backbone=args.freeze_backbone)

    ckpt_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=mlf_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=25,
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    
if __name__ == "__main__":
    cli_main()
