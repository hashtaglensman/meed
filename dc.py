# lightning_mobilenet_cifar_mlflow.py
import argparse, os, torch, lightning as L
from torch import nn
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small
# import mlflow
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger, MLFlowLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from torch.utils.data import default_collate
import mlflow.pytorch
from mlflow import MlflowClient

# pl.seed_everything(7)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision('high')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"#, 4, 5, 6, 7"
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"

# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:5005")
# mlflow.set_tracking_uri("file:./mlruns")


# # Create a new MLflow Experiment
# # mlflow.set_experiment("CIFAR-Mobilenet-v3")

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


# ------------- Data ----------------
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.train_t = T.Compose([
            T.Resize(224), T.RandomHorizontalFlip(),
            T.RandAugment(num_ops = 5),
            T.ToTensor(),
            T.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
            T.RandomErasing(),
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
        self.train_set, self.val_set = random_split(full, [45000, 5000])
        self.test_set = CIFAR10(self.hparams.data_dir, train=False, transform=self.test_t)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def train_dataloader(self): return self._loader(self.train_set, True)
    def val_dataloader(self):   return self._loader(self.val_set, False)
    def test_dataloader(self):  return self._loader(self.test_set, False)


# ------------- Model ---------------
class MobileNetCIFAR(L.LightningModule):
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


# checkpoint_callback = ModelCheckpoint(
#     dirpath="checkpoints",
#     filename="{epoch}-{val_loss:.2f}",
#     save_top_k=1,
#     monitor="val_loss",
# )


# ------------- Script --------------
def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--freeze_backbone", type=bool, default=False)
    parser.add_argument("--tracking_uri", type=str, default="mlruns/base_model_mobilenet")
    args = parser.parse_args()

    L.seed_everything(42, workers=True)
    
    mlflow.set_tracking_uri(args.tracking_uri)
    # mlflow.set_experiment("cifar10_mobilenet_v3")
    
    # MLflow logger (creates experiment on first use)
    mlf_logger = MLFlowLogger(
        experiment_name="cifar10_mobilenet_v3",
        tracking_uri=args.tracking_uri,
        log_model=True,                           # auto-logs best checkpoint
        tags={"model": "mobilenet_v3_small"}
    )

    dm = CIFAR10DataModule(batch_size=args.batch_size)
    model = MobileNetCIFAR(lr=args.lr, freeze_backbone=args.freeze_backbone)

    ckpt_cb = ModelCheckpoint(monitor="val_acc", filename="{epoch}-{val_loss:.2f}", mode="max", save_top_k=3)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=4,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=mlf_logger,
        callbacks=[ckpt_cb, 
                   StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.1, 
                                             annealing_epochs=10, annealing_strategy='cos', 
                                             avg_fn=None, device='cuda')],
        log_every_n_steps=25,
    )
    
    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    trainer.test(model, datamodule=dm)
    
if __name__ == "__main__":
    cli_main()
