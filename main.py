import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pathlib import Path
import typer
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.data import Endovis17BinaryDataset
from src.modules import InstrumentsUNetModel

app = typer.Typer()

# Set random seeds for reproducibility
torch.set_float32_matmul_precision("medium")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def parse_train_patients(train_patients: str):
    """Convert a comma-separated string to a list of integers."""
    return [int(x) for x in train_patients.split(",")]

@app.command()
def main(
    train_patients: str = "1",  # Default train patients (modifiable via CLI, as comma-separated string)
    wandb_project: str = "endovis17_model_training",  # Default WandB project name
    wandb_run_name: str = "default_run",  # Default WandB run name
):
    """
    Train a model on the Endovis17 dataset and log results to WandB.

    Args:
    - train_patients (str): Comma-separated list of patient IDs for training
    - wandb_project (str): The project name for WandB logging
    - wandb_run_name (str): The run name for WandB
    """

    # Convert the train_patients string into a list of integers
    train_patients_list = parse_train_patients(train_patients)

    # Data directory and test patients are set as constants
    data_dir = Path("../data")
    test_patients = [9, 10]

    # Initialize datasets
    train_dataset = Endovis17BinaryDataset(data_dir, train_patients_list)
    test_dataset = Endovis17BinaryDataset(data_dir, test_patients, test=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    batch_size = 4
    epochs = 10
    encoder_name = "resnet34"
    in_channels = 3
    out_classes = 1

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Initialize WandB Logger
    wandb.init(project=wandb_project, name=wandb_run_name)
    wandb.run.notes = f"tran-set:{train_patients}"  # Add this line here
    wandb_logger = WandbLogger(log_model=True)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="test/avg_iou",
        mode="max",
        save_weights_only=True,
        save_top_k=1,
        dirpath="saves",
        filename="best-model-{epoch:02d}-{test_avg_iou:.2f}",
    )

    # Initialize the model
    model = InstrumentsUNetModel(
        encoder_name=encoder_name,
        in_channels=in_channels,
        out_classes=out_classes,
    )

    # Initialize the PyTorch Lightning Trainer with the WandB logger
    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        #logger=wandb_logger,  # Adding the WandB logger to the trainer
    )

    # Start training
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

    print("Best model path: ", checkpoint_callback.best_model_path)
    wandb.save(checkpoint_callback.best_model_path)
    wandb.finish()

if __name__ == "__main__":
    app()
