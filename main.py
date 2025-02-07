import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
import typer
from src.data import Endovis17BinaryDataset
from src.modules import InstrumentsUNetModel

app = typer.Typer()

# Set random seeds for reproducibility
torch.set_float32_matmul_precision('medium')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def parse_train_patients(train_patients: str):
    """Convert a comma-separated string to a list of integers."""
    return [int(x) for x in train_patients.split(',')]

@app.command()
def main(
    train_patients: str = "1",  # Default train patients (modifiable via CLI, as comma-separated string)
):
    """
    Train a model on the Endovis17 dataset.

    Args:
    - train_patients (str): Comma-separated list of patient IDs for training
    """
    # Convert the train_patients string into a list of integers
    train_patients_list = parse_train_patients(train_patients)
    
    # Data directory and test patients are set as constants
    data_dir = Path("../data")
    test_patients = [9, 10]

    train_dataset = Endovis17BinaryDataset(data_dir, train_patients_list)
    test_dataset = Endovis17BinaryDataset(data_dir, test_patients, test=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    batch_size = 4
    encoder_name = "resnet34"
    in_channels = 3
    out_classes = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    T_MAX = 2500

    model = InstrumentsUNetModel(
        encoder_name=encoder_name,
        in_channels=in_channels,
        out_classes=out_classes,
    )

    trainer = pl.Trainer(max_steps=T_MAX, log_every_n_steps=1)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

if __name__ == "__main__":
    app()
