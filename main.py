import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from src.data import Endovis17BinaryDataset
from src.modules import InstrumentsUNetModel

torch.set_float32_matmul_precision('medium')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

DATA_DIR = Path("../data")
# orginal_image_size = 1280 x 1024
test_patient_set = [9,10]
train_dataset = Endovis17BinaryDataset(DATA_DIR,[1])
test_dataset = Endovis17BinaryDataset(DATA_DIR,test_patient_set,test=True) 

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4,num_workers=4, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=4,num_workers=4)

EPOCHS = 10
T_MAX = EPOCHS * len(train_loader) 
OUT_CLASSES = 1

model = InstrumentsUNetModel(
    encoder_name="resnet34",
    in_channels=3,
    out_classes=OUT_CLASSES,
)

trainer = pl.Trainer(max_epochs=EPOCHS,log_every_n_steps=1)
trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=test_loader,
)