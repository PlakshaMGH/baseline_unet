import torch
from pathlib import Path
import typer
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.modules import InstrumentsUNetModel
from src.inference import create_inference_video
import numpy as np

app = typer.Typer()

# Set random seeds for reproducibility
torch.set_float32_matmul_precision("medium")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


@app.command()
def main(
    train_patients: str = "1",  # Default train patients (modifiable via CLI, as comma-separated string)
    wandb_project: str = "DataVar_UNet_RLL_PA_Bin",  # Default WandB project name
    wandb_run_name: str | None = None,
):
    """
    Train a model on the Endovis17 dataset and log results to WandB.

    Args:
    - train_patients (str): Comma-separated list of patient IDs for training
    - wandb_project (str): The project name for WandB logging
    - wandb_run_name (str): The run name for WandB
    """

    # Data directory and test patients are set as constants
    data_dir = Path("../rll_data")
    lll_data_dir = Path("../lll_data")
    test_patients = [4, 10, 11, 13]

    encoder_name = "resnet34"
    in_channels = 3
    out_classes = 1

    # Initialize WandB Logger
    wandb.init(project=wandb_project, name=wandb_run_name,id="1ai2rspk",resume="must",mode="online")

    model_path = "saves/best-model-epoch=03-test_avg_iou=0.15.ckpt"

    best_model = InstrumentsUNetModel(
        encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes
    )
    best_model.load_state_dict(
        torch.load(model_path)["state_dict"]
    )

    # create inference video
    test_videos = [f"p{p:02d}" for p in test_patients]
    ious = []
    for video_name in test_videos:
        video_path,iou = create_inference_video(
            model=best_model.cuda(),
            video_name=video_name,
            video_frames_dir=data_dir / "frames" / video_name,
            video_masks_dir=data_dir / "masks" / video_name,
        )

        ious.append(iou)

        wandb.log({f"rll_{video_name}": wandb.Video(video_path, format="mp4")})
        wandb.log({f"test/rll_{video_name}_mIoU": iou})
    
    wandb.log({"test/rll_avg_iou": np.mean(ious)})

    ## lll inference
    test_videos = [f"p{p:02d}" for p in [1,3,6,7,9,10]]
    ious = []
    for video_name in test_videos:
        video_path,iou = create_inference_video(
            model=best_model.cuda(),
            video_name=video_name,
            video_frames_dir=lll_data_dir / "frames" / video_name,
            video_masks_dir=lll_data_dir / "masks" / video_name,
        )

        ious.append(iou)

        wandb.log({f"lll_{video_name}": wandb.Video(video_path, format="mp4")})
        wandb.log({f"test/lll_{video_name}_mIoU": iou})
    
    wandb.log({"test/lll_avg_iou": np.mean(ious)})

    wandb.finish()


if __name__ == "__main__":
    app()
    