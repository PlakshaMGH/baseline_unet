import torch
from pathlib import Path
import typer
from src.modules import InstrumentsUNetModel
from src.inference import create_inference_video

app = typer.Typer()

# Set random seeds for reproducibility
torch.set_float32_matmul_precision("medium")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


@app.command()
def main():

    # Data directory and test patients are set as constants
    data_dir = Path("../data")

    encoder_name = "resnet34"
    in_channels = 3
    out_classes = 1

    model_path = "../comp_images/models/unet_e17_bin.ckpt"

    best_model = InstrumentsUNetModel(
        encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes
    )
    best_model.load_state_dict(torch.load(model_path)["state_dict"])

    # create inference video
    video_name = "instrument_dataset_10"
    video_path, iou = create_inference_video(
        model=best_model.cuda(),
        video_name=video_name,
        video_frames_dir=data_dir / "frames/test" / video_name,
        video_masks_dir=data_dir / "masks/test/binary_masks" / video_name,
        mask_only=True,
    )

    print(f"Video saved at: {video_path}")
    print(f"IOU: {iou}")


if __name__ == "__main__":
    app()
