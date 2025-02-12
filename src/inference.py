import ffmpeg
from pathlib import Path
import numpy as np
import torch
from .data import VideoReader
import cv2
from tqdm import tqdm


def color_map(pred_mask: np.ndarray, gt_mask: np.ndarray):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # Intersection of pred_mask and gt_mask: True Positive
    true_positive = np.bitwise_and(pred_mask, gt_mask)
    true_positive = np.transpose(true_positive, (1, 2, 0)).squeeze()
    # Only Pred not GT: False Positive
    false_positive = np.bitwise_and(pred_mask, np.bitwise_not(gt_mask))
    false_positive = np.transpose(false_positive, (1, 2, 0)).squeeze()
    # Only GT not Pred: False Negative
    false_negative = np.bitwise_and(np.bitwise_not(pred_mask), gt_mask)
    false_negative = np.transpose(false_negative, (1, 2, 0)).squeeze()

    # Colors
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # Creating Color Map Image
    h, w = pred_mask.shape[1:]
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    color_map[true_positive != 0] = green
    color_map[false_positive != 0] = red
    color_map[false_negative != 0] = blue

    return color_map


def create_video_from_frames(video_frames, video_name):
    save_folder = Path("./saved_videos")
    save_folder.mkdir(parents=True, exist_ok=True)
    video_path = save_folder / f"{video_name}.mp4"
    sample_frame = list(video_frames.values())[0]
    size1, size2, _ = sample_frame.shape

    output_options = {
        "framerate": 1,
        "pix_fmt": "yuv420p",
        "s": f"{size2}x{size1}",
        "c:v": "h264_nvenc",
        "preset": "fast",
    }

    process = (
        ffmpeg.input(
            "pipe:0",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{size2}x{size1}",
            framerate=1,
        )
        .output(str(video_path), **output_options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for _, frame in sorted(video_frames.items(), key=lambda x: x[0]):
        # out_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

    return str(video_path.absolute())


def create_inference_video(
    model: torch.nn.Module,
    video_name: str,
    video_frames_dir: Path,
    video_masks_dir: Path,
    save_dir=Path("./saved_videos"),
):
    save_dir.mkdir(parents=True, exist_ok=True)
    video_path = save_dir / f"{video_name}.mp4"

    test_dataset = VideoReader(
        video_frames_dir, video_masks_dir, target_size=(736, 896)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    model.eval()
    video_frames = {}
    with torch.inference_mode():
        for idx, (frame, mask) in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Inference on {video_name}",
        ):
            frame = frame.to("cuda")
            mask = mask
            pred_mask = model(frame)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask > 0.5
            pred_mask = pred_mask.squeeze(dim=0).cpu().numpy().astype(np.uint8)
            frame = frame.squeeze(dim=0).cpu().permute(1, 2, 0).numpy()
            mask = mask.squeeze(dim=0).cpu().numpy().astype(np.uint8)
            color_map_img = color_map(pred_mask, mask)
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            video_frames[idx] = cv2.addWeighted(frame, 1, color_map_img, 0.5, 0)

    video_path = create_video_from_frames(video_frames, video_name)
    return video_path
