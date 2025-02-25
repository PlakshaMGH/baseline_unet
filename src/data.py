import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from pathlib import Path


class RLLPABinaryDataset(Dataset):
    def __init__(
        self,
        root_data_dir: Path,
        patient_set: list[int],
        target_size=None,
        test=False,
    ):
        # collecting the frame files
        self.frames_dir = root_data_dir / "frames" / ("test" if test else "train")
        self.frame_dirs = [
            self.frames_dir / f"instrument_dataset_{p:02d}" for p in patient_set
        ]
        # check that all dirs exist
        for dir in self.frame_dirs:
            assert dir.exists(), f"Directory {dir} does not exist"
        self.frame_file_names = [
            file.absolute()
            for frame_dir in self.frame_dirs
            for file in frame_dir.iterdir()
        ]
        # collecting the mask files
        self.masks_dir = (
            root_data_dir / "masks" / ("test" if test else "train") / "binary_masks"
        )
        self.mask_dirs = [
            self.masks_dir / f"instrument_dataset_{p:02d}" for p in patient_set
        ]
        # check that all dirs exist
        for dir in self.mask_dirs:
            assert dir.exists(), f"Directory {dir} does not exist"
        self.mask_file_names = [
            file.absolute()
            for mask_dir in self.mask_dirs
            for file in mask_dir.iterdir()
        ]

        self.to_tensor = v2.ToDtype(torch.float32, scale=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.mask_file_names)

    def __getitem__(self, idx):
        frame_path = self.frame_file_names[idx]
        mask_path = self.mask_file_names[idx]
        frame = torchvision.io.read_image(frame_path)
        mask = torchvision.io.read_image(mask_path)
        frame = self.to_tensor(frame)
        mask = mask.to(torch.float32)
        if self.target_size:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(dim=0), size=self.target_size, mode="bilinear"
            ).squeeze(dim=0)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(dim=0), size=self.target_size, mode="nearest"
            ).squeeze(dim=0)
        return frame, mask


class Endo18BinaryDataset(Dataset):
    def __init__(
        self,
        root_data_dir: Path,
        patient_set: list[int],
        target_size=None,
        test=False,
    ):
        # collecting the frame files
        self.frames_dir = root_data_dir / "frames" / ("test" if test else "train")
        self.frame_dirs = [self.frames_dir / f"seq_{p:02d}" for p in patient_set]
        # check that all dirs exist
        for dir in self.frame_dirs:
            assert dir.exists(), f"Directory {dir} does not exist"
        self.frame_file_names = [
            file.absolute()
            for frame_dir in self.frame_dirs
            for file in frame_dir.iterdir()
        ]
        # collecting the mask files
        self.masks_dir = (
            root_data_dir / "masks" / "binary_masks" / ("test" if test else "train")
        )
        self.mask_dirs = [self.masks_dir / f"seq_{p:02d}" for p in patient_set]
        # check that all dirs exist
        for dir in self.mask_dirs:
            assert dir.exists(), f"Directory {dir} does not exist"
        self.mask_file_names = [
            file.absolute()
            for mask_dir in self.mask_dirs
            for file in mask_dir.iterdir()
        ]

        self.to_tensor = v2.ToDtype(torch.float32, scale=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.mask_file_names)

    def __getitem__(self, idx):
        frame_path = self.frame_file_names[idx]
        mask_path = self.mask_file_names[idx]
        frame = torchvision.io.read_image(frame_path)
        mask = torchvision.io.read_image(mask_path)
        frame = self.to_tensor(frame)
        mask = mask.to(torch.float32)
        if self.target_size:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(dim=0), size=self.target_size, mode="bilinear"
            ).squeeze(dim=0)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(dim=0), size=self.target_size, mode="nearest"
            ).squeeze(dim=0)
        return frame, mask


class VideoReader(Dataset):
    def __init__(
        self,
        frames_dir: Path,
        masks_dir: Path,
        target_size=None,
    ):
        # collecting the frame files
        # check that dir exist
        assert frames_dir.exists(), f"Directory {dir} does not exist"
        self.frame_file_names = [file.absolute() for file in frames_dir.iterdir()]
        # collecting the mask files
        # check that dir exist
        assert masks_dir.exists(), f"Directory {dir} does not exist"
        self.mask_file_names = [file.absolute() for file in masks_dir.iterdir()]

        # sort the frames and mask names based on the number ie 'frame000.png' before the extension
        self.frame_file_names.sort(key=lambda x: int(x.stem[5:8]))
        self.mask_file_names.sort(key=lambda x: int(x.stem[5:8]))

        self.to_tensor = v2.ToDtype(torch.float32, scale=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.mask_file_names)

    def __getitem__(self, idx):
        frame_path = self.frame_file_names[idx]
        mask_path = self.mask_file_names[idx]
        frame = torchvision.io.read_image(frame_path)
        mask = torchvision.io.read_image(mask_path)
        frame = self.to_tensor(frame)
        mask = mask.to(torch.float32)
        if self.target_size:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(dim=0), size=self.target_size, mode="bilinear"
            ).squeeze(dim=0)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(dim=0), size=self.target_size, mode="nearest"
            ).squeeze(dim=0)
        return frame, mask
