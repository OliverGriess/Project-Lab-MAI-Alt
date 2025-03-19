import json
import math
import os
from typing import Literal

import torch
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from dataloader.video_transforms import (
    NormalizeVideo,
    ResizeVideo,
    SquareVideo,
    ToTensorVideo,
)

MAX_VIDEO_LENGTH_IN_SECONDS = 20
FRAME_RATE = 25
AVG_VIDEO_LENGTH_IN_SECONDS = 8

MAX_FRAMES = MAX_VIDEO_LENGTH_IN_SECONDS * FRAME_RATE
TARGET_LENGTH = AVG_VIDEO_LENGTH_IN_SECONDS * FRAME_RATE


def read_json(path: str, object_hook=None):
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def get_train_list(data_root: str) -> list[dict]:
    metadata_json = read_json(os.path.join(data_root, "metadata.json"))
    train_list = []
    max_fake_periods = 0
    for item in metadata_json:
        video_path = os.path.join(data_root, item["file"])
        file_id = video_path.split("/")[-1].split(".")[0]
        if not "test" in video_path:
            continue
        if not os.path.exists(video_path):
            continue
        max_fake_periods = max(max_fake_periods, len(item["fake_periods"]))
        train_list.append(
            {
                "file_id": file_id,
                "video_path": video_path,
                **item,
            }
        )
    return train_list, max_fake_periods


def get_only_audio_fakes(data):
    return [item for item in data if item["modify_audio"] and not item["modify_video"]]


def get_only_video_fakes(data):
    return [item for item in data if item["modify_video"] and not item["modify_audio"]]


def get_both_fakes(data):
    return [item for item in data if item["modify_audio"] and item["modify_video"]]


def get_real_videos(data):
    return [
        item for item in data if not item["modify_audio"] and not item["modify_video"]
    ]


import concurrent.futures
from itertools import repeat


def chunk_data(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def _process_metadata_chunk(data_root: str, data_chunk: list[dict]):
    longest_video_length = -1
    max_frame_rate = -1
    total_video_length = 0
    video_count = 0

    for item in data_chunk:
        video_path = os.path.join(data_root, item["file"])
        if not os.path.exists(video_path):
            continue
        frames, _, info = read_video(video_path)
        longest_video_length = max(longest_video_length, frames.shape[0])
        max_frame_rate = max(max_frame_rate, info["video_fps"])
        total_video_length += frames.shape[0]
        video_count += 1

    return longest_video_length, max_frame_rate, total_video_length, video_count


# results from zip file 1 of the dataset -> (497, 25.0, 211.88662108228667)
def get_lav_df_metadata(data_root: str) -> tuple[int, int, float]:
    longest_video_length = -1
    frame_rate = -1
    total_video_length = 0
    video_count = 0

    data = read_json(os.path.join(data_root, "metadata.json"))

    data_chunks = list(chunk_data(data, chunk_size=100))

    progress_bar = tqdm(total=len(data), desc="Processing videos")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(os.cpu_count() * 0.65)
    ) as executor:

        results = executor.map(_process_metadata_chunk, repeat(data_root), data_chunks)

        for longest_vid, max_fps, vid_length, count in results:

            longest_video_length = max(longest_video_length, longest_vid)
            frame_rate = max(frame_rate, max_fps)
            total_video_length += vid_length
            video_count += count

            progress_bar.update(count)

    avg_video_length = total_video_length / max(video_count, 1)
    return longest_video_length, frame_rate, avg_video_length


def read_video(path: str):
    """read video and audio from path

    Args:
        path (str): video path

    Returns:
        (tensor, tensor, dict): video in shape (T, H, W, C), audio in shape (L, K), info
    """
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.permute(1, 0)
    return video, audio, info


VIDEO_TYPES = Literal["real", "only_audio_fake", "only_video_fake", "both_fake", "all"]


class LAV_DF(Dataset):

    def __init__(
        self,
        data_root: str,
        split: Literal["train", "test"] = "train",
        target_video_length: int = TARGET_LENGTH,
        video_type: VIDEO_TYPES = "real",
    ):
        self.data_root = data_root
        self.split = split
        self.train_list, self.max_fake_periods = get_train_list(self.data_root)

        real_videos = get_real_videos(self.train_list)
        fake_videos = get_only_video_fakes(self.train_list)

        self.train_list = real_videos + fake_videos

        self.target_video_length = target_video_length
        self.video_transforms = transforms.Compose(
            [
                SquareVideo(),
                ResizeVideo((112, 112), InterpolationMode.BICUBIC),
                ToTensorVideo(),
                NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _pad_video(
        self, video: Tensor, target_video_length: int | None = None
    ) -> Tensor:
        """
        Pads a video to have target_length number of frames by repeating the last frame

        Args:
            video (torch.Tensor): Tensor has shape (num_frames, height, width, channels).
            target_video_length (int): Desired number of frames.

        Returns:
            torch.Tensor: A tensor of shape (target_length, height, width, channels).
        """
        num_frames, height, width, channels = video.shape
        target_length = target_video_length or self.target_video_length

        assert num_frames > 0, "Video has no frames"

        if num_frames >= target_length:
            # If video is longer or equal to target length, just slice
            return video[:target_length]
        else:
            # Create a tensor to hold the padded video
            padded_video = torch.zeros(
                (target_length, height, width, channels),
                dtype=video.dtype,
                device=video.device,
            )

            # Copy original video frames
            padded_video[:num_frames] = video

            # Repeat the last frame to fill the remaining slots
            padded_video[num_frames:] = video[-1]

        return padded_video

    @staticmethod
    def _pad_fake_periods(
        fake_periods: list[tuple[int, int]], max_fake_periods: int
    ) -> torch.Tensor:
        padded = fake_periods + [(0, 0)] * (max_fake_periods - len(fake_periods))
        return torch.tensor(padded, dtype=torch.long)

    def get_frame_labels(
        self, fake_periods: list[tuple[int, int]], video_length: int
    ) -> torch.Tensor:
        frames_labels = []
        for fake_period in fake_periods:
            frames_processed = len(frames_labels)
            time_start, time_end = fake_period

            fake_start = time_start * FRAME_RATE
            fake_end = time_end * FRAME_RATE

            frames_labels.extend([1] * math.ceil(fake_start - frames_processed))  # real
            frames_labels.extend([0] * math.ceil(fake_end - fake_start))  # fake

        frames_labels = torch.tensor(frames_labels)
        frames_labels = frames_labels[:video_length]
        if len(frames_labels) < video_length:
            frames_labels = torch.cat(
                [
                    frames_labels,
                    torch.ones(video_length - len(frames_labels)),
                ]
            )

        return frames_labels

    def __getitem__(self, index: int) -> dict:
        sample = self.train_list[index]
        video_path = sample["video_path"]

        outputs = {}

        video, _, _ = read_video(video_path)
        outputs["real_length"] = video.shape[0]
        # video = self._pad_video(video)
        video = rearrange(video, "t h w c -> t c h w")
        video = self.video_transforms(video)

        outputs["frames"] = video
        outputs["frames_labels"] = self.get_frame_labels(
            sample["fake_periods"], video.shape[0]
        )

        assert (
            outputs["frames"].shape[0] == outputs["frames_labels"].shape[0]
        ), "Frames and frames labels have different lengths"

        outputs["file_id"] = sample["file_id"]
        outputs["video_path"] = video_path
        outputs["n_fakes"] = sample["n_fakes"]
        # outputs["fake_periods"] = self._pad_fake_periods(
        #     sample["fake_periods"], self.max_fake_periods
        # )
        outputs["fake_periods"] = sample["fake_periods"]

        return outputs

    def __len__(self) -> int:
        return len(self.train_list)


def create_lav_df_dataloader(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    video_type: VIDEO_TYPES = "real",
    target_video_length: int = TARGET_LENGTH,
    split: Literal["train", "test"] = "train",
):
    """
    Create a DataLoader for the LAV-DF dataset
    """
    dataset = LAV_DF(
        data_root=data_root,
        split=split,
        video_type=video_type,
        target_video_length=target_video_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
