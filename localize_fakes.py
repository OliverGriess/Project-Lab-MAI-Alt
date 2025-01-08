import time

import numpy as np
import torch

from dataloader.LAV_DF import create_lav_df_dataloader, get_lav_df_metadata
from localization import find_fake_windows, windows_to_periods
from models import ict


def process_frames(
    video_frames: torch.Tensor,
    model: torch.nn.Module,
    real_lengths: torch.Tensor,
):
    B, T, C, H, W = video_frames.shape
    inner_emb_seq = []
    outer_emb_seq = []
    for i in range(B):
        real_length = real_lengths[i]
        inner_emb, outer_emb = model(video_frames[i, :real_length])
        padding_tensor = torch.zeros(T - real_length, inner_emb.shape[1]).to(
            video_frames.device
        )

        inner_emb = torch.concat([inner_emb, padding_tensor], dim=0)
        outer_emb = torch.concat([outer_emb, padding_tensor], dim=0)
        inner_emb_seq.append(inner_emb)
        outer_emb_seq.append(outer_emb)
    return torch.stack(inner_emb_seq), torch.stack(outer_emb_seq)


def main():
    data_root = "DATASET/LAV-DF/LAV-DF/LAV-DF"
    model_path = "checkpoints/ICT_Base.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    batch_size = 1
    num_workers = 1

    dataloader = create_lav_df_dataloader(
        data_root, batch_size=batch_size, num_workers=num_workers
    )

    model = ict.combface_base_patch8_112()
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)
    model.eval()

    for batch in dataloader:
        fake_periods = batch["fake_periods"]
        contains_fake = fake_periods.any()
        if not contains_fake:
            continue
        videos = batch["frames"].to(device)
        real_lengths = batch["real_length"]

        inner_emb, outer_emb = process_frames(videos, model, real_lengths)
        print(inner_emb.shape, outer_emb.shape)

        dist = (inner_emb - outer_emb).norm(dim=2)

        print("dist.shape", dist.shape)
        print("dist", dist)

        windows = []
        for i in range(dist.shape[0]):
            dist_i = dist[i, : real_lengths[i]]
            windows.append(
                windows_to_periods(
                    find_fake_windows(dist_i.detach().cpu().numpy()), frame_rate=25
                )
            )

        print("windows", windows)
        print("fake_periods", fake_periods)

        break


if __name__ == "__main__":
    main()
