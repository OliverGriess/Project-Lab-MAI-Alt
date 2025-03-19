import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.LAV_DF import create_lav_df_dataloader, get_lav_df_metadata
from localization import find_fake_windows, windows_to_periods
from models import ict
from verifacation import evaluate_new


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
        padding_size = T - real_length
        padding_size = max(padding_size, 0)
        padding_tensor = torch.zeros(padding_size, inner_emb.shape[1]).to(
            video_frames.device
        )
        inner_emb = torch.concat([inner_emb, padding_tensor], dim=0)
        outer_emb = torch.concat([outer_emb, padding_tensor], dim=0)
        inner_emb_seq.append(inner_emb)
        outer_emb_seq.append(outer_emb)
    return torch.stack(inner_emb_seq), torch.stack(
        outer_emb_seq
    )  # shape (B, T, emb_dim)


@dataclass
class Pred:
    file_id: str
    distance: list[float]
    n_fakes: int
    fake_periods: list[tuple[int, int]]
    real_length: int
    inner_embedding: list[float]
    outer_embedding: list[float]


def extract_fake_periods(
    preds: list[Pred],
) -> list[tuple[int, int]]:
    """
    Extract fake periods from the distance tensor.
    """
    windows = []
    for pred in preds:
        dist = pred.distance.detach().cpu().numpy()
        real_length = pred.real_length
        real_dist = dist[:real_length]
        windows.append(find_fake_windows(real_dist))
    return windows


def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> list[Pred]:
    batch_size = loader.batch_size
    count = 0
    preds = []

    with tqdm(total=len(loader.dataset), desc="Running inference") as pbar:
        with torch.no_grad():
            for batch in loader:
                fake_periods = batch["fake_periods"]
                videos = batch["frames"].to(device)
                real_lengths = batch["real_length"]

                inner_emb, outer_emb = process_frames(videos, model, real_lengths)

                dist = (inner_emb - outer_emb).norm(dim=2)

                for i in range(batch_size):
                    preds.append(
                        Pred(
                            file_id=batch["file_id"][i],
                            distance=dist[i].detach().cpu().tolist(),
                            n_fakes=batch["n_fakes"][i].item(),
                            fake_periods=fake_periods[i].detach().cpu().tolist(),
                            real_length=real_lengths[i].item(),
                            inner_embedding=inner_emb.detach().cpu().tolist(),
                            outer_embedding=outer_emb.detach().cpu().tolist(),
                        )
                    )
                count += batch_size

                pbar.update(batch_size)
                pbar.refresh()

    return preds


def run_eval(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> list[Pred]:
    batch_size = loader.batch_size
    count = 0
    inner_embeddings = []
    outer_embeddings = []
    is_same = []
    scores = []
    labels = []

    with tqdm(total=len(loader.dataset), desc="Running inference") as pbar:
        with torch.no_grad():
            for batch in loader:
                videos = batch["frames"].to(device)
                real_lengths = batch["real_length"][0]
                frame_labels = batch["frames_labels"][0]
                assert (
                    videos.shape[0] == 1
                ), "eval loop currently only supports one video at a time"

                # inner_emb, outer_emb = process_frames(videos, model, real_lengths)
                inner_emb, outer_emb = model(videos[0])

                dist = (inner_emb - outer_emb).norm(dim=1)  # shape (T,)
                scores.extend(dist.cpu().numpy())
                labels.extend(frame_labels.cpu().numpy())  # shape (T,)

                # inner_embeddings.extend(inner_emb.cpu().numpy())
                # outer_embeddings.extend(outer_emb.cpu().numpy())

                # temp = [
                #     True if frame_labels[i] else False for i in range(len(frame_labels))
                # ]
                # is_same.extend(temp)

                count += batch_size

                pbar.update(batch_size)
                pbar.refresh()

    # inner_embeddings = np.asarray(inner_embeddings)  # shape (N, T, emb_dim)
    # outer_embeddings = np.asarray(outer_embeddings)  # shape (N, T, emb_dim)
    # is_same = np.asarray(is_same)  # shape (N, T)

    # tpr, fpr, accuracy, best_thresholds = evaluate_new(
    #     inner_embeddings, outer_embeddings, is_same, 5
    # )
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    auc = metrics.roc_auc_score(labels, scores)
    print("AUC: ", auc)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print("Best threshold: ", best_threshold)
    fake_mask = scores > best_threshold
    scores[~fake_mask] = 1
    scores[fake_mask] = 0
    accuracy = metrics.accuracy_score(labels, scores.astype(int))
    print("Accuracy: ", accuracy)
    print("Processed: ", len(scores), "images")
    print("Real images: ", np.sum(labels))
    print("Fake images: ", len(labels) - np.sum(labels))

    # print("FPR: ", fpr)
    # print("TPR: ", tpr)
    # auc = metrics.auc(fpr, tpr)

    # print("Accuracy: ", accuracy.mean())
    # print("AUC: ", auc.mean())

    # return auc, accuracy, best_thresholds


def store_preds(preds: list[Pred], file_path: str):
    sanitized_preds = []
    for pred in preds:
        file_id = pred.file_id
        distance = pred.distance
        n_fakes = pred.n_fakes
        fake_periods = pred.fake_periods
        real_length = pred.real_length

        sanitized_preds.append(
            {
                "file_id": file_id,
                "distance": distance,
                "n_fakes": n_fakes,
                "fake_periods": fake_periods,
                "real_length": real_length,
            }
        )

    with open(file_path, "w") as f:
        json.dump(sanitized_preds, f)


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

    preds = run_inference(model, dataloader, device)
    fake_periods = extract_fake_periods(preds)

    for fake_period in fake_periods:
        print(fake_period)


if __name__ == "__main__":
    main()
