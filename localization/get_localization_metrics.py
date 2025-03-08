import json
import numpy as np
from metrics import AP, AR, Metadata



def get_proposals_dict(pred_score_dict: dict):
    proposals_dict = {}
    for video_id, score_array in pred_score_dict.items():
        new_id = "test/" + video_id + ".mp4"
        proposals_dict[new_id] = score_array
    return proposals_dict


def lav_df_metadata(lav_df_metadata_path: str):
    with open(lav_df_metadata_path, "r") as f:
        lav_df_metadata = json.load(f)

    metadata = []
    for video_data in lav_df_metadata:
        if "test/" in video_data["file"]:
            file = video_data["file"]
            n_fakes = video_data["n_fakes"]
            fake_periods = video_data["fake_periods"]
            duration = video_data["duration"]
            original = video_data["original"]
            modify_video = video_data["modify_video"]
            modify_audio = video_data["modify_audio"]
            split = video_data["split"]
            video_frames = video_data["video_frames"]
            audio_channels = video_data["audio_channels"]   
            audio_frames = video_data["audio_frames"]
            metadata.append(Metadata(file, n_fakes, fake_periods, duration, original, modify_video, modify_audio, split, video_frames, audio_channels, audio_frames))
    return metadata


def filter_metadata(metadata: list[Metadata], pred_windows: dict):
    filtered_metadata = []
    for meta in metadata:
        if meta.file in pred_windows.keys():
            filtered_metadata.append(meta)
    return filtered_metadata

FRAME_RATE = 25

def seconds_to_frames(seconds: float):
    return int(seconds * FRAME_RATE)

def window_to_score_array(windows: list[tuple[float, float]], video_length: int):
    scores = [0] * video_length
    for window in windows:
        start, end = window
        for i in range(seconds_to_frames(start), seconds_to_frames(end)):
            scores[i] = 1
    return np.array(scores, dtype=np.float32)

def metadata_to_score_dict(metadata: list[Metadata]):
    score_dict = {}
    for meta in metadata:
        windows = meta.fake_periods
        video_length = meta.video_frames
        scores = window_to_score_array(windows, video_length)
        score_dict[meta.file] = scores
    return score_dict

def get_metrics(lav_df_metadata_path: str, pred_windows, iou_thresholds=[0.5, 0.75, 0.9]):
    """
    Get localization metrics from predicted and ground truth windows.
    
    Args:
        pred_windows (dict): Dictionary mapping video_id to list of predicted windows.
            Each window should be [start, end] or empty list.
        gt_windows (dict): Dictionary mapping video_id to list of ground truth windows.
            Each window should be [start, end] or empty list.
        iou_thresholds (list): List of IOU thresholds to evaluate.
    """
    metadata = lav_df_metadata(lav_df_metadata_path)
    true_score_dict = metadata_to_score_dict(metadata)
    proposals_dict = get_proposals_dict(pred_windows)
    pred_scores = []
    true_scores = []
    for video_id, score_array in proposals_dict.items():
        true_score_array = true_score_dict[video_id]
        pred_scores.append(score_array)
        true_scores.append(true_score_array)


    metrics = dict()
    ap = AP(iou_thresholds=iou_thresholds)
    ar = AR(iou_thresholds=iou_thresholds)
    metrics["ap"] = ap(pred_scores, true_scores)
    # metrics["ar"] = ar(metadata, proposals_dict)
    return metrics


if __name__ == "__main__":
    print("LOADING DATA")
    with open("localized_fake_windows_per_video_0.4_0.8.json", "r") as f:
        localized_fake_windows_per_video = json.load(f)
    print("LOADING METADATA")
    lav_df_metadata_path = "/work/scratch/kurse/kurs00079/data/LAV-DF/metadata.json"
    print("LOADING METRICS")
    iou_thresholds = [0.5, 0.75, 0.9]
    metrics = get_metrics(lav_df_metadata_path, localized_fake_windows_per_video, iou_thresholds)
    metrics["ap"] = {key: value.item() for key, value in metrics["ap"].items()}
    print("METRICS", metrics)
    with open("localized_fake_windows_per_video_0.4_0.8_metrics.json", "w") as f:
        json.dump(metrics, f)
