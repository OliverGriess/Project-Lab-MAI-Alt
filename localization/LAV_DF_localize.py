from collections import defaultdict
import csv 
import json
import os
from hysteresis_tresholding import hysteresis_threshold_1d
import ast

def frame_level_scores_from_windows_with_padding(
    window_scores, video_length, window_size=5, stride=1
):
    """
    Convert window-level scores (one score per window) into frame-level scores
    by averaging all window predictions that cover each frame, then "pad" the
    last frames that have zero coverage with the most recent valid score.

    Args:
        window_scores (list or np.array): Window-level predictions of size (T - window_size + 1) 
                                          when using stride=1, or smaller if stride>1
        video_length (int): Number of frames in the video, T
        window_size (int): Size of each sliding window, default 5
        stride (int): Stride of the sliding window, default 1

    Returns:
        list: Frame-level scores of length T
    """
    frame_scores = [0.0] * video_length
    frame_counts = [0] * video_length

    # 1. Accumulate window scores into each covered frame
    index = 0
    num_windows = len(window_scores)
    for start in range(0, video_length - window_size + 1, stride):
        if index >= num_windows:
            break  # Just in case window_scores is shorter than expected
        score = window_scores[index]
        for frame_idx in range(start, start + window_size):
            frame_scores[frame_idx] += score
            frame_counts[frame_idx] += 1
        index += 1

    # 2. Average scores for frames that are covered
    for i in range(video_length):
        if frame_counts[i] > 0:
            frame_scores[i] /= frame_counts[i]

    # 3. Pad frames that have zero coverage with the last valid score
    #    (i.e., forward fill)
    last_valid_score = None
    for i in range(video_length):
        if frame_counts[i] == 0:
            # If there's no previous valid score, set to 0, else use last_valid_score
            if last_valid_score is not None:
                frame_scores[i] = last_valid_score
            else:
                frame_scores[i] = 0.0
        else:
            last_valid_score = frame_scores[i]

    return frame_scores



def get_window_scores(preds_path: str):
    preds = defaultdict(list)
    with open(preds_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img_path, y_real, y_pred = row
            img_name = os.path.basename(img_path)
            video_id, window_id, fake_frame_pos = img_name.split(".")[0].split("_")
            preds[video_id].append((window_id, fake_frame_pos, y_pred))

    preds = {k: sorted(v, key=lambda x: int(x[0])) for k, v in preds.items()} # sort by window_id
    final_preds = {}
    for key, value in preds.items():
        is_corrupted = False

        # remove videos where we have missing predictions for part of the video 
        correct_window_id = 0
        for idx, (window_id, fake_frame_pos, y_pred) in enumerate(value):
            if idx == 0:
                correct_window_id = int(window_id)
            elif int(window_id) != int(correct_window_id) + 1:
                print("Correct window id: ", correct_window_id)
                print("Current window id: ", window_id)
                is_corrupted = True
                break
            else:
                correct_window_id += 1

        if is_corrupted:
            continue
        else:
            final_preds[key] = value

    return final_preds


def get_video_length_dict(lav_df_metadata_path):
    """
    Get the length of the video
    """
    with open(lav_df_metadata_path, "r") as f:
        metadata_arr = json.load(f)

    video_length_dict = {}


    for video_data in metadata_arr:
        if "test/" in video_data["file"]:
            split, video_id = video_data["file"].split("/")
            video_id = video_id.split(".")[0]
            video_length_dict[video_id] = video_data["video_frames"]

    return video_length_dict


def get_gt_windows(lav_df_metadata_path):
    with open(lav_df_metadata_path, "r") as f:
        lav_df_metadata = json.load(f)
    
    gt_windows = {}
    for video_data in lav_df_metadata:
        if "test/" in video_data["file"]:
            split, video_id = video_data["file"].split("/")
            video_id = video_id.split(".")[0]
            gt_windows[video_id] = video_data["fake_periods"]

    return gt_windows



def store_frame_scores(preds_path: str, lav_df_metadata_path: str, frames_scores_path: str):
    """
    Localize the fake frames in the video using the window scores
    window_dict: dict of video_id -> list of (window_id, fake_frame_pos, y_pred)
    """
    window_dict = get_window_scores(preds_path)
    window_scores = {}
    for key, value in window_dict.items():
        window_scores[key] = [float(y_pred) for _, _, y_pred in value]

    video_length_dict = get_video_length_dict(lav_df_metadata_path)
    frame_scores = {}
    for key, value in window_scores.items():
        frame_scores[key] = frame_level_scores_from_windows_with_padding(value, video_length_dict[key])


    # save frame scores to csv
    with open(frames_scores_path, "w") as f:
        writer = csv.writer(f)
        for key, value in frame_scores.items():
            writer.writerow([key, value])



def localize_videos(frames_scores_path: str, low_thr: float, high_thr: float):
    """
    Localize the fake frames in the video using the frame scores
    """
    windows = defaultdict(list)
    with open(frames_scores_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            video_id, frame_scores = row
            frame_scores = [float(x) for x in ast.literal_eval(frame_scores)]
            windows[video_id] = hysteresis_threshold_1d(frame_scores, low_thr, high_thr).tolist()

    return windows

if __name__ == "__main__":
    preds_path = "./oli_prep_91-1_acc__test-set_preds_sorted.csv"
    lav_df_metadata_path = "/work/scratch/kurse/kurs00079/data/LAV-DF/metadata.json"
    frames_scores_path = "./frame_scores.csv"
    low_thr = 0.4
    high_thr = 0.8
    windows = localize_videos(frames_scores_path, low_thr, high_thr)
    with open(f"localized_fake_windows_per_video_{low_thr}_{high_thr}.json", "w") as f:
        json.dump(windows, f)