from collections import defaultdict
import pandas as pd
import numpy as np
from eval.utils import compute_and_visualize_metrics
from .TimeSerieClustering import TimeSeriesClustering


def train_model(scores_list, labels_list, save_path="my_tsc_state.joblib"):
    tsc = TimeSeriesClustering()
    tsc.hyperparameter_optimization_optuna(
        scores_list, labels_list)  # uses default n_trials
    tsc.save_all(save_path)
    return tsc


def predict(scores, labels, tsc):
    """
    Example wrapper that:
      - Calls predict_all using TSC's best params
      - For each method's predicted labels, compute metrics and visualize
      - Returns the dictionary of predictions
    """
    results = tsc.predict_all(scores)
    for method_name, pred_arr in results.items():
        if pred_arr is not None:
            compute_and_visualize_metrics(pred_arr, labels, method_name)
    return results


def load_and_process_data(file_paths, output_path='combined_data.csv'):
    # This is to read and combine LipDF
    all_dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None, names=[
                         'file_path', 'label_str', 'value'])
        all_dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}")

    # Process the combined data
    video_data = defaultdict(list)
    label_data = defaultdict(list)
    counter = 0
    for _, row in combined_df.iterrows():
        file_path = row['file_path']
        value = float(row['value'])
        counter += 1
        parts = file_path.split('/')
        video_info = parts[-2]  # e.g. "1_fake"
        filename = parts[-1]    # e.g. "077627_9_2.png"

        is_fake = 1 if "_fake" in video_info else 0

        filename_parts = filename.split('_')
        video_id = filename_parts[0]
        order = int(filename_parts[1])

        video_data[video_id].append((order, value))
        label_data[video_id].append((order, is_fake))

    sorted_videos = []
    sorted_labels = []

    for video_id in sorted(video_data.keys()):
        video_data[video_id].sort()
        label_data[video_id].sort()

        video_list = [v[1] for v in video_data[video_id]]
        label_list = [l[1] for l in label_data[video_id]]

        if len(video_list) >= 10:
            sorted_videos.append(video_list)
            sorted_labels.append(label_list)

    if sorted_videos:
        min_length = min(len(lst) for lst in sorted_videos)
        print(f"Shortest list length: {min_length}")
    else:
        print("No videos with at least 10 examples found.")

    return sorted_videos, sorted_labels, combined_df


def load_and_process_data_2(file_path, output_path=None):
    """
    This is for loading DDRCF

    :param file_path: Path to the CSV file
    :param output_path: Optional path to save combined data
    :return: Tuple of (sorted_videos, sorted_labels, dataframe)
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Save the dataframe to a CSV file if output_path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    # Process the data
    video_data = defaultdict(list)
    label_data = defaultdict(list)

    for _, row in df.iterrows():
        # Extract video ID from the path (assuming format "test/000001.mp4")
        video_path = row['Videopath']
        # Extracts "000001" from "test/000001.mp4"
        video_id = video_path.split('/')[-1].split('.')[0]

        # Use WindowID for sorting within a video
        window_id = int(row['WindowID'])
        label = int(row['label'])
        score = float(row['predictedScore'])

        video_data[video_id].append((window_id, score))
        label_data[video_id].append((window_id, label))

    sorted_videos = []
    sorted_labels = []

    for video_id in sorted(video_data.keys()):
        # Sort by window ID
        video_data[video_id].sort()
        label_data[video_id].sort()

        # Extract just the scores and labels after sorting
        video_list = [v[1] for v in video_data[video_id]]
        label_list = [l[1] for l in label_data[video_id]]

        # Only include videos with a minimum number of frames (same as original function)
        if len(video_list) >= 10:
            sorted_videos.append(video_list)
            sorted_labels.append(label_list)

    if sorted_videos:
        min_length = min(len(lst) for lst in sorted_videos)
        print(f"Shortest list length: {min_length}")
        print(f"Processed {len(sorted_videos)} videos with at least 10 frames")
    else:
        print("No videos with at least 10 examples found.")

    return sorted_videos, sorted_labels, df


FRAME_RATE = 25


def seconds_to_frames(seconds: float, frame_rate: int = FRAME_RATE):
    return int(seconds * frame_rate)


def window_to_score_array(windows: list[tuple[float, float]], video_length: int, frame_rate: int = FRAME_RATE):
    scores = [0] * video_length
    for window in windows:
        start, end = window
        for i in range(seconds_to_frames(start, frame_rate), seconds_to_frames(end, frame_rate)):
            scores[i] = 1
    return np.array(scores, dtype=np.float32)
