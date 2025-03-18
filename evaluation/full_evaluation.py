import numpy as np
from sklearn.cluster import DBSCAN
import joblib
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from flasc import FLASC
import hdbscan
from hysteresis_tresholding import hysteresis_threshold_1d
from scipy.ndimage import gaussian_filter1d
from DDM import DriftDetectionMethod


class TimeSeriesClustering:
    def __init__(self):
        # For storing best hyperparams
        self.dbscan_params = None
        self.optics_params = None
        self.hdbscan_params = None
        self.flasc_params  = None

        # A dictionary if we want to keep them all in one place
        self.best_params = {}

    def save_all(self, savepath="time_series_clustering_state.joblib"):
        joblib.dump(self, savepath)
        print(f"[INFO] Saved TimeSeriesClustering state to {savepath}.")

    @staticmethod
    def load_all(loadpath):
        loaded_obj = joblib.load(loadpath)
        if not isinstance(loaded_obj, TimeSeriesClustering):
            raise TypeError("Loaded object is not a TimeSeriesClustering instance.")
        print(f"[INFO] Loaded TimeSeriesClustering state from {loadpath}.")
        return loaded_obj
    

    # ---------------------
    # Density-based methods
    # ---------------------
    def flasc_cluster(self, scores, min_cluster_size=5, threshold=0.4):
        X = np.column_stack((np.arange(len(scores)), scores))
        clusterer = FLASC(min_cluster_size=min_cluster_size)
        raw_labels = clusterer.fit_predict(X)
        final_labels = self._postprocess_labels_threshold(raw_labels, scores, threshold=threshold)
        return final_labels, clusterer

    def dbscan_cluster(self, scores, eps=1.0, min_samples=5, threshold=0.4):
        """
        Original DBSCAN approach on the raw (index, score) pairs.
        """
        X = np.column_stack((np.arange(len(scores)), scores))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        raw_labels = model.fit_predict(X)
        final_labels = self._postprocess_labels_threshold(raw_labels, scores, threshold=threshold)
        return final_labels, model

    


    def hdbscan_cluster(self, scores, min_cluster_size=5, min_samples=None, threshold=0.4):
        import hdbscan
        X = np.column_stack((np.arange(len(scores)), scores))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        raw_labels = clusterer.fit_predict(X)
        final_labels = self._postprocess_labels_threshold(raw_labels, scores, threshold)
        return final_labels, clusterer

    # ---------------------
    # Sliding Window Baseline
    # ---------------------
    def sliding_window_baseline(self, scores, window_size=5, threshold=0.5):
        """
        A simple sliding-window baseline that assigns a label (0 or 1) to each
        frame by averaging the scores within a local window. No '0.5' label is used here.
        """
        scores = np.array(scores)
        n = len(scores)
        half_w = window_size // 2

        final_labels = np.zeros(n, dtype=float)
        for i in range(n):
            start_idx = max(0, i - half_w)
            end_idx = min(n, i + half_w + 1)
            window_scores = scores[start_idx:end_idx]
            avg_score = np.mean(window_scores)
            if avg_score > threshold:
                final_labels[i] = 1.0
            else:
                final_labels[i] = 0.0
        return final_labels

    # ---------------------
    # Predict all 
    # ---------------------
    def predict_all(self, scores,
                    dbscan_params=None,
                    optics_params=None,
                    hdbscan_params=None):
        """
        Returns predictions from multiple clustering methods plus naive + sliding-window.
        You can also add the new 'dbscan_with_transform' here if you'd like to compare.
        """
        if dbscan_params is None:
            dbscan_params = self.dbscan_params if self.dbscan_params else {}
        if hdbscan_params is None:
            hdbscan_params = self.hdbscan_params if self.hdbscan_params else {}

        dbscan_labels, _ = self.dbscan_cluster(scores, **dbscan_params)
        hdbscan_labels, _ = self.hdbscan_cluster(scores, **hdbscan_params)
        flasc_labels, _ = self.flasc_cluster(scores, **(self.flasc_params or {}))
        
        hyst_labels_02 = hysteresis_threshold_1d(scores=scores,low_thr=0.2,high_thr=0.6)
        hyst_labels_04 = hysteresis_threshold_1d(scores=scores,low_thr=0.4,high_thr=0.8)
        hyst_labels_06 = hysteresis_threshold_1d(scores=scores,low_thr=0.6,high_thr=0.8)
        gaus_score = gaussian_filter1d(scores, sigma=1.6)
        
        gaus_dbscan_labels, _ = self.dbscan_cluster(gaus_score, **dbscan_params)
        gaus_hdbscan_labels, _ = self.hdbscan_cluster(gaus_score, **hdbscan_params)
        gaus_flasc_labels, _ = self.flasc_cluster(gaus_score, **(self.flasc_params or {}))
        hyst_labels_gaus_02 = hysteresis_threshold_1d(scores=gaus_score,low_thr=0.2,high_thr=0.6)
        hyst_labels_gaus_04 = hysteresis_threshold_1d(scores=gaus_score,low_thr=0.4,high_thr=0.8)
        hyst_labels_gaus_06 = hysteresis_threshold_1d(scores=gaus_score,low_thr=0.6,high_thr=0.8)
        ddm = []
        for subscore in scores:
            ddm.append(DriftDetectionMethod(0.9, 3).cluster(subscore))
        # Example usage of the new transform-based DBSCAN

        # Baselines
        sliding_window_labels15 = self.sliding_window_baseline(scores, window_size=15, threshold=0.5)
        binary_scores = [1 if score >= 0.5 else 0 for score in scores]

        return {
            'dbscan': dbscan_labels,
            'hdbscan': hdbscan_labels,
            'flasc': flasc_labels,
            'gaus_dbscan': gaus_dbscan_labels,
            'gaus_hdbscan': gaus_hdbscan_labels,
            'gaus_flasc': gaus_flasc_labels,
            'gaus_naive': gaus_score,
            'naive': binary_scores,
            'sliding_window15': sliding_window_labels15,
            'hyst02':hyst_labels_02,
            'hyst04':hyst_labels_04,
            'hyst06':hyst_labels_06,
            'hyst_gaus02':hyst_labels_gaus_02,
            'hyst_gaus04':hyst_labels_gaus_04,
            'hyst_gaus06':hyst_labels_gaus_06,
            "ddm":ddm
        }


    def _postprocess_labels_threshold(self, raw_labels, scores, threshold=0.5):
        """
        For each cluster (label >= 0), compute the average score of the cluster.
        If the average is above 'threshold', assign label 1; otherwise 0.
        Outliers (label == -1) are assigned 0.5.
        """
        final_labels = np.full_like(raw_labels, 0.5, dtype=float)  # default to 0.5 for outliers
        scores = np.array(scores)
        cluster_ids = set(raw_labels) - {-1}
        for c_id in cluster_ids:
            c_indices = np.where(raw_labels == c_id)[0]
            c_mean_score = np.mean(scores[c_indices])
            if c_mean_score > threshold:
                final_labels[c_indices] = 1.0
            else:
                final_labels[c_indices] = 0.0
        return final_labels

    def _postprocess_labels(self, raw_labels, scores):
        """
        Example of a different approach to post-processing cluster labels.
        (Not used by default.)
        """
        final_labels = np.zeros(len(raw_labels), dtype=float)
        last_assigned = None
        for i, lbl in enumerate(raw_labels):
            if i == 0 or lbl != raw_labels[i-1]:
                if last_assigned is None:
                    final_labels[i] = 0
                    last_assigned = 0
                else:
                    if last_assigned == 0:
                        final_labels[i] = 1
                        last_assigned = 1
                    else:
                        final_labels[i] = 0
                        last_assigned = 0
            else:
                final_labels[i] = final_labels[i-1]

        zero_indices = np.where(final_labels == 0)[0]
        one_indices = np.where(final_labels == 1)[0]
        if len(zero_indices) > 0 and len(one_indices) > 0:
            avg0 = np.mean(np.array(scores)[zero_indices])
            avg1 = np.mean(np.array(scores)[one_indices])
            if avg0 >= avg1:
                final_labels[zero_indices] = 1
                final_labels[one_indices] = 0
        return final_labels

    # ---------------------
    # Domain accuracy
    # ---------------------
    def _domain_accuracy(self, pred_labels, true_labels):
        pred = np.array(pred_labels)
        true = np.array(true_labels)
        scores = np.zeros_like(true, dtype=float)

        # +1 for exact match
        exact_matches = (pred == true)
        scores[exact_matches] = 1.0

        # +0.5 for unknown
        unknown_mask = (pred == 0.5)
        scores[unknown_mask] = 0.5

        return np.mean(scores)

# --------------------------------------------------------
# Metrics + Visualization
# --------------------------------------------------------
def compute_and_visualize_metrics(pred_labels, true_labels, method_name="NoName"):
    """
    Compute metrics (AUC, domain accuracy) for a single method,
    and create a simple timeseries plot comparing predictions vs. ground truth.
    """
    from sklearn.metrics import roc_auc_score

    pred_prob = np.where(pred_labels == 1, 1.0,
                         np.where(pred_labels == 0.5, 0.5, 0.0))

    try:
        auc_val = roc_auc_score(true_labels, pred_prob)
    except ValueError:
        # e.g. if all true labels are 0 or all 1, roc_auc_score can fail
        auc_val = None

    domain_acc = domain_accuracy_method(pred_labels, true_labels)

    print(f"\nMetrics for {method_name}:")
    if auc_val is not None:
        print(f"  AUC: {auc_val:.3f}")
    else:
        print("  AUC: not computable (all true labels the same)")

    print(f"  Domain Accuracy: {domain_acc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(true_labels, label='Ground Truth (0=real,1=fake)', marker='o')
    ax.plot(pred_labels, label='Predicted (0=real,1=fake,0.5=unknown)', marker='x')
    ax.set_title(f"{method_name} Prediction vs. Ground Truth")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Label")
    ax.legend()
    plt.tight_layout()
    plt.show()

def domain_accuracy_method(pred_labels, true_labels):
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    scores = np.zeros_like(true, dtype=float)

    exact_matches = (pred == true)
    scores[exact_matches] = 1.0

    unknown_mask = (pred == 0.5)
    scores[unknown_mask] = 0.5

    return np.mean(scores)

# --------------------------------------------------------
#   Wrappers
# --------------------------------------------------------
def train_model(scores_list, labels_list, save_path="my_tsc_state.joblib"):
    tsc = TimeSeriesClustering()
    tsc.hyperparameter_optimization_optuna(scores_list, labels_list)  # uses default n_trials
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


from collections import defaultdict
import csv

import pandas as pd
from collections import defaultdict
import os

def load_and_process_data(file_paths, output_path='combined_data.csv'):
    # This is to read and combine LipDF
    all_dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None, names=['file_path', 'label_str', 'value'])
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
        counter +=1
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

import torch
from typing import List, Dict
from tqdm.auto import tqdm
from metrics import AP, AR

def evaluate_whole_dataset(videos: List[List[float]], labels: List[List[int]], tsc_model_path: str) -> Dict[str, Dict[str, Dict[float, float]]]:
    """
    Evaluates the entire dataset by making predictions using each clustering method
    (plus "naive" and the new "sliding_window" baseline) in the trained TimeSeriesClustering model,
    and computing AP and AR at IoU thresholds 0.5, 0.75, and 0.95.

    :param videos: List of lists containing time-series scores.
    :param labels: List of lists containing corresponding ground truth labels.
    :param tsc_model_path: Path to the saved TimeSeriesClustering model.
    :return: Dictionary containing AP and AR results for each method.
    """

    tsc_instance = TimeSeriesClustering.load_all(tsc_model_path)

    # Extended methods list to include the new "sliding_window"
    clustering_methods = ["dbscan", "hdbscan", "flasc", "naive","gaus_dbscan", "gaus_hdbscan", "gaus_flasc", "gaus_naive", "sliding_window15","hyst02","hyst_gaus02","hyst04","hyst_gaus04","hyst06","hyst_gaus06","ddm"]
    
    evaluation_results = {}
    method_predictions = {method: [] for method in clustering_methods}

    # Generate predictions for all videos
    print("\n[INFO] Generating predictions for all methods...")
    for video_scores in tqdm(videos, desc="Processing videos"):
        pred_dict = tsc_instance.predict_all(video_scores)
        
        for method in clustering_methods:
            selected_preds = pred_dict.get(method)
            if selected_preds is not None:
                method_predictions[method].append(selected_preds)
            else:
                print(f"[WARNING] {method} did not produce predictions for a sequence.")
    
    # Calculate metrics for each method using the stored predictions
    for method in clustering_methods:
        print(f"\n[INFO] Evaluating using {method.upper()} predictions...")
        predictions = method_predictions[method]

        ap_calculator = AP(iou_thresholds=[0.5, 0.75, 0.95])
        ar_calculator2 = AR(n_proposals_list=2)
        ar_calculator3 = AR(n_proposals_list=3)
        ar_calculator5 = AR(n_proposals_list=5)
        ar_calculator10 = AR(n_proposals_list=10)
        ar_calculator20 = AR(n_proposals_list=20)
        ar_calculator50 = AR(n_proposals_list=50)
        ar_calculator100 = AR(n_proposals_list=100)

        ap_results = ap_calculator(predictions, labels)
        ar_results ={}
        ar_results[2]=ar_calculator2(predictions, labels)[2]
        ar_results[3]=ar_calculator3(predictions, labels)[3]
        ar_results[5]=ar_calculator5(predictions, labels)[5]
        ar_results[10]=ar_calculator10(predictions, labels)[10]
        ar_results[20]=ar_calculator20(predictions, labels)[20]
        ar_results[50]=ar_calculator50(predictions, labels)[50]
        ar_results[100]=ar_calculator100(predictions, labels)[100]
        

        evaluation_results[method] = {"AP": ap_results, "AR": ar_results}

        print(f"\nEvaluation Results for {method.upper()}:")
        print("  Average Precision:", ap_results)
        print("  Average Recall:", ar_results)

    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Method':<14} | {'Metric':<5} | {'IoU=0.5':<10} | {'IoU=0.75':<10} | {'IoU=0.95':<10}")
    print("-"*70)

    # Print table header
    header = f"{'Method':<20} | {'AP0.5':<8} | {'AP0.75':<8} | {'AP0.95':<8} | {'AR2':<8} | {'AR3':<8} | {'AR5':<8} | {'AR10':<8} | {'AR20':<8} | {'AR50':<8} | {'AR100':<8}"
    print(header)
    print("-" * len(header))

    for method in clustering_methods:
        ap_values = evaluation_results[method].get("AP", {})
        ar_values = evaluation_results[method].get("AR", {})

        # Extract values, default to "N/A" if not present
        ap_05 = f"{ap_values.get(0.5, 'N/A'):.4f}" 
        ap_075 = f"{ap_values.get(0.75, 'N/A'):.4f}" 
        ap_095 = f"{ap_values.get(0.95, 'N/A'):.4f}" 
        
        ar_2 = f"{ar_values.get(2, 'N/A'):.4f}" if isinstance(ar_values.get(2), float) else "N/A"
        ar_3 = f"{ar_values.get(3, 'N/A'):.4f}" if isinstance(ar_values.get(3), float) else "N/A"
        ar_5 = f"{ar_values.get(5, 'N/A'):.4f}" if isinstance(ar_values.get(5), float) else "N/A"
        ar_10 = f"{ar_values.get(10, 'N/A'):.4f}" if isinstance(ar_values.get(10), float) else "N/A"
        ar_20 = f"{ar_values.get(20, 'N/A'):.4f}" if isinstance(ar_values.get(20), float) else "N/A"
        ar_50 = f"{ar_values.get(50, 'N/A'):.4f}" if isinstance(ar_values.get(50), float) else "N/A"
        ar_100 = f"{ar_values.get(100, 'N/A'):.4f}" if isinstance(ar_values.get(100), float) else "N/A"

        # Print row
        print(f"{method:<20} | {ap_05:<8} | {ap_075:<8} | {ap_095:<8}| {ar_2:<8} | {ar_3:<8} | {ar_5:<8} | {ar_10:<8} | {ar_20:<8} | {ar_50:<8} | {ar_100:<8}")


    print("="*80)

    return evaluation_results


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
        video_id = video_path.split('/')[-1].split('.')[0]  # Extracts "000001" from "test/000001.mp4"
        
        window_id = int(row['WindowID'])  # Use WindowID for sorting within a video
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #videos, labels, combined_data = load_and_process_data(["data84.csv"])
    videos, labels, combined_data = load_and_process_data(["data91.csv","data_batch_16.csv","data_batch_17.csv","data_batch_18.csv","data_batch_13.csv","data_batch_14.csv","data_batch_12.csv","data_batch_11.csv"])
    #videos, labels, combined_data = load_and_process_data_2("pred_final_new_model.csv")
    # Example usage:
    tsc_instance = TimeSeriesClustering()
    evaluate_whole_dataset(videos, labels, "my_tsc_state.joblib")
    print("")
