import numpy as np
import matplotlib.pyplot as plt
from localize.TimeSerieClustering import TimeSeriesClustering
from .metrics import AP, AR
from tqdm import tqdm
from typing import List, Dict


def domain_accuracy_method(pred_labels, true_labels):
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    scores = np.zeros_like(true, dtype=float)

    exact_matches = (pred == true)
    scores[exact_matches] = 1.0

    unknown_mask = (pred == 0.5)
    scores[unknown_mask] = 0.5

    return np.mean(scores)


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

    evaluation_results = {}
    # We'll initialize method_predictions after we get the first prediction
    # to dynamically capture all available methods
    method_predictions = None

    # Generate predictions for all videos
    print("\n[INFO] Generating predictions for all methods...")
    for idx, video_scores in enumerate(tqdm(videos, desc="Processing videos")):
        pred_dict = tsc_instance.predict_all(video_scores)

        # For the first video, initialize method_predictions with all available methods
        if idx == 0:
            clustering_methods = list(pred_dict.keys())
            method_predictions = {method: [] for method in clustering_methods}

        # Store predictions for each method
        for method in clustering_methods:
            selected_preds = pred_dict.get(method)
            if selected_preds is not None:
                method_predictions[method].append(selected_preds)
            else:
                print(
                    f"[WARNING] {method} did not produce predictions for a sequence.")

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
        ar_results = {}
        ar_results[2] = ar_calculator2(predictions, labels)[2]
        ar_results[3] = ar_calculator3(predictions, labels)[3]
        ar_results[5] = ar_calculator5(predictions, labels)[5]
        ar_results[10] = ar_calculator10(predictions, labels)[10]
        ar_results[20] = ar_calculator20(predictions, labels)[20]
        ar_results[50] = ar_calculator50(predictions, labels)[50]
        ar_results[100] = ar_calculator100(predictions, labels)[100]

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

        ar_2 = f"{ar_values.get(2, 'N/A'):.4f}" if isinstance(
            ar_values.get(2), float) else "N/A"
        ar_3 = f"{ar_values.get(3, 'N/A'):.4f}" if isinstance(
            ar_values.get(3), float) else "N/A"
        ar_5 = f"{ar_values.get(5, 'N/A'):.4f}" if isinstance(
            ar_values.get(5), float) else "N/A"
        ar_10 = f"{ar_values.get(10, 'N/A'):.4f}" if isinstance(
            ar_values.get(10), float) else "N/A"
        ar_20 = f"{ar_values.get(20, 'N/A'):.4f}" if isinstance(
            ar_values.get(20), float) else "N/A"
        ar_50 = f"{ar_values.get(50, 'N/A'):.4f}" if isinstance(
            ar_values.get(50), float) else "N/A"
        ar_100 = f"{ar_values.get(100, 'N/A'):.4f}" if isinstance(
            ar_values.get(100), float) else "N/A"

        # Print row
        print(f"{method:<20} | {ap_05:<8} | {ap_075:<8} | {ap_095:<8}| {ar_2:<8} | {ar_3:<8} | {ar_5:<8} | {ar_10:<8} | {ar_20:<8} | {ar_50:<8} | {ar_100:<8}")

    print("="*80)

    return evaluation_results
