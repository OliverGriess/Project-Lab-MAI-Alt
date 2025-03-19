from typing import List, Tuple
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from flasc import FLASC
from scipy.ndimage import gaussian_filter1d
from .hysteresis_tresholding import hysteresis_threshold_1d
from .DDM import DriftDetectionMethod
from dataclasses import dataclass, asdict, field


@dataclass
class BaseParams:
    def to_dict(self):
        return asdict(self)


@dataclass
class DBSCANParams(BaseParams):
    eps: float = 1.0
    min_samples: int = 5
    threshold: float = 0.4


@dataclass
class HDBSCANParams(BaseParams):
    min_cluster_size: int = 5
    min_samples: int = None
    threshold: float = 0.4


@dataclass
class FLASCParams(BaseParams):
    min_cluster_size: int = 5
    threshold: float = 0.4


@dataclass
class HysteresisParams(BaseParams):
    tresholds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.6), (0.4, 0.8), (0.6, 0.8)])


@dataclass
class SlidingWindowParams(BaseParams):
    window_size: int = 15
    threshold: float = 0.5


@dataclass
class DDMParams(BaseParams):
    detection_level: float = 0.9
    window_size: int = 3


class TimeSeriesClustering:
    def __init__(self):
        # For storing best hyperparams
        self.dbscan_params = DBSCANParams()
        self.hdbscan_params = HDBSCANParams()
        self.flasc_params = FLASCParams()
        self.hysteresis_params = HysteresisParams()
        self.sliding_window_params = SlidingWindowParams()
        self.ddm_params = DDMParams()
        self.moving_average_window_size = 5
        self.gaussian_sigma = 1.6

        # A dictionary if we want to keep them all in one place
        self.best_params = {}

    def save_all(self, savepath="time_series_clustering_state.joblib"):
        joblib.dump(self, savepath)
        print(f"[INFO] Saved TimeSeriesClustering state to {savepath}.")

    @staticmethod
    def load_all(loadpath):
        loaded_obj = joblib.load(loadpath)
        if not isinstance(loaded_obj, TimeSeriesClustering):
            raise TypeError(
                "Loaded object is not a TimeSeriesClustering instance.")
        print(f"[INFO] Loaded TimeSeriesClustering state from {loadpath}.")
        return loaded_obj

    # ---------------------
    # Density-based methods
    # ---------------------

    def flasc_cluster(self, scores, min_cluster_size=5, threshold=0.4):
        X = np.column_stack((np.arange(len(scores)), scores))
        clusterer = FLASC(min_cluster_size=min_cluster_size)
        raw_labels = clusterer.fit_predict(X)
        final_labels = self._postprocess_labels_threshold(
            raw_labels, scores, threshold=threshold)
        return final_labels, clusterer

    def dbscan_cluster(self, scores, eps=1.0, min_samples=5, threshold=0.4):
        """
        Original DBSCAN approach on the raw (index, score) pairs.
        """
        X = np.column_stack((np.arange(len(scores)), scores))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        raw_labels = model.fit_predict(X)
        final_labels = self._postprocess_labels_threshold(
            raw_labels, scores, threshold=threshold)
        return final_labels, model

    def hdbscan_cluster(self, scores, min_cluster_size=5, min_samples=None, threshold=0.4):
        import hdbscan
        X = np.column_stack((np.arange(len(scores)), scores))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples)
        raw_labels = clusterer.fit_predict(X)
        final_labels = self._postprocess_labels_threshold(
            raw_labels, scores, threshold)
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
                    dbscan_params: DBSCANParams | None = None,
                    hdbscan_params: HDBSCANParams | None = None,
                    flasc_params: FLASCParams | None = None,
                    hysteresis_params: HysteresisParams | None = None,
                    sliding_window_params: SlidingWindowParams | None = None,
                    ddm_params: DDMParams | None = None,
                    moving_average_window_size: int | None = None,
                    gaussian_sigma: float | None = None):
        """
        Returns predictions from multiple clustering methods plus naive + sliding-window.
        You can also add the new 'dbscan_with_transform' here if you'd like to compare.
        """
        predictions = {}
        dbscan_params = dbscan_params or self.dbscan_params
        hdbscan_params = hdbscan_params or self.hdbscan_params
        flasc_params = flasc_params or self.flasc_params
        hysteresis_params = hysteresis_params or self.hysteresis_params
        sliding_window_params = sliding_window_params or self.sliding_window_params
        ddm_params = ddm_params or self.ddm_params
        moving_average_window_size = moving_average_window_size or self.moving_average_window_size
        gaussian_sigma = gaussian_sigma or self.gaussian_sigma

        dbscan_labels, _ = self.dbscan_cluster(
            scores, **dbscan_params.to_dict())
        predictions['dbscan'] = dbscan_labels

        hdbscan_labels, _ = self.hdbscan_cluster(
            scores, **hdbscan_params.to_dict())
        predictions['hdbscan'] = hdbscan_labels

        flasc_labels, _ = self.flasc_cluster(
            scores, **flasc_params.to_dict())
        predictions['flasc'] = flasc_labels

        for low_thr, high_thr in hysteresis_params.tresholds:
            hyst_labels = hysteresis_threshold_1d(
                scores=scores, low_thr=low_thr, high_thr=high_thr)
            predictions[f'hyst_{low_thr}_{high_thr}'] = hyst_labels

        # Add gaussian smoothing
        gaus_score = gaussian_filter1d(scores, sigma=gaussian_sigma)

        gaus_dbscan_labels, _ = self.dbscan_cluster(
            gaus_score, **dbscan_params.to_dict())
        predictions['gaus_dbscan'] = gaus_dbscan_labels

        gaus_hdbscan_labels, _ = self.hdbscan_cluster(
            gaus_score, **hdbscan_params.to_dict())
        predictions['gaus_hdbscan'] = gaus_hdbscan_labels

        gaus_flasc_labels, _ = self.flasc_cluster(
            gaus_score, **flasc_params.to_dict())
        predictions['gaus_flasc'] = gaus_flasc_labels

        # Add moving average calculation
        ma_score = np.convolve(scores, np.ones(
            moving_average_window_size)/moving_average_window_size, mode='same')

        # Add moving average versions of the clustering algorithms
        ma_dbscan_labels, _ = self.dbscan_cluster(
            ma_score, **dbscan_params.to_dict())
        predictions['ma_dbscan'] = ma_dbscan_labels

        ma_hdbscan_labels, _ = self.hdbscan_cluster(
            ma_score, **hdbscan_params.to_dict())
        predictions['ma_hdbscan'] = ma_hdbscan_labels

        ma_flasc_labels, _ = self.flasc_cluster(
            ma_score, **flasc_params.to_dict())
        predictions['ma_flasc'] = ma_flasc_labels

        for low_thr, high_thr in hysteresis_params.tresholds:
            hyst_labels = hysteresis_threshold_1d(
                scores=gaus_score, low_thr=low_thr, high_thr=high_thr)
            predictions[f'hyst_gaus_{gaussian_sigma}_{low_thr}_{high_thr}'] = hyst_labels

        # Add moving average versions of the hysteresis thresholding
        for low_thr, high_thr in hysteresis_params.tresholds:
            hyst_labels = hysteresis_threshold_1d(
                scores=ma_score, low_thr=low_thr, high_thr=high_thr)
            predictions[f'hyst_ma_{moving_average_window_size}_{low_thr}_{high_thr}'] = hyst_labels

        ddm = []
        for subscore in scores:
            ddm.append(DriftDetectionMethod(
                **ddm_params.to_dict()).cluster(subscore))

        predictions['ddm'] = ddm

        # Baselines
        sliding_window_labels15 = self.sliding_window_baseline(
            scores, **sliding_window_params.to_dict())
        predictions[f'sliding_window_{sliding_window_params.window_size}'] = sliding_window_labels15

        binary_scores = [1 if score >= 0.5 else 0 for score in scores]
        predictions['binary_scores'] = binary_scores

        return predictions

    def _postprocess_labels_threshold(self, raw_labels, scores, threshold=0.5):
        """
        For each cluster (label >= 0), compute the average score of the cluster.
        If the average is above 'threshold', assign label 1; otherwise 0.
        Outliers (label == -1) are assigned 0.5.
        """
        final_labels = np.full_like(
            raw_labels, 0.5, dtype=float)  # default to 0.5 for outliers
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
