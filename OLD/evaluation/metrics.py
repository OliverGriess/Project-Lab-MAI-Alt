import torch
from typing import List, Dict, Union
from tqdm.auto import tqdm
import numpy as np


def iou_1d(proposal, target):
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union

def iou_1d_oli(pred_intervals: torch.Tensor, gt_intervals: torch.Tensor) -> torch.Tensor:
    """Computes IoU for 1D intervals (start, end)."""
    if len(gt_intervals) == 0 or len(pred_intervals) == 0:
        return torch.zeros((len(pred_intervals), len(gt_intervals)))

    # Get intersection
    inter_start = torch.max(pred_intervals[:, None, 0], gt_intervals[None, :, 0])
    inter_end = torch.min(pred_intervals[:, None, 1], gt_intervals[None, :, 1])
    inter = torch.clamp(inter_end - inter_start, min=0)

    # Compute union
    pred_lengths = pred_intervals[:, 1] - pred_intervals[:, 0]
    gt_lengths = gt_intervals[:, 1] - gt_intervals[:, 0]
    union = pred_lengths[:, None] + gt_lengths[None, :] - inter

    return inter / union  # IoU matrix


class AP:
    """
    Computes Average Precision (AP) at different IoU thresholds.
    """

    def __init__(self, iou_thresholds: Union[float, List[float]] = [0.5, 0.75, 0.95]):
        self.iou_thresholds = iou_thresholds if isinstance(iou_thresholds, list) else [iou_thresholds]
        self.n_labels = 0
        self.ap: Dict[float, float] = {}

    def __call__(self, predictions: List[List[int]], ground_truths: List[List[int]]) -> Dict[float, float]:
        """
        Compute AP for multiple IoU thresholds.

        :param predictions: List of lists containing predicted labels (0 or 1).
        :param ground_truths: List of lists containing ground truth labels (0 or 1).
        :return: Dictionary with AP values at each IoU threshold.
        """

        proposals_dict, labels_dict = self._convert_to_intervals(predictions, ground_truths)

        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = sum(len(v) for v in labels_dict.values())

            for key in tqdm(proposals_dict.keys(), desc=f"AP IoU {iou_threshold}"):
                proposals = proposals_dict[key]
                labels = labels_dict[key]
                values.append(self._get_values(iou_threshold, proposals, labels))

            # Sort proposals by confidence
            values = torch.cat(values) if values else torch.zeros((0, 2))
            if len(values) == 0:
                self.ap[iou_threshold] = 0.0
                continue

            ind = values[:, 0].sort(stable=True, descending=True).indices
            values = values[ind]

            # Compute precision-recall curve
            curve = self._calculate_curve(values)
            self.ap[iou_threshold] = self._calculate_ap(curve)

        return self.ap

    def _convert_to_intervals(self, predictions: List[List[int]], ground_truths: List[List[int]]):
        proposals_dict, labels_dict = {}, {}

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Convert predictions and ground truths to (start, end) intervals
            proposals_dict[f"video_{i}"] = self._extract_intervals(pred)
            labels_dict[f"video_{i}"] = self._extract_intervals(gt)

        return proposals_dict, labels_dict

    def _extract_intervals(self, sequence: List[int]):
        """Extracts (start, end) intervals from binary sequences."""
        intervals = []
        start = None

        for i, label in enumerate(sequence):
            if label == 1 and start is None:
                start = i
            elif label == 0 and start is not None:
                intervals.append([start, i])
                start = None

        if start is not None:
            intervals.append([start, len(sequence)])

        return torch.tensor(intervals, dtype=torch.float32)

    def _get_values(self, iou_threshold: float, proposals: torch.Tensor, labels: torch.Tensor):
        if proposals.shape[0] == 0:
            return torch.zeros((0, 2))

        ious = iou_1d(proposals, labels) if labels.shape[0] > 0 else torch.zeros((len(proposals), 0))
        is_TP = (ious > iou_threshold).any(dim=1).float()
        confidence = torch.ones(len(proposals))

        return torch.stack([confidence, is_TP], dim=1)

    def _calculate_curve(self, values):
        acc_TP = torch.cumsum(values[:, 1], dim=0)
        precision = acc_TP / (torch.arange(len(acc_TP)) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def _calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        return (x_diff * y_max[:-1]).sum()


class AR:
    """
    Computes Average Recall (AR) at different IoU thresholds with proposal candidates.
    """

    def __init__(self, n_proposals_list: Union[List[int], int] = 100, iou_thresholds: List[float] = None):
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] #[0.5]
        
        self.n_proposals_list = n_proposals_list if isinstance(n_proposals_list, list) else [n_proposals_list]
        self.n_proposals_list = torch.tensor(self.n_proposals_list)
        self.iou_thresholds = iou_thresholds
        self.ar: dict = {}

    def __call__(self, predictions: List[List[int]], ground_truths: List[List[int]]) -> dict:
        proposals_dict, labels_dict = self._convert_to_intervals(predictions, ground_truths)

        values = torch.zeros((len(predictions), len(self.iou_thresholds), len(self.n_proposals_list), 2))
        for i in range(len(tqdm(predictions))):
            proposals = torch.tensor(proposals_dict[f"video_{i}"])
            labels = torch.tensor(labels_dict[f"video_{i}"])
            values[i] = self._get_values(self.iou_thresholds, proposals, labels, 25.)

        values_sum = values.sum(dim=0)

        TP = values_sum[:, :, 0]
        FN = values_sum[:, :, 1]
        recall = TP / (TP + FN)  # (n_iou_thresholds, n_proposal_thresholds)
        for i, n_proposals in enumerate(self.n_proposals_list):
            self.ar[n_proposals.item()] = recall[:, i].mean().item()

        return self.ar

        # valid_values = []
        # for key in tqdm(proposals_dict.keys(), desc="Computing AR"):
        #     proposals = proposals_dict[key]
        #     labels = labels_dict[key]
        #     try:
        #         vals = self._get_values(self.iou_thresholds, proposals, labels)
        #         valid_values.append(vals)
        #     except Exception as e:
        #         # Just skip this example if any error occurs in _get_values
        #         # (e.g. shape mismatch, invalid indexing, etc.)
        #         print(f"Skipping {key} due to error: {e}")
        #         continue

        # # If no valid examples, you could return an empty dict or zeros
        # if not valid_values:
        #     print("No valid examples found; returning empty results.")
        #     return self.ar

        # values_tensor = torch.stack(valid_values, dim=0)
        # values_sum = values_tensor.sum(dim=0)
        # TP = values_sum[:, :, 0]
        # FN = values_sum[:, :, 1]
        # recall = TP / (TP + FN)  # (n_iou_thresholds, n_proposal_thresholds)
        
        # for i, n_proposals in enumerate(self.n_proposals_list):
        #     self.ar[n_proposals.item()] = recall[:, i].mean().item()
        
        # return self.ar

    def _convert_to_intervals(self, predictions: List[List[int]], ground_truths: List[List[int]]):
        proposals_dict, labels_dict = {}, {}

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            proposals_dict[f"video_{i}"] = self._extract_intervals_filling(pred)
            labels_dict[f"video_{i}"] = self._extract_intervals(gt)
        
        return proposals_dict, labels_dict

    def _extract_intervals(self, sequence: List[int]):
        intervals = []
        start = None
        current_interval = -1

        for i, label in enumerate(sequence):
            if start is None:
                start = i
                current_interval = label
            elif label != current_interval:
                if sequence[start] == 1:
                    intervals.append([start, i])
                start = i
                current_interval = label

        return torch.tensor(intervals, dtype=torch.float32)

    def _extract_intervals_filling(self, sequence: List[int]):
        intervals = []
        start = None
        current_interval = -1

        for i, label in enumerate(sequence):
            if start is None:
                start = i
                current_interval = label
            elif label != current_interval:
                if sequence[start] == 1:
                    intervals.append([start, i])
                start = i
                current_interval = label

        if len(intervals) > 0:
            intervals = sorted(intervals, key=lambda x: x[1] - x[0], reverse=True)
            while len(intervals) < 100:
                intervals.append(intervals[-1])
        else:
            while len(intervals) < 100:
                intervals.append([])
        return torch.tensor(intervals, dtype=torch.float32)

    def _get_values(
        self,
        iou_thresholds: List[float],
        proposals,
        labels,
        fps: float,
    ):
        n_proposals_list = self.n_proposals_list
        max_proposals = max(n_proposals_list)
        
        proposals = proposals[:max_proposals]
        n_labels = len(labels)

        if n_labels > 0:
            if proposals.shape[1] != 0:
                ious = iou_1d(proposals, labels)
            else:
                ious = torch.zeros((max_proposals, 0))
        else:
            ious = torch.zeros((max_proposals, 0))

        # values: matrix of (TP, FN), shapes (n_iou_thresholds, n_proposal_thresholds, 2)
        iou_max = ious.cummax(0).values[n_proposals_list - 1]  # shape (n_iou_thresholds, n_labels)
        iou_max = iou_max[None]

        iou_thresholds = torch.tensor(iou_thresholds)[:, None, None]
        TP = (iou_max > iou_thresholds).sum(-1)
        FN = n_labels - TP
        values = torch.stack([TP, FN], dim=-1)

        return values
    
    # def _get_values(self, iou_thresholds: List[float], proposals: torch.Tensor, labels: torch.Tensor):
    #     n_proposals_list = self.n_proposals_list
    #     max_proposals = max(n_proposals_list)

    #     # Example: If proposals is too small or something else is off, 
    #     # ensure you handle that or raise an exception.
    #     if proposals.size(0) < max_proposals:
    #         raise ValueError("Not enough proposals to slice up to max_proposals.")

    #     proposals = proposals[:max_proposals]
    #     n_labels = len(labels)

    #     if n_labels > 0:
    #         ious = iou_1d(proposals, labels)
    #     else:
    #         # If you want to allow zero labels as valid, handle it gracefully
    #         ious = torch.zeros((max_proposals, 0))

    #     # Could still raise errors if out of range
    #     max_index = max_proposals - 1
    #     iou_max = ious.cummax(dim=0).values[max_index]  # shape: (M, )

    #     iou_max = iou_max[None]  # shape now (1, M)

    #     iou_thresholds = torch.tensor(iou_thresholds)[:, None, None]  # shape (n_iou, 1, 1)
    #     TP = (iou_max > iou_thresholds).sum(-1)  # shape (n_iou, 1)
    #     FN = n_labels - TP
    #     values = torch.stack([TP, FN], dim=-1)  # shape (n_iou, 1, 2)

    #     # Because we slice proposals up to max_proposals, we want one dimension
    #     # for each value in n_proposals_list. So re-expanding or reindexing might be needed,
    #     # as in your original code, if you want shape = (n_iou, len(n_proposals_list), 2)
    #     return values