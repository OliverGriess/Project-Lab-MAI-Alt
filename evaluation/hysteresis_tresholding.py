import numpy as np


def hysteresis_threshold_1d(scores, low_thr, high_thr):
    """
    Apply hysteresis thresholding on a 1D array of scores.

    Parameters
    ----------
    scores : 1D list or numpy array of floats
        The scores (e.g. deep-fake likelihood) in [0, 1].
    low_thr : float
        The lower threshold for hysteresis.
    high_thr : float
        The upper threshold for hysteresis.

    Returns
    -------
    labels : numpy array of int
        An array of the same length as 'scores'.
        labels[i] = 0 --> real
        labels[i] = 1 --> fake
    """
    scores = np.array(scores, dtype=float)
    n = len(scores)

    # Step 1: Initialize labels to 0 (real)
    labels = np.zeros(n, dtype=int)

    # Step 2: Mark definitely fake (>= high_thr) as 1
    labels[scores >= high_thr] = 1

    # Step 3: Mark uncertain/floating values (-1) where low_thr <= score < high_thr
    uncertain_mask = (scores >= low_thr) & (scores < high_thr)
    labels[uncertain_mask] = -1

    # Step 4: Expand fake regions into uncertain neighbors until no more changes
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if labels[i] == 1:
                # Check left neighbor
                if i > 0 and labels[i - 1] == -1:
                    labels[i - 1] = 1
                    changed = True
                # Check right neighbor
                if i < n - 1 and labels[i + 1] == -1:
                    labels[i + 1] = 1
                    changed = True

    # Step 5: All remaining -1 become 0 (real)
    labels[labels == -1] = 0

    return labels


def extract_fake_windows(labels):
    """
    Given a 1D array of binary labels (1=fake, 0=real),
    return a list of (start_index, end_index) for consecutive fake segments.
    """
    windows = []
    n = len(labels)
    start = None

    for i in range(n):
        if labels[i] == 1 and start is None:
            start = i
        elif labels[i] == 0 and start is not None:
            # ended a run of 1's
            windows.append((start, i - 1))
            start = None

    # if the array ends in a 1-run
    if start is not None:
        windows.append((start, n - 1))

    return windows


def merge_small_gaps(windows, max_gap=1):
    """
    Optionally merge windows if the gap between them is small.
    For example, if max_gap=1, then two windows separated
    by 1 or fewer frames get merged into a single window.
    """
    if not windows:
        return []

    merged = [windows[0]]
    for current in windows[1:]:
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = current

        gap = cur_start - prev_end - 1
        if gap <= max_gap:
            # Merge
            new_window = (prev_start, cur_end)
            merged[-1] = new_window
        else:
            merged.append(current)

    return merged


def find_fake_windows(scores, low_thr=0.3, high_thr=0.7, gap_thr=1):
    """
    Full pipeline:
      1) Hysteresis thresholding
      2) Extract raw fake windows
      3) Optionally merge small real gaps

    Note: scores are expected to be 0 for real and 1 for fake.

    Returns a list of (start_index, end_index) indicating fake segments.
    """
    labels = hysteresis_threshold_1d(scores, low_thr, high_thr)
    windows = extract_fake_windows(labels)

    # If you want to merge small gaps, set gap_thr > 0
    if gap_thr > 0:
        windows = merge_small_gaps(windows, max_gap=gap_thr)

    return windows


def windows_to_periods(
    windows: list[tuple[int, int]], frame_rate: int
) -> list[tuple[float, float]]:
    """
    Convert windows to periods in seconds
    """
    periods = []
    for window in windows:
        start, end = window
        periods.append((start / frame_rate, end / frame_rate))
    return periods


if __name__ == "__main__":
    # Example usage:
    # scores = [0, 1, 0.5, 0.6, 0, 1, 0.2, 0.2, 0.4, 1, 0.7, 0.8] # (should be 2 windows, i.e. (1-5) and (8-11))
    scores = [
        0,
        0.5,
        0.7,
        0.8,
        0,
        0.2,
        0.4,
        0.1,
        0,
        0.6,
        0.7,
        0.9,
    ]  # (should be 2 windows, i.e. (1-3) and (9-11))

    # these tresholds seem to work well
    low_thr = 0.3
    high_thr = 0.7
    gap_thr = 1

    windows = find_fake_windows(scores, low_thr, high_thr, gap_thr)
    print("Scores:", scores)
    print("Fake windows (start-end):", windows)
