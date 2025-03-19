import numpy as np
from numpy.typing import NDArray


def hysteresis_threshold_1d(scores: NDArray, low_thr: float, high_thr: float) -> NDArray:
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
        labels[i] = 1 --> fake
        labels[i] = 0 --> real
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
