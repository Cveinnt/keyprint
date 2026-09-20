"""Development selector; not imported by the SDK or frozen reference filter."""
import numpy as np


def select(candidate_ids, scores, top_k):
    """Same descending-score/ascending-ID order on validated filter candidates.

    The filter supplies strictly increasing candidate IDs and finite float64
    scores converted exactly from float32. This helper does not change filtering,
    temperature arithmetic, gap boundaries or the final immutable probability
    support. It only avoids sorting discarded candidates.
    """
    if len(scores) <= top_k:
        return np.lexsort((candidate_ids, -scores))
    if np.all(scores == scores[0]):
        return np.arange(top_k)
    threshold = np.partition(scores, len(scores) - top_k)[len(scores) - top_k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:top_k - len(above)]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((candidate_ids[selected], -scores[selected]))]
