"""Distances for row matrices; Jaccard treats binary rows as approval sets."""

import numpy as np
from scipy.spatial.distance import cdist


def row_distances(left, right, metric="hamming"):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("distance inputs must be row matrices of equal width")
    if metric == "jaccard":
        if not all(np.all((a == 0) | (a == 1)) for a in (left, right)):
            raise ValueError("Jaccard distance requires binary approval rows")
        # Use integer intersection/union counts, including d(empty, empty)=0.
        return cdist(left, right, "jaccard")
    if metric == "hamming":
        if left.shape[1] == 0:
            return np.zeros((len(left), len(right)), dtype=np.int64)
        return np.rint(cdist(left, right, "hamming") * left.shape[1]).astype(np.int64)
    if metric == "euclidean":
        return cdist(left, right, "euclidean")
    raise ValueError(f"unknown metric: {metric}")
