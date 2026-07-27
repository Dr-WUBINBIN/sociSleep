"""Identity assignment and position averaging utilities."""

import numpy as np


def nearest_assignment(preds, dets, max_dist=150):
    """
    Greedy nearest-neighbor assignment between predicted positions and detections.

    Returns (mapping dict label->detection, list of unmatched detection indices).
    """
    mapping = {label: None for label in preds.keys()}
    if len(dets) == 0:
        return mapping, list(range(len(dets)))
    costs = []
    for label, p in preds.items():
        if p is None:
            continue
        for i, d in enumerate(dets):
            dist = np.hypot(p[0] - d[0], p[1] - d[1])
            costs.append((dist, label, i))
    costs.sort(key=lambda x: x[0])
    used_labels, used_inds = set(), set()
    for dist, label, i in costs:
        if label in used_labels or i in used_inds:
            continue
        if dist <= max_dist:
            mapping[label] = dets[i]
            used_labels.add(label)
            used_inds.add(i)
    unmatched = [i for i in range(len(dets)) if i not in used_inds]
    return mapping, unmatched


def avg_position(list_pts):
    """Return the mean (x, y) of non-None points, or None if empty."""
    pts = [p for p in list_pts if p is not None]
    if not pts:
        return None
    return (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))
