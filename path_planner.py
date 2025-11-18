import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

def greedy_order_from_center(nodes):
    positions = np.array([p for p,_ in nodes])
    center = np.mean(positions, axis=0)
    # find start = closest to center
    dists = np.linalg.norm(positions - center[None,:], axis=1)
    start = int(np.argmin(dists))
    n = len(positions)
    visited = [start]
    remaining = set(range(n)) - {start}
    cur = start
    while remaining:
        rem = np.array(list(remaining))
        d = np.linalg.norm(positions[rem] - positions[cur], axis=1)
        idx = rem[int(np.argmin(d))]
        visited.append(idx)
        remaining.remove(idx)
        cur = idx
    return visited

def nodes_positions(nodes, order=None):
    if order is None:
        return np.array([p for p,_ in nodes])
    else:
        positions = np.array([p for p,_ in nodes])
        return positions[order]

def smooth_path(positions, n_samples=300):
    if positions.shape[0] < 2:
        return positions
    t = np.linspace(0, 1, positions.shape[0])
    cs = [CubicSpline(t, positions[:,i]) for i in range(3)]
    ts = np.linspace(0, 1, n_samples)
    pts = np.vstack([cs[0](ts), cs[1](ts), cs[2](ts)]).T
    return pts

def generate_trajectory(nodes, center, total_frames=30*60):
    order = greedy_order_from_center(nodes)
    ordered_pos = nodes_positions(nodes, order)
    smoothed = smooth_path(ordered_pos, n_samples=total_frames)
    lookats = np.zeros_like(smoothed)
    for i in range(len(smoothed) - 1):
        lookats[i] = smoothed[i + 1]
    lookats[-1] = smoothed[-1]

    return smoothed, lookats
