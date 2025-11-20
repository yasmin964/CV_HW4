import numpy as np
from scipy.interpolate import CubicSpline


def smooth_path(positions, n_samples=300):
    """Smooth a path using cubic spline interpolation"""
    if len(positions) < 2:
        return positions

    positions = np.array(positions)

    # Calculate cumulative distance along path
    distances = np.zeros(len(positions))
    for i in range(1, len(positions)):
        distances[i] = distances[i - 1] + np.linalg.norm(positions[i] - positions[i - 1])

    if distances[-1] == 0:  # All points are the same
        return positions

    # Normalize to [0, 1]
    t = distances / distances[-1]

    # Create spline for each dimension
    cs = [CubicSpline(t, positions[:, i]) for i in range(3)]

    # Sample at regular intervals
    ts = np.linspace(0, 1, n_samples)
    pts = np.vstack([cs[0](ts), cs[1](ts), cs[2](ts)]).T

    return pts


def generate_trajectory(path_points, total_frames=30 * 10):
    """
    Generate smooth camera trajectory from path points
    """
    if len(path_points) < 2:
        return path_points, path_points

    # Smooth the path
    smoothed = smooth_path(path_points, n_samples=total_frames)

    # Calculate look-at points (look slightly ahead on the path)
    lookats = np.zeros_like(smoothed)
    look_ahead_frames = min(20, len(smoothed) // 10)

    for i in range(len(smoothed)):
        if i + look_ahead_frames < len(smoothed):
            lookats[i] = smoothed[i + look_ahead_frames]
        else:
            lookats[i] = smoothed[-1]

    return smoothed, lookats