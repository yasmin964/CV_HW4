import json
import numpy as np
from explorer import navigate, load_ply, auto_fix_axes, detect_floor_plane, build_occupancy_grid
from path_planner import generate_trajectory
from renderer import render_frames

def get_good_start_goal_from_free_space(ply_path):
    """Find actually free points by analyzing occupancy grid"""

    # Load point cloud and build occupancy grid
    pcd = load_ply(ply_path)
    pcd, perm, inv_perm = auto_fix_axes(pcd)
    floor_z = detect_floor_plane(pcd)
    grid, xmin, ymin, res = build_occupancy_grid(pcd, floor_z)

    # Find all free cells
    free_cells = np.argwhere(grid == 0)

    if len(free_cells) < 2:
        raise RuntimeError("Not enough free space found")

    print(f"Free cells found: {len(free_cells)}")

    # Find the largest connected component of free space
    from scipy.ndimage import label
    labeled, num_components = label(grid == 0)
    counts = np.bincount(labeled.ravel())
    largest_component_id = np.argmax(counts[1:]) + 1

    # Take points only from the largest component
    main_free_cells = free_cells[labeled[free_cells[:, 0], free_cells[:, 1]] == largest_component_id]

    if len(main_free_cells) < 2:
        raise RuntimeError("Main free area is too small")

    print(f"Points in main free area: {len(main_free_cells)}")

    # Find points with maximum distance between them for a nice path
    from scipy.spatial.distance import cdist

    # Take a random sample of points for speed
    sample_size = min(100, len(main_free_cells))
    sample_indices = np.random.choice(len(main_free_cells), sample_size, replace=False)
    sample_cells = main_free_cells[sample_indices]

    # Find the pair of points with maximum distance
    distances = cdist(sample_cells, sample_cells)
    max_idx = np.unravel_index(np.argmax(distances), distances.shape)

    start_grid = sample_cells[max_idx[0]]
    goal_grid = sample_cells[max_idx[1]]

    print(f"Selected grid points: start={start_grid}, goal={goal_grid}")

    # Convert to world coordinates
    start_x = xmin + start_grid[0] * res + res / 2  # +res/2 for cell center
    start_y = ymin + start_grid[1] * res + res / 2
    goal_x = xmin + goal_grid[0] * res + res / 2
    goal_y = ymin + goal_grid[1] * res + res / 2

    # Consider permutation = (0, 2, 1)
    camera_height = 1.7
    start = (start_x, floor_z + camera_height, start_y)  # (X, Z, Y)
    goal = (goal_x, floor_z + camera_height, goal_y)  # (X, Z, Y)

    return start, goal


def save_camera_trajectory_to_json(trajectory, lookat_points, output_path="panorama_path.json"):
    """Save camera trajectory in X,Y,Z order for HTML/Three.js"""

    camera_data = []

    for pos, lookat in zip(trajectory, lookat_points):
        pos_xyz = [pos[0], pos[2], pos[1]]  # X,Z,Y -> X,Y,Z
        lookat_xyz = [lookat[0], lookat[2], lookat[1]]

        camera_frame = {
            "pos": pos_xyz,
            "target": lookat_xyz,
            "up": [0, -1, 0]
        }
        camera_data.append(camera_frame)

    with open(output_path, 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f"Camera trajectory for HTML saved to: {output_path}")
    print(f"Total frames: {len(camera_data)}")

if __name__ == "__main__":
    ply_path = r"spark_viewer/ConferenceHall_uncompressed.ply"
    out_video = r"outputs/scene_1/spark_tour.mp4"

    start, goal = get_good_start_goal_from_free_space(ply_path)
    print(f"Automatically selected points:")
    print(f"START: {start}")
    print(f"GOAL: {goal}")

    print("Building path along floor (A*)...")
    pcd, path3d, center, perm = navigate(ply_path, start, goal)
    print(f"Found path with {len(path3d)} points")

    # Show actual path points
    if len(path3d) > 0:
        actual_start = path3d[0]
        actual_goal = path3d[-1]
        print(f"Actual path points:")
        print(f"START: {actual_start}")
        print(f"GOAL: {actual_goal}")

    print("Generating smooth trajectory...")
    traj, lookats = generate_trajectory(
        path_points=path3d,
        total_frames=30 * 60
    )

    print("Saving camera trajectory to JSON...")
    save_camera_trajectory_to_json(traj, lookats, "spark_viewer/camera_path.json")

    print("Starting distributed rendering with Spark...")
    render_frames(
        ply_path,
        traj,
        lookats,
        out_video,
        fps=30,
        resolution=(1280, 720),  # Increased resolution
        visible=True,  # For Spark it's better not to show windows
        perm=perm,
        flip_camera=True,
        use_spark=True  # Enable Spark rendering
    )

print("Done! Video saved to:", out_video)