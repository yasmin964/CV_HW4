import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import binary_opening, binary_closing, label as cc_label
from queue import PriorityQueue
from collections import deque


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def detect_floor_plane(pcd, distance_thresh=0.03):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_thresh,
        ransac_n=3,
        num_iterations=500
    )

    [a, b, c, d] = plane_model

    if abs(c) < 0.8:
        raise RuntimeError("Floor plane not horizontal")

    floor_points = pcd.select_by_index(inliers)
    floor_height = np.mean(np.asarray(floor_points.points)[:, 2])

    return floor_height


def build_occupancy_grid(
    pcd,
    floor_z,
    res=0.05,
    person_height=1.0,
    floor_band=0.1,
    clearance_band=0.2,
    floor_threshold=1,
    obstacle_threshold=50,
    smooth_iters=1
):
    pts = np.asarray(pcd.points)

    pts = np.asarray(pcd.points)
    print("min", pts.min(axis=0))
    print("max", pts.max(axis=0))
    print("floor_z:", floor_z)

    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()

    W = int((xmax - xmin) / res) + 1
    H = int((ymax - ymin) / res) + 1

    # 1 means occupied/blocked, 0 means free
    grid = np.ones((W, H), dtype=np.uint8)
    floor_support = np.zeros((W, H), dtype=np.uint16)
    obstacle_count = np.zeros((W, H), dtype=np.uint16)

    max_walkable_z = floor_z + person_height

    for x, y, z in pts:
        gx = int((x - xmin) / res)
        gy = int((y - ymin) / res)

        if not (0 <= gx < W and 0 <= gy < H):
            continue

        if abs(z - floor_z) <= floor_band:
            floor_support[gx, gy] += 1

        if floor_z + clearance_band <= z <= max_walkable_z:
            obstacle_count[gx, gy] += 1

    walkable = (floor_support >= floor_threshold) & (obstacle_count <= obstacle_threshold)

    if smooth_iters > 0:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
        walkable = binary_opening(walkable, structure=structure, iterations=smooth_iters)
        walkable = binary_closing(walkable, structure=structure, iterations=smooth_iters)

    grid[walkable] = 0

    return grid, xmin, ymin, res


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def prune_path(path):
    if len(path) < 3:
        return path

    pruned = [path[0]]

    def collinear(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    for i in range(1, len(path) - 1):
        p1 = pruned[-1]
        p2 = path[i]
        p3 = path[i + 1]
        if not collinear(p1, p2, p3):
            pruned.append(p2)

    pruned.append(path[-1])
    return pruned


from scipy.ndimage import distance_transform_edt

def astar(grid, start, goal):
    W, H = grid.shape

    dist = distance_transform_edt(grid == 0)

    pq = PriorityQueue()
    pq.put((0, start))

    came = {start: None}
    cost = {start: 0}

    dirs = [
        (1,0),(0,1),(-1,0),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]

    while not pq.empty():
        _, cur = pq.get()

        if cur == goal:
            break

        for dx, dy in dirs:
            nx, ny = cur[0] + dx, cur[1] + dy

            if not (0 <= nx < W and 0 <= ny < H):
                continue

            if grid[nx, ny] != 0:
                continue

            base_cost = np.sqrt(dx**2 + dy**2)

            wall_penalty = 2.0 / (dist[nx, ny] + 1e-3)

            new_cost = cost[cur] + base_cost + wall_penalty

            if (nx, ny) not in cost or new_cost < cost[(nx, ny)]:
                cost[(nx, ny)] = new_cost
                priority = new_cost + heuristic((nx, ny), goal)
                pq.put((priority, (nx, ny)))
                came[(nx, ny)] = cur

    if goal not in came:
        raise RuntimeError("Path not found")

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came[node]

    return path[::-1]


def smooth_path(path, iterations=100, smooth_factor=0.5):
    """Better path smoothing using Laplacian smoothing"""
    if len(path) < 3:
        return path

    path = np.array(path, dtype=float)

    for _ in range(iterations):
        new_path = path.copy()
        for i in range(1, len(path) - 1):
            # Laplacian smoothing: move toward the average of neighbors
            new_path[i] = (1 - smooth_factor) * path[i] + \
                          smooth_factor * 0.5 * (path[i - 1] + path[i + 1])
        path = new_path

    return [tuple(map(int, p)) for p in path]


def find_nearest_free(grid, seed, max_radius=None):
    if grid[seed] == 0:
        return seed

    W, H = grid.shape
    visited = set()
    q = deque()
    q.append((seed, 0))
    visited.add(seed)

    while q:
        (x, y), dist = q.popleft()
        if max_radius is not None and dist > max_radius:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                    if grid[nx, ny] == 0:
                        return (nx, ny)
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
    raise RuntimeError("No nearby free cell found for the given seed point")

def grid_to_world(path, xmin, ymin, res, floor_z, cam_h=-1.5):
    """Convert grid coordinates to world coordinates with proper camera height"""
    pts = []
    for gx, gy in path:
        x = xmin + gx * res
        y = ymin + gy * res
        # Camera should be at floor height + human eye level
        z = floor_z + cam_h
        pts.append(np.array([x, y, z]))
    return pts

def auto_fix_axes(pcd):
    pts = np.asarray(pcd.points)
    ranges = pts.max(axis=0) - pts.min(axis=0)

    vertical = np.argmin(ranges)

    if vertical == 2:
        perm = (0, 1, 2)
    elif vertical == 0:
        perm = (1, 2, 0)
    else:
        perm = (0, 2, 1)

    if perm != (0, 1, 2):
        rot = pts[:, perm]
        pcd.points = o3d.utility.Vector3dVector(rot)

    inv_perm = tuple(np.argsort(perm))
    return pcd, perm, inv_perm

def navigate(ply_path, start_xyz, goal_xyz):
    pcd = load_ply(ply_path)
    pcd, perm, inv_perm = auto_fix_axes(pcd)

    print(f"Permutation: {perm}")
    print(f"Inverse permutation: {inv_perm}")

    pts = np.asarray(pcd.points)

    # ADD DEBUG INFO HERE:
    print("=== SCENE COORDINATE ANALYSIS ===")
    print(f"Full point cloud range:")
    print(f"X: {pts[:, 0].min():.2f} to {pts[:, 0].max():.2f}")
    print(f"Y: {pts[:, 1].min():.2f} to {pts[:, 1].max():.2f}")
    print(f"Z: {pts[:, 2].min():.2f} to {pts[:, 2].max():.2f}")
    print(f"Point cloud center: {pts.mean(axis=0)}")

    # detect floor
    floor_z = detect_floor_plane(pcd)
    print(f"Detected floor height (Z): {floor_z:.2f}")
    print("=================================")
    ranges = pts.max(axis=0) - pts.min(axis=0)

    center_rot = pts.mean(axis=0)

    # detect floor
    floor_z = detect_floor_plane(pcd)

    # grid
    grid, xmin, ymin, res = build_occupancy_grid(pcd, floor_z)
    free_mask = (grid == 0)
    free_ratio = np.mean(free_mask)
    print(f"Free cells in grid: {free_ratio:.2%}")

    # convert input coords to grid coords
    perm_arr = np.array(perm)
    inv_perm_arr = np.array(inv_perm)

    start_rot = np.asarray(start_xyz)[perm_arr]
    goal_rot = np.asarray(goal_xyz)[perm_arr]

    print(f"=== DEBUG COORDINATE CONVERSION ===")
    print(f"Original start: {start_xyz}")
    print(f"Original goal: {goal_xyz}")
    print(f"After perm start_rot: {start_rot}")
    print(f"After perm goal_rot: {goal_rot}")
    print(f"xmin: {xmin}, ymin: {ymin}, res: {res}")

    sx = int((start_rot[0] - xmin) / res)
    sy = int((start_rot[1] - ymin) / res)
    gx = int((goal_rot[0] - xmin) / res)
    gy = int((goal_rot[1] - ymin) / res)

    print(f"Grid coordinates - start: ({sx}, {sy}), goal: ({gx}, {gy})")
    print(f"Grid shape: {grid.shape}")
    print("=====================================")

    labeled, num_components = cc_label(free_mask)
    if num_components == 0:
        raise RuntimeError("No free areas in grid after filtering")
    counts = np.bincount(labeled.ravel())
    component_sizes = counts[1:]
    largest_index = int(np.argmax(component_sizes))
    largest_label = largest_index + 1
    largest_count = component_sizes[largest_index]
    largest_ratio = largest_count / free_mask.size
    print(f"Free components: {num_components}, "
          f"largest: {largest_count} cells ({largest_ratio:.2%})")
    grid[labeled != largest_label] = 1

    # convert input coords to grid coords
    perm_arr = np.array(perm)
    inv_perm_arr = np.array(inv_perm)

    start_rot = np.asarray(start_xyz)[perm_arr]
    goal_rot = np.asarray(goal_xyz)[perm_arr]

    sx = int((start_rot[0] - xmin) / res)
    sy = int((start_rot[1] - ymin) / res)
    gx = int((goal_rot[0] - xmin) / res)
    gy = int((goal_rot[1] - ymin) / res)

    W, H = grid.shape
    for label, x, y in (("start", sx, sy), ("goal", gx, gy)):
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"{label} point mapped to ({x}, {y}) is outside the occupancy grid after axis alignment")

    if grid[sx, sy] != 0:
        print(f"Start cell ({sx}, {sy}) is occupied, searching for nearest free...")
        old_sx, old_sy = sx, sy
        sx, sy = find_nearest_free(grid, (sx, sy))
        print(f"Found nearest free: ({sx}, {sy}) instead of ({old_sx}, {old_sy})")
    if grid[gx, gy] != 0:
        print(f"Goal cell ({gx}, {gy}) is occupied, searching for nearest free...")
        gx, gy = find_nearest_free(grid, (gx, gy))

    # find path
    path2d = astar(grid, (sx, sy), (gx, gy))

    print("Pruning path...")
    path2d = prune_path(path2d)

    print("Smoothing path...")
    path2d = smooth_path(path2d)

    # convert to 3D camera path
    path3d_rot = np.stack(grid_to_world(path2d, xmin, ymin, res, floor_z))
    path3d_world = path3d_rot[:, inv_perm_arr]
    path_render = path3d_world[:, perm]

    center_world = center_rot[inv_perm_arr]
    lookat_render = center_world

    if perm != (0, 1, 2):
        pts_world = np.asarray(pcd.points)[:, inv_perm_arr]
        pcd.points = o3d.utility.Vector3dVector(pts_world)

    path_render = [pt.copy() for pt in path_render]

    return pcd, path_render, lookat_render, perm