import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from queue import PriorityQueue


# ------------------------------
# 1. Load PLY and detect floor
# ------------------------------

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def detect_floor_plane(pcd, distance_thresh=0.03):
    """
    RANSAC — ищем основную горизонтальную плоскость (пол)
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_thresh,
        ransac_n=3,
        num_iterations=500
    )

    [a, b, c, d] = plane_model

    # Проверяем что плоскость горизонтальная
    if abs(c) < 0.8:
        raise RuntimeError("Floor plane not horizontal")

    floor_points = pcd.select_by_index(inliers)
    floor_height = np.mean(np.asarray(floor_points.points)[:, 2])

    return floor_height


# ------------------------------
# 2. Create occupancy grid
# ------------------------------

def build_occupancy_grid(pcd, floor_z, res=0.05, person_height=1.6):
    pts = np.asarray(pcd.points)

    # 2D bounding box
    xmin, ymin = pts[:,0].min(), pts[:,1].min()
    xmax, ymax = pts[:,0].max(), pts[:,1].max()

    W = int((xmax - xmin) / res) + 1
    H = int((ymax - ymin) / res) + 1

    grid = np.zeros((W, H), dtype=np.uint8)

    for x, y, z in pts:
        gx = int((x - xmin) / res)
        gy = int((y - ymin) / res)

        # препятствие, если слишком высоко
        if z > floor_z + 0.3:
            grid[gx, gy] = 1

    return grid, xmin, ymin, res


# ------------------------------
# 3. A* pathfinding
# ------------------------------

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def astar(grid, start, goal):
    W, H = grid.shape
    pq = PriorityQueue()
    pq.put((0, start))

    came = {start: None}
    cost = {start: 0}

    dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while not pq.empty():
        _, cur = pq.get()

        if cur == goal:
            break

        for dx, dy in dirs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < W and 0 <= ny < H and grid[nx, ny] == 0:

                new_cost = cost[cur] + 1
                if (nx, ny) not in cost or new_cost < cost[(nx,ny)]:
                    cost[(nx,ny)] = new_cost
                    pq.put((new_cost + heuristic((nx,ny), goal), (nx,ny)))
                    came[(nx,ny)] = cur

    # reconstruct path
    if goal not in came:
        raise RuntimeError("No path found")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]

    return path[::-1]


# ------------------------------
# 4. Convert grid path to 3D
# ------------------------------

def grid_to_world(path, xmin, ymin, res, floor_z, cam_h=1.6):
    pts = []
    for gx, gy in path:
        x = xmin + gx * res
        y = ymin + gy * res
        z = floor_z + cam_h
        pts.append(np.array([x, y, z]))
    return pts

def auto_fix_axes(pcd):
    """
    Автоматически определяет какая ось является вертикальной.
    Выбирает ту, где минимальный размах (range) — это обычно высота комнаты.
    Затем переназначает оси так, чтобы вертикальная стала Z.
    """
    pts = np.asarray(pcd.points)
    ranges = pts.max(axis=0) - pts.min(axis=0)

    # Индекс оси с минимальным размахом = вертикаль
    vertical = np.argmin(ranges)

    if vertical == 2:
        return pcd  # уже правильно

    # Перестановка axes → так, чтобы vertical → Z
    if vertical == 0:
        rot = pts[:, [1, 2, 0]]
    elif vertical == 1:
        rot = pts[:, [0, 2, 1]]

    pcd.points = o3d.utility.Vector3dVector(rot)
    return pcd

# ------------------------------
# High-level function
# ------------------------------

def navigate(ply_path, start_xyz, goal_xyz):
    pcd = load_ply(ply_path)
    pcd = auto_fix_axes(pcd)
    center = np.asarray(pcd.points).mean(axis=0)

    # detect floor
    floor_z = detect_floor_plane(pcd)

    # grid
    grid, xmin, ymin, res = build_occupancy_grid(pcd, floor_z)

    # convert input coords to grid coords
    sx = int((start_xyz[0] - xmin) / res)
    sy = int((start_xyz[1] - ymin) / res)
    gx = int((goal_xyz[0] - xmin) / res)
    gy = int((goal_xyz[1] - ymin) / res)

    # find path
    path2d = astar(grid, (sx, sy), (gx, gy))

    # convert to 3D camera path
    path3d = grid_to_world(path2d, xmin, ymin, res, floor_z)

    return pcd, path3d, center
