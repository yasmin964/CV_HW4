import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from queue import PriorityQueue


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


def build_occupancy_grid(pcd, floor_z, res=0.05, person_height=1.6):
    pts = np.asarray(pcd.points)

    xmin, ymin = pts[:,0].min(), pts[:,1].min()
    xmax, ymax = pts[:,0].max(), pts[:,1].max()

    W = int((xmax - xmin) / res) + 1
    H = int((ymax - ymin) / res) + 1

    grid = np.zeros((W, H), dtype=np.uint8)

    for x, y, z in pts:
        gx = int((x - xmin) / res)
        gy = int((y - ymin) / res)

        if z > floor_z + 0.3:
            grid[gx, gy] = 1

    return grid, xmin, ymin, res


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

    if goal not in came:
        raise RuntimeError("No path found")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]

    return path[::-1]

def grid_to_world(path, xmin, ymin, res, floor_z, cam_h=1.6):
    pts = []
    for gx, gy in path:
        x = xmin + gx * res
        y = ymin + gy * res
        z = floor_z + cam_h + 0.05*np.sin(gx*0.1 + gy*0.1)
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

    pts = np.asarray(pcd.points)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    print("Диапазоны по осям X, Y, Z:", ranges)

    center_rot = pts.mean(axis=0)

    # detect floor
    floor_z = detect_floor_plane(pcd)

    # grid
    grid, xmin, ymin, res = build_occupancy_grid(pcd, floor_z)

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

    # find path
    path2d = astar(grid, (sx, sy), (gx, gy))

    # convert to 3D camera path
    path3d_rot = np.stack(grid_to_world(path2d, xmin, ymin, res, floor_z))
    path3d_world = path3d_rot[:, inv_perm_arr]

    center_world = center_rot[inv_perm_arr]

    if perm != (0, 1, 2):
        pts_world = np.asarray(pcd.points)[:, inv_perm_arr]
        pcd.points = o3d.utility.Vector3dVector(pts_world)

    path3d_world = [pt.copy() for pt in path3d_world]

    return pcd, path3d_world, center_world
