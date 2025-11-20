import open3d as o3d
import numpy as np
import imageio
import os
from tqdm import tqdm


def _compute_camera_pose(eye, target, world_up=None):
    """Return a 4x4 extrinsic matrix that positions the camera at `eye` looking at `target`."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        world_up = np.asarray(world_up, dtype=np.float64)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward /= forward_norm

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward and up are collinear; choose new up vector
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, alt_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= right_norm
    else:
        right /= right_norm

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    R = np.stack((right, -up, forward), axis=0)
    t = -R @ eye

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic


def render_frames(ply_path, trajectory_world, lookat_world, output_path, fps=30, resolution=(1280, 720), visible=True,
                  perm=None, flip_camera=False):
    """
    trajectory_world: (N,3) np.array positions
    lookat_world: (N,3) np.array lookat points
    perm: optional, порядок перестановки точек
    flip_camera: если True, переворачивает камеру на 180°
    """
    if len(trajectory_world) != len(lookat_world):
        raise ValueError("trajectory_world and lookat_world must have the same length")

    pcd = o3d.io.read_point_cloud(ply_path)

    # Если передали perm, применяем его к точкам
    if perm is not None:
        pts = np.asarray(pcd.points)
        pts = pts[:, perm]  # перестановка
        pcd.points = o3d.utility.Vector3dVector(pts)
        print(f"Applied permutation {perm} to point cloud")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="render", width=resolution[0], height=resolution[1], visible=visible)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=1)

    for pos, lookat in tqdm(zip(trajectory_world, lookat_world), total=len(trajectory_world)):
        extrinsic = _compute_camera_pose(pos, lookat)

        # ДОБАВЛЯЕМ РОТАЦИЮ КАМЕРЫ НА 180°
        if flip_camera:
            # Матрица ротации на 180° вокруг оси Z (вертикальной)
            flip_rotation = np.array([
                [-1, 0, 0, 0],  # инвертируем X
                [0, -1, 0, 0],  # инвертируем Y
                [0, 0, 1, 0],  # Z остается
                [0, 0, 0, 1]
            ])
            extrinsic = flip_rotation @ extrinsic

        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (img * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()
    vis.destroy_window()
    print("Видео сохранено:", output_path)