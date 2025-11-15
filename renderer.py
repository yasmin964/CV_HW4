# renderer.py
import open3d as o3d
import numpy as np
import imageio
import os
from tqdm import tqdm

def render_frames(ply_path, trajectory_world, lookat_world, output_path, fps=30, resolution=(1280,720), visible=True):
    """
    trajectory_world: (N,3) np.array positions
    lookat_world: (N,3) np.array lookat points
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="render", width=resolution[0], height=resolution[1], visible=visible)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=1)

    for pos, lookat in tqdm(zip(trajectory_world, lookat_world), total=len(trajectory_world)):
        front = (lookat - pos)
        norm = np.linalg.norm(front)
        if norm == 0:
            front = np.array([0,0,1])
        else:
            front = front / norm
        # set camera orientation
        try:
            ctr.set_lookat(lookat.tolist())
            ctr.set_front(front.tolist())
            ctr.set_up([0,0,1])
            # optional: adjust zoom if needed
            ctr.set_zoom(0.8)
        except Exception:
            # fallback: try to set pinhole extrinsic if control methods fail
            pass

        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (img * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()
    vis.destroy_window()
    print("✅ Видео сохранено:", output_path)
