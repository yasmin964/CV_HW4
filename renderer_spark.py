import open3d as o3d
import numpy as np
import imageio
import os
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import tempfile
import subprocess
import warnings
import base64
import pickle

# Ignore Open3D warnings
warnings.filterwarnings('ignore')


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


def render_single_frame_spark(args):
    """Render single frame - executed on Spark worker nodes"""
    try:
        frame_idx, pos, lookat, ply_path, resolution, perm, flip_camera = args

        # Load point cloud on worker
        pcd = o3d.io.read_point_cloud(ply_path)

        # Apply permutation if needed
        if perm is not None:
            pts = np.asarray(pcd.points)
            pts = pts[:, perm]
            pcd.points = o3d.utility.Vector3dVector(pts)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=resolution[0],
            height=resolution[1],
            visible=False,
            offscreen=True
        )
        vis.add_geometry(pcd)

        # Configure render options
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.15])
        render_opt.point_size = 3.0
        render_opt.light_on = True

        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()

        # Compute camera pose
        extrinsic = _compute_camera_pose(pos, lookat)

        # Apply camera flip if needed
        if flip_camera:
            flip_rotation = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            extrinsic = flip_rotation @ extrinsic

        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

        # Render frame
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Close visualizer
        vis.destroy_window()

        # Serialize image for Spark transmission
        # Use base64 for reliable binary data transmission
        img_serialized = base64.b64encode(pickle.dumps(img)).decode('utf-8')

        return (frame_idx, img_serialized)

    except Exception as e:
        print(f"Error rendering frame {frame_idx} on worker: {e}")
        # Return black frame in case of error
        black_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        img_serialized = base64.b64encode(pickle.dumps(black_frame)).decode('utf-8')
        return (frame_idx, img_serialized)


def init_spark(app_name="3D_Rendering", master="local[*]", memory="2g"):
    """Initialize Spark session optimized for rendering"""
    conf = SparkConf().setAppName(app_name) \
        .setMaster(master) \
        .set("spark.executor.memory", memory) \
        .set("spark.driver.memory", memory) \
        .set("spark.sql.adaptive.enabled", "true") \
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .set("spark.kryoserializer.buffer.max", "512m") \
        .set("spark.driver.maxResultSize", "2g")

    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Reduce logging
    sc.setLogLevel("WARN")

    return spark, sc


def render_frames_spark(ply_path, trajectory_world, lookat_world, output_path,
                        fps=30, resolution=(1280, 720),
                        perm=None, flip_camera=False, spark_master="local[*]"):
    """
    Distributed rendering using Apache Spark
    """
    if len(trajectory_world) != len(lookat_world):
        raise ValueError("trajectory_world and lookat_world must have the same length")

    print("=== SPARK DISTRIBUTED RENDERING ===")
    print(f"Rendering {len(trajectory_world)} frames using Spark")
    print(f"Spark Master: {spark_master}")

    # Initialize Spark
    spark, sc = init_spark(master=spark_master)

    try:
        # Check PLY file availability
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Prepare data for distributed processing
        frame_data = []
        for i, (pos, lookat) in enumerate(zip(trajectory_world, lookat_world)):
            frame_data.append((i, pos, lookat, ply_path, resolution, perm, flip_camera))

        # Determine optimal number of partitions
        num_partitions = min(sc.defaultParallelism, len(frame_data))
        print(f"Using {num_partitions} partitions for {len(frame_data)} frames")

        # Create RDD and start distributed rendering
        frames_rdd = sc.parallelize(frame_data, numSlices=num_partitions)

        # Render frames with progress tracking
        print("Starting distributed rendering across Spark workers...")

        # Use map for rendering and collect results
        rendered_frames_serialized = frames_rdd.map(render_single_frame_spark).collect()

        # Deserialize and sort frames by index
        rendered_frames = []
        for frame_idx, img_serialized in rendered_frames_serialized:
            img = pickle.loads(base64.b64decode(img_serialized))
            rendered_frames.append((frame_idx, img))

        rendered_frames.sort(key=lambda x: x[0])
        frames = [frame for _, frame in rendered_frames]

        print(f"Successfully rendered {len(frames)} frames")

        # Create video from frames
        print("Creating MP4 video from rendered frames...")
        create_video_from_frames(frames, output_path, fps)

        print("Video saved with Spark rendering:", output_path)

    except Exception as e:
        print(f"Spark rendering failed: {e}")
        raise
    finally:
        # Always stop Spark
        spark.stop()
        print("Spark session stopped")


def create_video_from_frames(frames, output_path, fps=30):
    """Create MP4 video from frame array"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use imageio for high-quality video creation
    with imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
    ) as writer:
        for frame in tqdm(frames, desc="Encoding video"):
            writer.append_data(frame)


def render_frames_sequential(ply_path, trajectory_world, lookat_world, output_path,
                             fps=30, resolution=(1280, 720), visible=True,
                             perm=None, flip_camera=False):
    """
    Sequential rendering (fallback option)
    """
    if len(trajectory_world) != len(lookat_world):
        raise ValueError("trajectory_world and lookat_world must have the same length")

    pcd = o3d.io.read_point_cloud(ply_path)

    if perm is not None:
        pts = np.asarray(pcd.points)
        pts = pts[:, perm]
        pcd.points = o3d.utility.Vector3dVector(pts)
        print(f"Applied permutation {perm} to point cloud")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="render", width=resolution[0], height=resolution[1], visible=visible)
    vis.add_geometry(pcd)

    # Improve rendering settings
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.1, 0.1, 0.15])
    render_opt.point_size = 3.0
    render_opt.light_on = True

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, pixelformat='yuv420p') as writer:

        for pos, lookat in tqdm(zip(trajectory_world, lookat_world),
                                total=len(trajectory_world),
                                desc="Rendering frames sequentially"):
            extrinsic = _compute_camera_pose(pos, lookat)

            if flip_camera:
                flip_rotation = np.array([
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                extrinsic = flip_rotation @ extrinsic

            camera_params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()
            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            writer.append_data(img)

    vis.destroy_window()
    print("Video saved with sequential rendering:", output_path)


def render_frames(ply_path, trajectory_world, lookat_world, output_path,
                  fps=30, resolution=(1280, 720), visible=True,
                  perm=None, flip_camera=False, use_spark=True, spark_master="local[*]"):
    """
    Main rendering function with Spark support

    Args:
        ply_path: path to PLY file
        trajectory_world: list of camera positions
        lookat_world: list of camera lookat points
        output_path: output video path
        fps: frames per second
        resolution: video resolution
        visible: show rendering window (sequential only)
        perm: coordinate permutation
        flip_camera: invert camera
        use_spark: use Spark for distributed rendering
        spark_master: Spark master URL (local[*], yarn, spark://host:port)
    """
    if use_spark:
        try:
            render_frames_spark(
                ply_path, trajectory_world, lookat_world, output_path,
                fps, resolution, perm, flip_camera, spark_master
            )
        except Exception as e:
            print(f"Spark rendering failed: {e}")
            print("Falling back to sequential rendering...")
            render_frames_sequential(
                ply_path, trajectory_world, lookat_world, output_path,
                fps, resolution, visible, perm, flip_camera
            )
    else:
        render_frames_sequential(
            ply_path, trajectory_world, lookat_world, output_path,
            fps, resolution, visible, perm, flip_camera
        )