import os
from explorer import navigate
from path_planner import generate_trajectory
from renderer import render_frames


if __name__ == "__main__":
    ply_path = r"ConferenceHall_uncompressed - Cloud - Cloud.ply"
    out_video = r"outputs/scene_1/navigation_tour.mp4"

    os.makedirs(os.path.dirname(out_video), exist_ok=True)

    start = (32, -0.2, 29)
    goal  = (10, 15, 29)

    print("Строим путь по полу (A*)...")
    pcd, path3d, center = navigate(ply_path, start, goal)
    print(f"Найден путь из {len(path3d)} точек")

    print("Генерируем плавную траекторию...")
    traj, lookats = generate_trajectory(
        [(p, center) for p in path3d],
        center=center,
        total_frames=30 * 60,
    )

    print("Рендерим видео...")
    render_frames(
        ply_path,
        traj,
        lookats,
        out_video,
        fps=30,
        resolution=(1280, 720),
        visible=True
    )

    print("Готово!")
