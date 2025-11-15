import open3d as o3d
import numpy as np
import imageio

# Загружаем облако точек
scene = o3d.io.read_point_cloud("ConferenceHall_uncompressed.ply")

# Создаём окно визуализации
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720, visible=True)
vis.add_geometry(scene)

ctr = vis.get_view_control()

# Траектория движения камеры
path = [
    np.array([0, 0, 1]),
    np.array([1, 0, 1.2]),
    np.array([2, 1, 1.5]),
    np.array([3, 2, 1.2]),
]

frames = []

# Двигаем камеру и сохраняем кадры
for point in path:
    ctr.translate(point[0], point[1], point[2])
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer()
    frames.append((np.asarray(img) * 255).astype(np.uint8))

# После цикла закрываем окно
vis.destroy_window()

# Сохраняем видео
imageio.mimsave("panorama_tour.mp4", frames, fps=10)
