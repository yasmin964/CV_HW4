## Installation

### Requirements

* Python 3.10
* Dependencies (can be installed via `requirements.txt`):

```text
open3d~=0.19.0
numpy~=2.2.6
ImageIO~=2.37.2
scipy~=1.16.3
tqdm~=4.67.1
imageio-ffmpeg~=0.6.0
pyspark~=4.0.1
```

### Installation Steps

1. Create a Python 3.10 virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install `ffmpeg` if not available on your system, required for video rendering:

```bash
brew install ffmpeg      # macOS
sudo apt install ffmpeg  # Ubuntu
```

---

## Usage

### 1. Run the main pipeline

```bash
python main.py
```

This will:

1. Load the `.ply` point cloud.
2. Detect the floor plane.
3. Build an occupancy grid.
4. Automatically select start and goal positions.
5. Generate a smooth path using A* search.
6. Smooth the path with spline interpolation.
7. Render a cinematic video using either Spark distributed or sequential rendering.

### 2. Example Output

* Input: `ConferenceHall_uncompressed.ply`
* Output: `outputs/scene_1/spark_tour.mp4`

### 3. Custom Parameters

You can modify:

* Video resolution: `resolution=(1280, 720)`
* Frames per second: `fps=30`
* Enable/disable Spark distributed rendering: `use_spark=True/False`
* Camera flip or permutation: `flip_camera=True/False`, `perm=(0,1,2)`

---

## Algorithm Descriptions

### Floor Detection

* Uses RANSAC plane fitting to detect the horizontal floor.
* Validates horizontal alignment via the normal vector.

### Occupancy Grid

* Converts 3D points into a 2D grid.
* Marks free and blocked cells based on floor support and obstacle height thresholds.
* Optional morphological operations smooth the walkable area.

### Path Planning

* **A* search**: finds an initial 2D path on the occupancy grid.
* **Path pruning**: removes redundant intermediate points.
* **Smoothing**: Laplacian smoothing followed by cubic spline interpolation in 3D space.

### Camera Trajectory

* Generates smooth camera positions and look-at points.
* Uses cubic spline interpolation along the path.
* Look-at points are slightly ahead along the path for cinematic effect.

### Rendering

* Sequential rendering using Open3D.
* Optional distributed rendering using Apache Spark.
* Video is encoded in MP4 (`libx264`) with high-quality settings.

---

## Dependencies and Requirements

* **Python**: 3.10
* **Libraries**: `open3d`, `numpy`, `ImageIO`, `scipy`, `tqdm`, `imageio-ffmpeg`, `pyspark`
* **System**: `ffmpeg` installed for MP4 video creation
* **Optional**: Apache Spark for distributed rendering

---

## Known Limitations

1. Floor plane detection assumes a mostly horizontal floor. Sloped floors may fail detection.
2. Path planning is 2D-based; multilevel navigation is not supported.
3. Memory usage may grow for very large point clouds.
4. Spark distributed rendering requires proper configuration and may fallback to sequential rendering if Spark fails.
5. Camera collision avoidance with 3D obstacles is limited to vertical clearance checks.
6. Currently, the pipeline supports `.ply` point clouds only.
