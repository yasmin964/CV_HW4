
## Installation

### Requirements

* Python 3.10
* Modern web browser with WebGL support
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

### 1. Generate Camera Path and Render Video

```bash
python main.py
```

This will:

1. Load the `.ply` point cloud
2. Detect the floor plane and analyze scene structure
3. Build an occupancy grid and find optimal start/end points
4. Generate a smooth path using A* search with obstacle avoidance
5. Create smooth camera trajectory using spline interpolation
6. **Save camera path** to `spark_viewer/camera_path.json`
7. Render cinematic video using Spark distributed rendering

### 2. Real-time WebGL Viewer

Open `spark_viewer/index.html` 
```bash
 python3 -m http.server 8000
```

in a modern web browser to:

- View the 3D scene in real-time using Gaussian Splatting
- See the camera automatically follow the generated path
- **Record a video** directly from the WebGL renderer (automatically starts)
- Download the recorded tour as MP4/WebM video

## Output Files

- **`spark_viewer/camera_path.json`**: Camera trajectory for WebGL viewer
- **`outputs/scene_1/spark_tour.mp4`**: High-quality rendered video (Python)
- **`outputs/scene_1/camera_tour.mp4/`**: Browser-recorded video (WebGL)

## Key Features

### Dual Rendering System

1. **Python/Spark Rendering** (`main.py` + `renderer.py`)
   - High-quality video output
   - Distributed rendering with Apache Spark
   - Collision-free path planning
   - Professional video encoding

2. **WebGL Real-time Viewer** (`spark_viewer/index.html`)
   - Real-time Gaussian Splatting rendering
   - Built-in video recording
   - Optimized performance (30 FPS cap, memory management)
   - Direct browser download

### Smart Path Planning

- **Automatic start/end point selection** in largest free space area
- **Obstacle avoidance** using occupancy grid analysis
- **Smooth trajectories** with cubic spline interpolation
- **Cinematic camera movement** with look-ahead targeting

### Performance Optimizations

- **Spark distributed rendering** for faster video generation
- **WebGL optimizations**: limited splat count, disabled antialiasing
- **Smart FPS control** to maintain smooth recording
- **Memory management** with automatic cleanup

---

## Algorithm Descriptions

### Scene Analysis (`explorer.py`)

- **Floor Detection**: RANSAC plane fitting to detect horizontal surfaces
- **Occupancy Grid**: 2D grid mapping with obstacle detection
- **Free Space Analysis**: Connected component analysis to find navigable areas

### Path Planning (`path_planner.py`)

- **A* Search**: Optimal path finding on occupancy grid
- **Path Pruning**: Remove redundant waypoints
- **Spline Smoothing**: Cubic spline interpolation for smooth camera motion
- **Look-ahead Targeting**: Camera naturally follows path direction

### Rendering Systems 

#### Python/Spark Rendering (`renderer.py`)
- **Distributed Processing**: Parallel frame rendering across multiple cores
- **High Quality**: 1280×720 resolution, professional encoding
- **Collision Detection**: Camera stays within navigable space

#### WebGL Real-time Rendering (`index.html`)
- **Gaussian Splatting**: Real-time 3D point cloud rendering
- **Automated Recording**: Built-in MediaRecorder API
- **Performance Optimized**: 30 FPS target, memory efficient

---

## Known Limitations

1. **WebGL Recording**: Browser recording may have lower quality than Python rendering
2. **Memory Usage**: Large point clouds may impact browser performance
3. **Floor Detection**: Assumes mostly horizontal floor surfaces
4. **Browser Support**: Requires modern browser with MediaRecorder API
5. **Spark Setup**: Distributed rendering requires proper Spark configuration
