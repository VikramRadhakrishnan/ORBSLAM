## Monocular Visual SLAM in Python

A Python implementation of ORB-SLAM with real-time 3D visualization powered by Open3D.

---

## Dependencies

Install Python dependencies via pip:

```bash
pip install opencv-python numpy open3d scikit-image g2o-python
```

---

## How to run

**Run SLAM on a video:**

```bash
python3 main.py
```

The script reads `videos/car.mp4` by default. While running, two windows are shown:
- **ORB-SLAM Map** — interactive 3D view of the reconstructed point cloud, camera frustums (green), and trajectory (blue)
- **Camera Feed** — current video frame with tracked feature points

When the video ends the map is saved automatically to `map_output/`:
- `map_points.ply` — 3D point cloud (loadable in MeshLab, CloudCompare, Open3D, etc.)
- `camera_poses.npy` — NumPy array of shape `(N, 4, 4)` with each camera's pose matrix

**Display a saved map:**

```bash
python3 display_map.py                      # loads map_output/ by default
python3 display_map.py --map-dir <path>     # load from a custom directory
```

Mouse controls in both 3D windows: left-drag to rotate, scroll to zoom, ctrl+drag to pan.

---

## Code structure

```
├── main.py           # Entry point: reads video, runs SLAM pipeline, saves map
├── extractor.py      # ORB feature extraction, matching, and pose estimation
├── pointmap.py       # Map class: 3D viewer (Open3D), point/frame management, map save
├── display_map.py    # Standalone script to visualize a previously saved map
├── display.py        # 2D SDL2 display (unused, kept for reference)
├── utils.py          # Calibration file parsing helpers
└── notebooks/
    ├── SLAM_pipeline_step_by_step.ipynb   # Full pipeline walkthrough
    ├── mapping.ipynb                      # Structure-from-Motion reference
    └── bundle_adjustment.ipynb            # g2o / bundle adjustment reference
```

---

## Notebooks

- `SLAM_pipeline_step_by_step.ipynb` — walks through the entire monocular SLAM pipeline step by step. Uses the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale, 22 GB).
- `mapping.ipynb` — additional Structure-from-Motion reference ([source](https://github.com/SiddhantNadkarni/Parallel_SFM))
- `bundle_adjustment.ipynb` — g2o and bundle adjustment walkthrough ([source](https://github.com/maxcrous/multiview_notebooks))
