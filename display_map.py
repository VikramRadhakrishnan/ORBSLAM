import argparse
import numpy as np
import open3d as o3d


FRUSTUM_SCALE = 0.3
FRUSTUM_SHAPE = np.array([
    [ 0.0,   0.0,  0.0],
    [-0.5, -0.375, 1.0],
    [ 0.5, -0.375, 1.0],
    [ 0.5,  0.375, 1.0],
    [-0.5,  0.375, 1.0],
]) * FRUSTUM_SCALE
FRUSTUM_EDGES = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]


def build_camera_lineset(poses):
    all_pts, all_lines, all_colors = [], [], []
    offset = 0
    for pose in poses:
        c2w = np.linalg.inv(pose)
        pts = (c2w[:3, :3] @ FRUSTUM_SHAPE.T).T + c2w[:3, 3]
        all_pts.extend(pts.tolist())
        all_lines.extend([[e[0] + offset, e[1] + offset] for e in FRUSTUM_EDGES])
        all_colors.extend([[0.0, 0.8, 0.0]] * len(FRUSTUM_EDGES))
        offset += 5

    for i in range(len(poses) - 1):
        all_lines.append([i * 5, (i + 1) * 5])
        all_colors.append([0.0, 0.5, 1.0])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines = o3d.utility.Vector2iVector(all_lines)
    ls.colors = o3d.utility.Vector3dVector(all_colors)
    return ls


def main():
    parser = argparse.ArgumentParser(description="Display a saved ORB-SLAM map.")
    parser.add_argument("--map-dir", default="map_output",
                        help="Directory containing map_points.ply and camera_poses.npy")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(f"{args.map_dir}/map_points.ply")
    poses = np.load(f"{args.map_dir}/camera_poses.npy")

    print(f"Loaded {len(pcd.points):,} map points and {len(poses)} camera poses.")

    cam_lines = build_camera_lineset(poses)

    o3d.visualization.draw_geometries(
        [pcd, cam_lines],
        window_name="ORB-SLAM Map",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
