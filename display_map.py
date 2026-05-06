import argparse
import numpy as np
import cv2
import open3d as o3d
from pointmap import generate_occupancy_grid


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
    pts = np.asarray(pcd.points)

    print(f"Loaded {len(pts):,} map points and {len(poses)} camera poses.")

    cam_lines = build_camera_lineset(poses)

    # 3-D viewer (non-blocking so the occupancy grid window stays alive alongside it)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='ORB-SLAM Map', width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(cam_lines)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0

    # 2-D occupancy grid
    occ = generate_occupancy_grid(pts, poses)
    cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Occupancy Grid', 600, 600)
    cv2.imshow('Occupancy Grid', occ)

    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vis.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
