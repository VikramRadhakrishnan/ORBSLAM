from multiprocessing import Process, Queue
import numpy as np
import cv2
import open3d as o3d
import os


def generate_occupancy_grid(pts, poses, grid_size=600):
    """Project 3D map points onto the XZ plane (bird's-eye view) and return a BGR image.

    Occupied cells are dark gray. The camera trajectory is drawn in blue,
    with a green circle at the start and a red circle at the current position.
    """
    img = np.full((grid_size, grid_size, 3), 240, dtype=np.uint8)

    if len(pts) == 0:
        return img

    xs, zs = pts[:, 0], pts[:, 2]
    cx = (xs.min() + xs.max()) / 2
    cz = (zs.min() + zs.max()) / 2
    extent = max(xs.max() - xs.min(), zs.max() - zs.min(), 1.0)
    scale = (grid_size * 0.8) / extent  # pixels per world unit

    # Mark occupied cells (fully vectorised)
    px = np.clip(((xs - cx) * scale + grid_size / 2).astype(np.int32), 0, grid_size - 1)
    pz = np.clip(((zs - cz) * scale + grid_size / 2).astype(np.int32), 0, grid_size - 1)
    img[pz, px] = [60, 60, 60]

    if len(poses) > 0:
        centers = [np.linalg.inv(p)[:3, 3] for p in poses]
        cpx = [int(np.clip((c[0] - cx) * scale + grid_size / 2, 0, grid_size - 1)) for c in centers]
        cpz = [int(np.clip((c[2] - cz) * scale + grid_size / 2, 0, grid_size - 1)) for c in centers]

        for i in range(len(centers) - 1):
            cv2.line(img, (cpx[i], cpz[i]), (cpx[i + 1], cpz[i + 1]), (200, 100, 0), 2)

        cv2.circle(img, (cpx[0],  cpz[0]),  5, (0, 180,   0), -1)  # green: start
        cv2.circle(img, (cpx[-1], cpz[-1]), 6, (0,   0, 220), -1)  # red:   current pose

    cv2.putText(img, f'{len(pts):,} pts  {len(poses)} frames',
                (8, grid_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    return img


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None
        self.q_image = None

    def create_viewer(self):
        self.q = Queue()
        self.q_image = Queue()
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init(1280, 720)
        while self.viewer_refresh(q):
            pass

    def viewer_init(self, w, h):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='ORB-SLAM Map', width=w, height=h)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 2.0

        # Point cloud: added once, updated in-place via update_geometry
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # All camera frustums + trajectory packed into one LineSet.
        # Removed and re-added only when new state arrives (not every frame).
        self.cam_lines = o3d.geometry.LineSet()
        self.cam_lines_in_scene = False

        img_w, img_h = 480, 270
        self.feed_image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', img_w, img_h)

        cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Occupancy Grid', 600, 600)
        cv2.imshow('Occupancy Grid', np.full((600, 600, 3), 240, dtype=np.uint8))

    def _build_camera_lines(self, poses):
        """Pack all camera frustums and the trajectory into a single LineSet."""
        all_pts, all_lines, all_colors = [], [], []
        offset = 0

        pts_cam_base = np.array([
            [ 0.0,   0.0,  0.0],
            [-0.5, -0.375, 1.0],
            [ 0.5, -0.375, 1.0],
            [ 0.5,  0.375, 1.0],
            [-0.5,  0.375, 1.0],
        ]) * 0.3
        frustum_edges = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

        for pose in poses:
            c2w = np.linalg.inv(pose)
            pts_world = (c2w[:3, :3] @ pts_cam_base.T).T + c2w[:3, 3]
            all_pts.extend(pts_world.tolist())
            all_lines.extend([[e[0] + offset, e[1] + offset] for e in frustum_edges])
            all_colors.extend([[0.0, 0.8, 0.0]] * len(frustum_edges))
            offset += 5

        # Trajectory: connect successive camera centres (vertex 0 of each frustum)
        for i in range(len(poses) - 1):
            all_lines.append([i * 5, (i + 1) * 5])
            all_colors.append([0.0, 0.5, 1.0])

        self.cam_lines.points = o3d.utility.Vector3dVector(all_pts)
        self.cam_lines.lines = o3d.utility.Vector2iVector(all_lines)
        self.cam_lines.colors = o3d.utility.Vector3dVector(all_colors)

    def _update_geometries(self):
        poses, pts = self.state

        # Update point cloud in-place
        if len(pts) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
            self.pcd.colors = o3d.utility.Vector3dVector(
                np.tile([1.0, 0.0, 0.0], (len(pts), 1))
            )
        self.vis.update_geometry(self.pcd)

        # Swap in the new camera LineSet (remove old, build new, add)
        if self.cam_lines_in_scene:
            self.vis.remove_geometry(self.cam_lines, reset_bounding_box=False)
            self.cam_lines_in_scene = False

        if len(poses) > 0:
            self._build_camera_lines(poses)
            self.vis.add_geometry(self.cam_lines, reset_bounding_box=False)
            self.cam_lines_in_scene = True

        cv2.imshow('Occupancy Grid', generate_occupancy_grid(pts, poses))

    def viewer_refresh(self, q):
        img_w, img_h = 480, 270

        # Pull new state only when available; otherwise keep rendering the last one
        if self.state is None or not q.empty():
            self.state = q.get()
            if self.state is None:      # shutdown sentinel
                self.vis.destroy_window()
                cv2.destroyAllWindows()
                return False
            self._update_geometries()

        # Keep the window responsive every iteration regardless of new data
        if not self.vis.poll_events():  # returns False when window is closed
            return False
        self.vis.update_renderer()

        if not self.q_image.empty():
            img = self.q_image.get()
            if img.ndim == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            self.feed_image = cv2.resize(img, (img_w, img_h))

        cv2.imshow('Camera Feed', self.feed_image)
        cv2.waitKey(1)
        return True

    def display(self):
        if self.q is None:
            return
        poses = np.array([f.pose for f in self.frames]) if self.frames else np.empty((0, 4, 4))
        pts   = np.array([p.pt  for p in self.points])  if self.points else np.empty((0, 4))
        self.q.put((poses, pts))

    def display_image(self, ip_image):
        if self.q_image is None:
            return
        self.q_image.put(ip_image)

    def stop_viewer(self):
        """Send shutdown sentinel and release queue threads so the process exits cleanly."""
        if self.q is not None:
            self.q.put(None)
            self.q.cancel_join_thread()
        if self.q_image is not None:
            self.q_image.cancel_join_thread()

    def save_map(self, output_dir="map_output"):
        os.makedirs(output_dir, exist_ok=True)

        if self.points:
            xyz = np.array([p.pt[:3] for p in self.points])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile([1.0, 0.0, 0.0], (len(xyz), 1))
            )
            ply_path = os.path.join(output_dir, "map_points.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"Point cloud saved: {ply_path}  ({len(xyz)} points)")

        if self.frames:
            poses = np.array([f.pose for f in self.frames])
            poses_path = os.path.join(output_dir, "camera_poses.npy")
            np.save(poses_path, poses)
            print(f"Camera poses saved: {poses_path}  ({len(poses)} frames)")


class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames

    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
