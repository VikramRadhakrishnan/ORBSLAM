from multiprocessing import Process, Queue
import numpy as np
import cv2
import open3d as o3d


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
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='ORB-SLAM Map', width=w, height=h)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 2.0

        # Persistent point cloud geometry added once; updated in-place each frame
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # Camera frustum / trajectory geometries are replaced each frame
        self.cam_geometries = []

        # Camera feed shown in a separate OpenCV window
        img_w, img_h = 480, 270
        self.feed_image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', img_w, img_h)

    def _make_frustum(self, pose, scale=0.3):
        # pose is world-to-camera (4×4); invert to get camera-to-world
        cam_to_world = np.linalg.inv(pose)
        R = cam_to_world[:3, :3]
        t = cam_to_world[:3, 3]

        pts_cam = np.array([
            [ 0.0,   0.0,   0.0],
            [-0.5, -0.375,  1.0],
            [ 0.5, -0.375,  1.0],
            [ 0.5,  0.375,  1.0],
            [-0.5,  0.375,  1.0],
        ]) * scale
        pts_world = (R @ pts_cam.T).T + t

        lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                 [1, 2], [2, 3], [3, 4], [4, 1]]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_world)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0.0, 0.8, 0.0]] * len(lines))
        return ls

    def viewer_refresh(self, q):
        img_w, img_h = 480, 270

        if self.state is None or not q.empty():
            self.state = q.get()

        poses, pts = self.state

        # Update map point cloud (red points)
        if pts.ndim == 2 and len(pts) > 0:
            xyz = pts[:, :3]
            self.pcd.points = o3d.utility.Vector3dVector(xyz)
            self.pcd.colors = o3d.utility.Vector3dVector(
                np.tile([1.0, 0.0, 0.0], (len(xyz), 1))
            )
        self.vis.update_geometry(self.pcd)

        # Remove previous camera frustums and trajectory line
        for geom in self.cam_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.cam_geometries = []

        # Draw one frustum per camera pose (green)
        if poses.ndim == 3 and len(poses) > 0:
            for pose in poses:
                frustum = self._make_frustum(pose)
                self.vis.add_geometry(frustum, reset_bounding_box=False)
                self.cam_geometries.append(frustum)

            # Connect camera centers with a trajectory line
            if len(poses) > 1:
                centers = [np.linalg.inv(p)[:3, 3] for p in poses]
                lines = [[i, i + 1] for i in range(len(centers) - 1)]
                traj = o3d.geometry.LineSet()
                traj.points = o3d.utility.Vector3dVector(centers)
                traj.lines = o3d.utility.Vector2iVector(lines)
                traj.colors = o3d.utility.Vector3dVector(
                    [[0.0, 0.8, 0.0]] * len(lines)
                )
                self.vis.add_geometry(traj, reset_bounding_box=False)
                self.cam_geometries.append(traj)

        self.vis.poll_events()
        self.vis.update_renderer()

        # Update camera feed window
        if not self.q_image.empty():
            img = self.q_image.get()
            if img.ndim == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            self.feed_image = cv2.resize(img, (img_w, img_h))

        cv2.imshow('Camera Feed', self.feed_image)
        cv2.waitKey(1)

    def display(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))

    def display_image(self, ip_image):
        if self.q_image is None:
            return
        self.q_image.put(ip_image)


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
