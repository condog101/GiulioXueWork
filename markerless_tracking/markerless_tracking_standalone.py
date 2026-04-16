#! /usr/bin/env python3
"""
Standalone markerless tracking script - No ROS dependencies.
Performs deep learning-based segmentation inference using Intel RealSense camera.

Usage:
    python markerless_tracking_standalone.py
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import sys
import os
import time
import torch

# Use relative imports for standalone operation
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

from utils.model_pytorch import AlexNet2, PointNetSeg, nms


# Helper function for safe OpenCV display (may not have GUI support)
def safe_imshow(name, img):
    try:
        cv2.imshow(name, img)
    except cv2.error:
        pass


def safe_waitkey(delay=1):
    try:
        return cv2.waitKey(delay) & 0xFF
    except cv2.error:
        return 255


class MarkerlessTracker:
    """Standalone markerless tracking using deep learning segmentation."""

    def __init__(self):
        """Initialize the tracker with hardcoded paths."""
        self.NUM_POINT = 2000

        # Hardcoded checkpoint paths
        self.rgb_ckpt_path = os.path.join(script_dir, 'ckpt', 'rgb', 'alexnet2.pt')
        self.pcl_ckpt_path = os.path.join(script_dir, 'ckpt', 'pcl', 'pointnet_seg.pt')

        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Build and load RGB network (AlexNet2 SSD detector)
        print("Loading RGB model...")
        self.rgb_model = AlexNet2()
        self.rgb_model.load_state_dict(
            torch.load(self.rgb_ckpt_path, map_location=self.device, weights_only=True))
        self.rgb_model.to(self.device)
        # Original TF code uses is_training=True for BN even at inference
        self.rgb_model.train()

        # Build and load PCL network (PointNet segmenter)
        print("Loading PCL model...")
        self.pcl_model = PointNetSeg(num_point=self.NUM_POINT)
        self.pcl_model.load_state_dict(
            torch.load(self.pcl_ckpt_path, map_location=self.device, weights_only=True))
        self.pcl_model.to(self.device)
        # Original TF code uses is_training=False (uses running BN stats)
        self.pcl_model.eval()

        print("Finished init")

    @torch.no_grad()
    def process_frame(self, rgb_raw, depth_raw):
        """
        Process a single frame from RealSense camera.

        Args:
            rgb_raw: RGB image as numpy array (H, W, 3)
            depth_raw: Depth/XYZ data as numpy array (H, W, 3) in meters
        """
        start = time.perf_counter()

        # Store original dimensions for coordinate mapping
        orig_h, orig_w = rgb_raw.shape[:2]

        # Resize to 512x512 for network input
        rgb = cv2.resize(rgb_raw, (512, 512))
        depth = cv2.resize(depth_raw, (512, 512))

        # Scale factors for mapping back to original coordinates
        scale_x = orig_w / 512.0
        scale_y = orig_h / 512.0

        # Reduce the RGB field of view and limit what the camera has to consider for the ROI
        rgb[np.sqrt(np.power(depth[:, :, 0], 2) + np.power(depth[:, :, 1],
                    2) + np.power(depth[:, :, 2], 2)) > 2] = np.array([0, 0, 0])

        # Filter depth
        depth[depth[:, :, 2] > 0.8] = np.array([0, 0, 0])

        # Run RGB network for bounding box detection
        rgb_tensor = torch.from_numpy(
            rgb.reshape(1, 512, 512, 3).astype(np.float32)
        ).permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW

        probs_t, preds_loc_t = self.rgb_model(rgb_tensor)
        probs_val = probs_t.cpu().numpy()
        preds_loc_val = preds_loc_t.cpu().numpy()

        preds_conf_val = (probs_val > 0.8).astype(int)
        y_pred_conf = preds_conf_val[0].astype('float32')
        y_pred_loc = preds_loc_val[0]
        prob = probs_val[0]

        # Perform NMS to get bounding box
        box = nms(y_pred_conf, y_pred_loc, prob, extend=1.5)

        try:
            box_coords = [int(round(x)) for x in box[:4]]
            cls = int(box[4])
            cls_prob = box[5]

            x_min = np.maximum(box_coords[0], 0)
            x_max = np.minimum(box_coords[2], 512)
            y_min = np.maximum(box_coords[1], 0)
            y_max = np.minimum(box_coords[3], 512)

            # Annotate image
            label_str = '%s %.2f' % ('knee', cls_prob)
            rgb = cv2.rectangle(rgb, tuple(box_coords[:2]), tuple(
                box_coords[2:]), (0, 0, 255))
            rgb = cv2.putText(
                rgb, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Crop ROI (from resized images)
            cropped = depth[y_min:y_max, x_min:x_max, :]
            cropped_rgb = rgb[y_min:y_max, x_min:x_max, :]
            h_crop, w_crop, _ = cropped.shape

            data = cropped.reshape((-1, cropped.shape[-1]))
            data_rgb = cropped_rgb.reshape((-1, cropped_rgb.shape[-1]))

            # Filter out zero points
            index0 = np.where(~np.all(data == 0, axis=1))[0]
            data = data[index0, :]
            data_rgb = data_rgb[index0, :]

            # Sample points for network input
            index1 = np.random.choice(range(data.shape[0]), self.NUM_POINT)
            cropped_linear = data[index1, :].reshape((1, self.NUM_POINT, 3))
            cropped_linear_rgb = data_rgb[index1,
                                          :].reshape((1, self.NUM_POINT, 3))

        except Exception as e:
            print(e)
            print("An exception was thrown!")
            return

        # Run segmentation network
        thresh = 0.5
        pcl_tensor = torch.from_numpy(
            cropped_linear.astype(np.float32)
        ).to(self.device)

        pred_val = self.pcl_model(pcl_tensor).cpu().numpy()  # [1, N]
        index2 = np.where(pred_val[0] > thresh)[0]

        if len(index2) > 10:
            index = [index1[i] for i in index2]
            index = [index0[i] for i in index]

            row_col = [[i // w_crop, i % w_crop] for i in index]
            for pair in row_col:
                row = pair[0] + y_min
                col = pair[1] + x_min
                rgb[row, col] = np.array([0, 0, 255])

        # Extract segmented points
        femur = cropped_linear.squeeze()[pred_val[0] > thresh]

        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        safe_imshow('Segmentation', bgr)
        safe_waitkey(1)

        if femur.shape[0] < 140:
            print('rejected! (%d points)' % femur.shape[0])
        else:
            print('accepted! (%d points)' % femur.shape[0])

        elapsed = (time.perf_counter() - start) * 1000
        print(f"Processing time: {elapsed:.1f} ms")

    def close(self):
        """Clean up resources."""
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # OpenCV may not have GUI support


class RealSenseCamera:
    """Intel RealSense camera wrapper for RGB-D streaming."""

    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense camera.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps

        # Configure streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable depth and color streams
        self.config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.rgb8, fps)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")

        # Create align object to align depth to color frame
        self.align = rs.align(rs.stream.color)

        # Get camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print(f"Camera intrinsics: fx={self.intrinsics.fx}, fy={self.intrinsics.fy}, "
              f"cx={self.intrinsics.ppx}, cy={self.intrinsics.ppy}")

        # Warm up camera
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("Camera ready!")

    def get_frame(self):
        """
        Get aligned RGB and depth frames.

        Returns:
            rgb: RGB image as numpy array (H, W, 3)
            depth_xyz: XYZ point cloud as numpy array (H, W, 3) in meters
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert to numpy arrays
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # Convert depth to XYZ point cloud
        depth_xyz = self._depth_to_xyz(depth)

        return rgb, depth_xyz

    def _depth_to_xyz(self, depth):
        """
        Convert depth image to XYZ point cloud using camera intrinsics.

        Args:
            depth: 2D depth image (H, W) in raw units

        Returns:
            xyz: XYZ coordinates (H, W, 3) in meters
        """
        height, width = depth.shape
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.ppx
        cy = self.intrinsics.ppy

        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # Convert depth to meters
        z = depth.astype(np.float32) * self.depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        xyz = np.stack([x, y, z], axis=-1)
        return xyz

    def stop(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()


def main():
    print("Starting Markerless Tracking (Standalone)")
    print("=========================================")

    # Initialize camera
    print("Initializing Intel RealSense camera...")
    camera = RealSenseCamera(width=640, height=480, fps=30)

    # Initialize tracker
    print("Initializing tracker...")
    tracker = MarkerlessTracker()

    print("Beginning of monitoring")
    print("Press 'q' or ESC to quit (or Ctrl+C)")

    try:
        while True:
            # Get frame from camera
            rgb, depth_xyz = camera.get_frame()

            if rgb is None or depth_xyz is None:
                continue

            # Process frame
            tracker.process_frame(rgb, depth_xyz)

            # Check for quit
            key = safe_waitkey(1)
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Quit requested")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    finally:
        print("End of monitoring")
        camera.stop()
        tracker.close()
        print("Cleanup complete")


if __name__ == '__main__':
    main()
