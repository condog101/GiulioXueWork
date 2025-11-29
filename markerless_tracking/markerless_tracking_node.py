#! /usr/bin/env python3
import utils.registration as reg
import utils.model as model
import math
import time

import numpy as np
import cv2
import pcl

import threading

import rclpy
from rclpy.node import Node
import ros2_numpy

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

import open3d as o3d
from scipy.spatial.transform import Rotation as spRot
import copy

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_registration(target):
    target_temp = copy.deepcopy(target)
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([target_temp])


def Test():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcd_data = o3d.data.DemoICPPointClouds()
    source_raw = o3d.io.read_point_cloud(pcd_data.paths[0])
    target_raw = o3d.io.read_point_cloud(pcd_data.paths[1])

    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)
    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

# From https://www.programcreek.com/python/example/99841/sensor_msgs.msg.PointCloud2


def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=False):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data


class pcCallback(Node):
    def __init__(self):
        super().__init__('markerless')
        self.pc_sub = self.create_subscription(
            PointCloud2, "/kinect/pc", self.callback, 1)
        self.pose_pub = self.create_publisher(
            PoseStamped, "/markerless/pose", 1)
        self.path = '/home/oceane/dev_ws/Experiments/Experiment1/'
        self.NUM_POINT = 2000

        self.femur_cl = o3d.io.read_point_cloud(
            "/data/catkin_ws_iros/models/knee_model_icp.ply")
        # Convert the model from m to ''
        femur_cl = np.asarray(self.femur_cl.points)
        self.femur_cl.points = o3d.utility.Vector3dVector(femur_cl * 39.3701)
        self.femur_cl.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        self.femur_cl.orient_normals_to_align_with_direction([0.0, 0.0, -1.0])

        # o3d.visualization.draw_geometries([self.femur_cl], point_show_normal=True)

        # draw_registration(self.femur_cl)

        '''----------- tensorflow graph --------------'''
        with tf.device('/gpu:0'):
            # with tf.device('/device:CPU:0'):
            '''new network'''
            # newrgb
            # self.RGB = tf.placeholder(tf.float32, [None, 360, 640, 3], name='x')
            self.RGB = tf.placeholder(
                tf.float32, [None, 512, 512, 3], name='x')
            self.probs, self.preds_loc = model.AlexNet2(
                self.RGB, scope='newroi')
            # newseg
            self.PCL = tf.placeholder(tf.float32, shape=(1, self.NUM_POINT, 3))
            self.prediction = model.get_model(self.PCL, scope='newseg')

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        self.sess = tf.Session(config=config)

        # Restore variables from disk.
        variables_to_restore = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='newroi')
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, tf.train.latest_checkpoint(
            '/home/oceane/dev_ws/src/markerless_tracking/markerless_tracking/ckpt/rgb'))

        variables_to_restore = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='newseg')
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, tf.train.latest_checkpoint(
            '/home/oceane/dev_ws/src/markerless_tracking/markerless_tracking/ckpt/pcl'))

        self.buffer_idx = 0
        self.buffer_size = 50

        self.origin_buffer = np.zeros((3, self.buffer_size))
        self.quat_buffer = np.zeros((4, self.buffer_size))

        self.target = o3d.geometry.PointCloud()
        self.target.points = o3d.utility.Vector3dVector(
            np.zeros((self.NUM_POINT, 3)))
        # self.target.paint_uniform_color([0, 0.651, 0.929])
        self.source = o3d.geometry.PointCloud()
        self.source.points = o3d.utility.Vector3dVector(np.zeros((900, 3)))
        self.source.paint_uniform_color([1, 0.706, 0])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=960, height=540)
        self.vis.add_geometry(self.target)
        self.vis.add_geometry(self.source)
        self.get_logger().info("Finished init")

    def callback(self, pc_msg):
        # self.get_logger().info("Callback!")
        start = time.perf_counter()
        pc_np = ros2_numpy.point_cloud2.pointcloud2_to_array(pc_msg, False)
        pc_np = ros2_numpy.point_cloud2.split_rgb_field(pc_np)
        depth_raw = ros2_numpy.point_cloud2.get_xyz_points(pc_np, False)/1000.0
        rgb_raw = np.zeros((pc_np.shape[0], pc_np.shape[1], 3))
        rgb_raw[:, :, 0] = pc_np['r']
        rgb_raw[:, :, 1] = pc_np['g']
        rgb_raw[:, :, 2] = pc_np['b']

        rgb = np.copy(rgb_raw)
        depth = np.copy(depth_raw)
        # Reduce the RGB field of view and limit what the camera has to consider for the ROI
        rgb[np.sqrt(np.power(depth[:, :, 0], 2) + np.power(depth[:, :, 1],
                    2) + np.power(depth[:, :, 2], 2)) > 2] = np.array([0, 0, 0])
        # rgb[depth[:, :, 0] > 0.8] = np.array([0, 0, 0])
        # rgb[depth[:, :, 1] > 0.8] = np.array([0, 0, 0])
        # rgb[depth[:, :, 2] > 2] = np.array([0, 0, 0])

        ############
        depth[depth[:, :, 2] > 0.8] = np.array([0, 0, 0])
        # cv2.imshow('depth', depth[:, :, -1])

        '''
        # downsample and crop to the input size

        # WFOV kinect settings
        rgb = 255 * np.ones((360, 640, 3))  # container in [360, 640]
        rgb[:, 64:576, :] = rgb_raw[76:436, :, :].astype(np.uint8)  # [512, 512] -> [360, 640]

        depth = np.zeros((360, 640, 3))
        depth[:, 64:576, :] = depth_raw[76:436, :, :]
        depth[depth[:,:, 2] > 0.8] = np.array([0,0,0])
        
        # NFOV kinect settings
        rgb = 255 * np.ones((360, 640, 3))  # container in [360, 640]
        rgb = rgb_raw[108:468, :, :].astype(np.uint8)  # [512, 512] -> [360, 640]

        depth = np.zeros((360, 640, 3))
        depth = depth_raw[108:468, :, :]        
        '''
        ''' inference: rgb for bounding box '''
        # if False:
        rgb_img = rgb.reshape([1, rgb.shape[0], rgb.shape[1], 3])
        # Run the Deep learning network on the image
        preds_loc_val, probs_val = self.sess.run(
            [self.preds_loc, self.probs], feed_dict={self.RGB: rgb_img})
        # Intermediary structure filled with one for the pixels with probability higher than 80 %
        preds_conf_val = (probs_val > 0.8).astype(int)
        # Final structure of pixels with probability higher than 80 %
        y_pred_conf = preds_conf_val[0].astype(
            'float32')  # batch size of 1, so just take [0]
        y_pred_loc = preds_loc_val[0]
        prob = probs_val[0]

        cv2.imshow('depth', depth[:, :, -1])
        rgb_raw_unprocessed = cv2.cvtColor(
            rgb_raw.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('rgb not processed', rgb_raw_unprocessed)  # BGR

        # Perform NMS
        box = model.nms(y_pred_conf, y_pred_loc, prob, extend=1.5)
        try:
            box_coords = [int(round(x)) for x in box[:4]]
            cls = int(box[4])
            cls_prob = box[5]

            x_min = np.maximum(box_coords[0], 0)
            x_max = np.minimum(box_coords[2], depth_raw.shape[1])
            y_min = np.maximum(box_coords[1], 0)
            y_max = np.minimum(box_coords[3], depth_raw.shape[0])

            # Annotate image
            label_str = '%s %.2f' % ('knee', cls_prob)
            rgb = cv2.rectangle(rgb, tuple(box_coords[:2]), tuple(
                box_coords[2:]), (0, 0, 255))
            rgb = cv2.putText(rgb, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0, 0, 255), 1,
                              cv2.LINE_AA)

            cropped = depth[y_min:y_max, x_min:x_max, :]
            cropped_rgb = np.copy(rgb_raw[y_min:y_max, x_min:x_max, :])
            h_crop, w_crop, _ = cropped.shape

            data = cropped.reshape((-1, cropped.shape[-1]))
            data_rgb = cropped_rgb.reshape((-1, cropped_rgb.shape[-1]))

            index0 = np.where(~np.all(data == 0, axis=1))[0]
            # data = data[~np.all(data == 0, axis=1)] # Delete datapoints that are 0 in the cropped ROI
            # data_rgb = data_rgb[~np.all(data == 0, axis=1)]
            # Delete datapoints that are 0 in the cropped ROI
            data = data[index0, :]
            data_rgb = data_rgb[index0, :]

            index1 = np.random.choice(range(data.shape[0]), self.NUM_POINT)
            # Random is causing the exception when the selection has no depth so random returns 0
            cropped_linear = data[index1, :].reshape(
                (1, self.NUM_POINT, 3))  # 0-1
            cropped_linear_rgb = data_rgb[index1,
                                          :].reshape((1, self.NUM_POINT, 3))

            self.target.points = o3d.utility.Vector3dVector(
                (cropped_linear * 39.3701).reshape((self.NUM_POINT, 3)))
            self.target.colors = o3d.utility.Vector3dVector(
                (cropped_linear_rgb / 255.0).reshape((self.NUM_POINT, 3)))
            # self.target.paint_uniform_color([0, 0.651, 0.929])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data[index1, :])
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
            pcd.orient_normals_towards_camera_location()
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            # o3d.io.write_point_cloud('/home/oceane/dev_ws/Tests_PCL/test.ply', pcd)

        except Exception as e:
            print(e)
            self.get_logger().info("An exception was thrown!")
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow('bgr', bgr)  # BGR
            key = cv2.waitKey(1)
            return

        ''' inference: segmentation '''
        thresh = 0.5
        pred_val = self.sess.run([self.prediction], feed_dict={
                                 self.PCL: cropped_linear})
        index2 = np.where(pred_val[0].squeeze() > thresh)[0]

        if len(index2) > 10:
            index = [index1[i] for i in index2]
            index = [index0[i] for i in index]

            row_col = [[i // w_crop, i % w_crop] for i in index]
            for pair in row_col:
                row = pair[0] + y_min
                col = pair[1] + x_min
                rgb[row, col] = np.array([0, 0, 255])

        # np.savetxt('/data/catkin_ws_iros/test.txt', pred_val[0].squeeze())
        femur = cropped_linear.squeeze()[pred_val[0].squeeze() > thresh]
        normals = np.asarray(pcd.normals)[index2]
        '''
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(femur)
        draw_registration_result(self.femur_cl, cloud)
        
        o3d.io.write_point_cloud('/home/oceane/dev_ws/Tests_PCL/test.ply', cloud)
        '''

        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('bgr', bgr)  # BGR
        key = cv2.waitKey(1)

        # print(femur.shape[0])

        if femur.shape[0] < 140:
            print('rejected!')
            print()
        else:
            print('accepted!')

            cloud = o3d.geometry.PointCloud()
            # Save pointcloud as '' instead of meters
            cloud.points = o3d.utility.Vector3dVector(femur * 39.3701)
            cloud.normals = o3d.utility.Vector3dVector(normals)

            # this is the pose
            fitness, mse, FD, correspondences = reg.registration(src_cloud=self.femur_cl, trg_cloud=cloud, init='mean',
                                                                 show=False, max_correspondence_distance=0.2)

            if correspondences < 500:
                print('Registration rejected due to only %d correspondences' %
                      correspondences)
                return

            if mse > 0.12:
                print('Registration rejected due to %f RMS error' % mse)
                return

            self.origin_buffer[0, self.buffer_idx] = FD[0, 3].item()
            self.origin_buffer[1, self.buffer_idx] = FD[1, 3].item()
            self.origin_buffer[2, self.buffer_idx] = FD[2, 3].item()

            quat = spRot.from_matrix(np.array(FD[0:3, 0:3].numpy())).as_quat()

            self.quat_buffer[0, self.buffer_idx] = quat[0]
            self.quat_buffer[1, self.buffer_idx] = quat[1]
            self.quat_buffer[2, self.buffer_idx] = quat[2]
            self.quat_buffer[3, self.buffer_idx] = quat[3]

            self.buffer_idx += 1
            if self.buffer_idx == self.buffer_size:
                self.buffer_idx = 0

            if np.sum(self.quat_buffer[:, self.buffer_size - 1]) == 0:
                return

            # mean_orig = np.mean(self.origin_buffer, 1)
            mean_orig = np.median(self.origin_buffer, 1)
            mean_quat = np.mean(self.quat_buffer, 1)
            mean_quat = mean_quat/np.linalg.norm(mean_quat)

            FD = np.eye(4)
            FD[0, 3] = mean_orig[0]
            FD[1, 3] = mean_orig[1]
            FD[2, 3] = mean_orig[2]

            mean_rot = spRot.from_quat(mean_quat)
            FD[0:3, 0:3] = mean_rot.as_matrix()

            src = np.asarray(self.femur_cl.points)
            index1 = np.random.choice(range(1323), 900)
            self.source.points = o3d.utility.Vector3dVector(
                src[index1, :].reshape((900, 3)))
            self.source.transform(FD)
            self.source.paint_uniform_color([1, 0.706, 0])
            flip_transform = [[1, 0, 0, 0], [
                0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            self.source.transform(flip_transform)
            self.target.transform(flip_transform)
            # rot_transform = [[0.8660254, 0, 0.5, 0], [0, 1, 0, 0], [-0.5, 0, 0.8660254, 0], [0, 0, 0, 1]]
            # self.source.transform(rot_transform)
            # self.target.transform(rot_transform)

            self.vis.add_geometry(self.source)
            self.vis.add_geometry(self.target)
            self.vis.poll_events()
            self.vis.update_renderer()

            depthTfem = PoseStamped()
            depthTfem.header.stamp = pc_msg.header.stamp
            depthTfem.pose.position.x = FD[0, 3].item()
            depthTfem.pose.position.y = FD[1, 3].item()
            depthTfem.pose.position.z = FD[2, 3].item()

            depthTfem_quat = spRot.from_matrix(
                np.array(FD[0:3, 0:3])).as_quat()

            depthTfem.pose.orientation.x = depthTfem_quat[0]
            depthTfem.pose.orientation.y = depthTfem_quat[1]
            depthTfem.pose.orientation.z = depthTfem_quat[2]
            depthTfem.pose.orientation.w = depthTfem_quat[3]

            self.pose_pub.publish(depthTfem)
            # print((time.perf_counter() - start) * 1000)


def main(args=None):
    rclpy.init(args=args)

    read_pc = pcCallback()
    read_pc.get_logger().info("Beginning of monitoring")

    while rclpy.ok():
        try:
            rclpy.spin_once(read_pc)
        except KeyboardInterrupt:
            rclpy.logging.get_logger("markerless").info("Keyboard interrupt")
            read_pc.vis.destroy_window()
            break

    rclpy.logging.get_logger("markerless").info("End of monitoring")
    read_pc.destroy_node()
    rclpy.logging.get_logger("markerless").info("Node destroyed")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
