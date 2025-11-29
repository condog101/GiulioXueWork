import os, shutil, json, cv2, socket, errno, time, math
import open3d as o3d, numpy as np
import atracsys.ftk as tracker_sdk
import pyrealsense2 as rs
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
from markerless_tracking.utils.registration import draw_registration_result
from scipy import linalg

def pcl2depth_acusense(pcl, rgb_intri, extri, offx=0, offy=0):
    '''   - z = d / depth_scale
          - x = (u - cx) * z / fx
          - y = (v - cy) * z / fy   '''
    d = np.zeros((pcl.shape[0], 2), dtype=int)  # converted depth

    for i in range(pcl.shape[0]):
        tmpx = pcl[i, 0] + extri[0, 3]
        tmpy = pcl[i, 1] + extri[1, 3]
        tmpz = pcl[i, 2] + extri[2, 3]
        tmp2x = tmpx * extri[0, 0] + tmpy * extri[1, 0] + tmpz * extri[2, 0]
        tmp2y = tmpx * extri[0, 1] + tmpy * extri[1, 1] + tmpz * extri[2, 1]
        tmp2z = tmpx * extri[0, 2] + tmpy * extri[1, 2] + tmpz * extri[2, 2]
        d[i, 0] = int(
            (rgb_intri.get_focal_length()[0] * tmp2x / tmp2z + rgb_intri.get_principal_point()[0]) / 1200 * 600 - offy)
        d[i, 1] = int(
            (rgb_intri.get_focal_length()[1] * tmp2y / tmp2z + rgb_intri.get_principal_point()[1]) / 1600 * 800 - offx)

    return d

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def messageMaker(array):
    message = ''
    array_flat = array.reshape(-1)
    for i in range(array_flat.shape[0] - 1):
        message = message + '%.5f' % array_flat[i] + ','
    message = message + '%.5f' % array_flat[-1]
    return message


# manual blocking up to 1s
def unblock_receive(sock, timeout=1):
    start = time.time()
    current = time.time()
    stringList = None
    while (current - start) < timeout:
        try:  # unblocked receiving
            data = sock.recv(1024)
            response = data.decode()
            stringList = response.split(',')
            print(time.time() - start)
            break
        except socket.error as e:
            error = e.args[0]
            current = time.time()

            if not (error == errno.EAGAIN or error == errno.EWOULDBLOCK):
                print(e)

    return stringList


''' draw '''


def visual_o3d(data, color=None, show=False):
    if color is None:
        color = [1, 0, 0]
    cloud = o3d.geometry.PointCloud()
    data = data.squeeze().reshape((-1, 3))
    data = data[~np.all(data == 0, axis=1)]
    cloud.points = o3d.utility.Vector3dVector(data)
    cloud.paint_uniform_color(color)
    if show:
        o3d.visualization.draw_geometries([cloud])
    return cloud


def pcl2depth(pcl, mtx, color=np.array([255, 0, 0])):
    '''   - z = d / depth_scale
          - x = (u - cx) * z / fx
          - y = (v - cy) * z / fy   '''
    d = np.zeros((pcl.shape[0], 5))  # converted depth

    for i in range(pcl.shape[0]):
        if pcl[i, 2] == 0:
            print('here')
        d[i, 0] = pcl[i, 0] * mtx[0, 0] / pcl[i, 2] + mtx[0, 2]  # u - 640
        d[i, 1] = pcl[i, 1] * mtx[1, 1] / pcl[i, 2] + mtx[1, 2]  # v - 360
        # color
        d[i, 2:5] = color
    return d


''' atracsys '''

def exit_with_error(error, tracking_system):
    print(error)
    errors_dict = {}
    if tracking_system.get_last_error(errors_dict) == tracker_sdk.Status.Ok:
        for level in ['errors', 'warnings', 'messages']:
            if level in errors_dict:
                print(errors_dict[level])
    exit(1)


class atracsys:
    def __init__(self, markerList):
        # load point cloud data
        self.tracking_system = tracker_sdk.TrackingSystem()
        if self.tracking_system.initialise() != tracker_sdk.Status.Ok:
            exit_with_error("Error, can't initialise the atracsys SDK api.", self.tracking_system)
        if self.tracking_system.enumerate_devices() != tracker_sdk.Status.Ok:
            exit_with_error("Error, can't enumerate devices.", self.tracking_system)
        self.frame = tracker_sdk.FrameData()
        if self.tracking_system.create_frame(False, 10, 20, 20, 10) != tracker_sdk.Status.Ok:
            exit_with_error("Error, can't create frame object.", self.tracking_system)
        print("Tracker with serial ID {0} detected".format(
            hex(self.tracking_system.get_enumerated_devices()[0].serial_number)))

        geometry_path = self.tracking_system.get_data_option("Data Directory")
        for geometry in markerList:
            if self.tracking_system.set_geometry(os.path.join(geometry_path, geometry)) != tracker_sdk.Status.Ok:
                exit_with_error("Error, can't create frame object.", self.tracking_system)

    def get(self, Md_id=1000, M_id=1000, tool_id=1000):
        MdA = None
        MA = None
        tool = None
        self.tracking_system.get_last_frame(self.frame)
        for marker in self.frame.markers:
            if marker.geometry_id == Md_id:
                # transformation from M_d -> A
                MdA = np.zeros((4, 4))
                MdA[0:3, 3] = np.array(marker.position) / 1000
                MdA[0:3, 0:3] = np.array(marker.rotation)
                MdA[3, :] = np.array([0, 0, 0, 1])

            elif marker.geometry_id == M_id:
                # transformation from M_f -> A
                MA = np.zeros((4, 4))
                MA[0:3, 3] = np.array(marker.position) / 1000
                MA[0:3, 0:3] = np.array(marker.rotation)
                MA[3, :] = np.array([0, 0, 0, 1])
            elif marker.geometry_id == tool_id:
                tool = np.identity(4)  # surgical pos and dir
                tip = np.array(marker.position) / 1000
                tool[0:3, 3] = tip
                tool[0:3, 0:3] = np.array(marker.rotation)

        return MdA, MA, tool


''' realsense '''

class realsense:
    def __init__(self, json_path="../scan/april30.json"):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        device = profile.get_device()

        with open(json_path) as file:
            as_json_object = json.load(file)
        json_str = str(as_json_object).replace("'", '\"')
        rs.rs400_advanced_mode(device).load_json(json_str)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.align = rs.align(rs.stream.color)

        self.hole_filling_filter = rs.hole_filling_filter()

    def get(self):
        ''' realsense update '''
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # colour frame
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        intrinsic = color_frame.profile.as_video_stream_profile().intrinsics

        # depth frame
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = self.hole_filling_filter.process(depth_frame)
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        rs_pointcloud = pc.calculate(depth_frame)
        v = rs_pointcloud.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        depth_raw = verts.reshape((360, 640, 3))  # for segmentation

        # depth_color_frame = rs.colorizer().colorize(depth_frame)
        # depth_color_image = np.asanyarray(depth_color_frame.get_data())

        return color_image, depth_raw, intrinsic

    def quit(self):
        self.pipeline.stop()


''' marker detection '''


class ToolDetection:
    def __init__(self, tool_cloud, cameraMatrix, distCoeffs, markerLength,
                 marker_dict=aruco.Dictionary_get(aruco.DICT_4X4_250)):
        self.tool_cloud = tool_cloud

        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.marker_dict = marker_dict
        self.markerLength = markerLength

        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = 10
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

        # lower left->upper left->upper right->lower right
        self.box = np.array([[-0.06, -0.07, 0, 1],
                             [-0.06, 0.07, 0, 1],
                             [0.06, 0.07, 0, 1],
                             [0.06, -0.07, 0, 1]])

    def detect(self, frame, depth, raw=False, show=False):
        MD = None
        data = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.marker_dict, parameters=self.parameters)

        # if no check is added the code will crash
        if ids is not None:
            # code to show ids of the marker found
            strg = str(ids[0][0])
            cv2.putText(frame, "Id: " + strg, (0, 64), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # initial transformation
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], self.markerLength, self.cameraMatrix,
                                                            self.distCoeffs)
            # aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, rvec, tvec, length=0.1)

            init = np.identity(4)
            rot = cv2.Rodrigues(rvec[0])[0]
            init[0:3, 0:3] = rot  # np.dot(rot, matz)
            init[0:3, 3] = tvec[0]

            # crop according to marker pose
            new_box = np.dot(init, self.box.transpose()).transpose()

            new_corners = pcl2depth(new_box[:, 0:3], self.cameraMatrix)

            x_min = int(np.min(new_corners[:, 0]))
            x_max = int(np.max(new_corners[:, 0]))
            y_min = int(np.min(new_corners[:, 1]))
            y_max = int(np.max(new_corners[:, 1]))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255))

            data = depth[y_min:y_max, x_min:x_max, :].reshape((-1, 3))
            mask = (data[:, 2:] < init[2, 3] + 0.03) * (data[:, 2:] > init[2, 3] - 0.03)
            data = data * mask
            data = data[~np.all(data == 0, axis=1)]

            # uvd = pcl2depth(data, self.cameraMatrix)
            # for i in range(uvd.shape[0]):
            #     if frame.shape[0] > int(uvd[i, 1]) > 0 and frame.shape[1] > int(uvd[i, 0]) > 0:
            #         frame[int(uvd[i, 1]), int(uvd[i, 0])] = uvd[i, 2:]

            if data.size > 10:  # surface points
                seg_cloud = o3d.geometry.PointCloud()
                seg_cloud.points = o3d.utility.Vector3dVector(data)
                # seg_cloud.paint_uniform_color([0,0,1])
                # o3d.visualization.draw_geometries([seg_cloud])
                '''registration'''
                reg_p2p = o3d.registration.registration_icp(self.tool_cloud, seg_cloud, 0.01, init,
                                                            o3d.registration.TransformationEstimationPointToPoint())
                # draw_registration_result(source=self.tool_cloud, target=seg_cloud, transformation=reg_p2p.transformation)
                if reg_p2p.fitness > 0.5:
                    MD = reg_p2p.transformation

        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0, 64), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # draw a square around the markers
        if show:
            aruco.drawDetectedMarkers(frame, corners)

        # if raw:
        #     return MD, data
        # else:
        return MD

    def detect_simple(self, frame):
        init = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.marker_dict, parameters=self.parameters)

        # if no check is added the code will crash
        if ids is not None:
            # code to show ids of the marker found
            strg = str(ids[0][0])
            cv2.putText(frame, "Id: " + strg, (0, 64), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # initial transformation
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], self.markerLength, self.cameraMatrix,
                                                            self.distCoeffs)
            aruco.drawAxis(frame, self.cameraMatrix, self.distCoeffs, rvec, tvec, length=0.1)

            init = np.identity(4)
            rot = cv2.Rodrigues(rvec[0])[0]
            init[0:3, 0:3] = rot  # np.dot(rot, matz)
            init[0:3, 3] = tvec[0]

        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0, 64), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return init


def rotation_around_y(d):
    r = np.deg2rad(d)
    return np.matrix([[np.cos(r), 0, -np.sin(r), 0], [0, 1, 0, 0], [np.sin(r), 0, np.cos(r), 0], [0, 0, 0, 1]],
                     dtype=np.float32)


def rotation_around_z(d):
    r = np.deg2rad(d)
    return np.matrix([[np.cos(r), np.sin(r), 0, 0], [-np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=np.float32)


def to_homogenous_position(a):
    size = a.shape[0]
    res = np.ones((size + 1, 1))
    res[:size, :] = a
    return res


def to_homogenous_translation(a):
    size = a.shape[0]
    res = np.identity(size + 1)
    res[:size, size] = a.flatten()
    return res


def to_homogenous_rotation(a):
    size = a.shape[0]
    res = np.identity(size + 1)
    res[:size, :size] = a
    return res


def translation(tx, ty, tz):
    return np.matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]], dtype=np.float32)


def hom2cart(p):
    return p[:-1] / p[-1]


# def aruco_points():
#     radius = 25  # mm
#     tc = radius / 3  # mm translation in x and y of corners from center of face
#     all_aruco_points = []
#     # points in local marker
#     origin_points = np.matrix([[-tc, -tc, 0, 1], [-tc, tc, 0, 1], [tc, tc, 0, 1], [tc, -tc, 0, 1]], dtype=np.float32).T
#
#     # first row
#     for i in [2, 1, 0, 4, 3]:
#         # pt_cube = marker2cube * pt_marker->cube
#         aruco_corners = rotation_around_z(72 * i) * rotation_around_y(116.565) * \
#                         rotation_around_z(180) * translation(0, 0, radius) * origin_points
#         all_aruco_points.append(hom2cart(aruco_corners).T)
#     # second row
#     for i in [0, 4, 3, 2, 1]:
#         aruco_corners = rotation_around_z(72 * i) * rotation_around_y(116.565) * translation(0, 0,
#                                                                                              -radius) * rotation_around_y(
#             180) * origin_points
#         all_aruco_points.append(hom2cart(aruco_corners).T)
#
#     # top
#     aruco_corners = translation(0, 0, radius) * origin_points
#     all_aruco_points.append(hom2cart(aruco_corners).T)
#     # bottom
#     aruco_corners = translation(0, 0, -radius) * rotation_around_y(180) * origin_points
#     all_aruco_points.append(hom2cart(aruco_corners).T)
#
#     all_aruco_points = np.array(all_aruco_points, dtype=np.float32)
#
#     return all_aruco_points

def aruco_points(MarkerLength=40, FaceSize=60) -> np.array:
    # express of cube in the face's frame
    flip_dict = {0: np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, -FaceSize / 2],
                              [0, 0, 0, 1]]),
                 1: np.array([[0, 0, -1, 0],
                              [0, 1, 0, 0],
                              [1, 0, 0, -FaceSize / 2],
                              [0, 0, 0, 1]]),
                 2: np.array([[0, 0, -1, 0],
                              [-1, 0, 0, 0],
                              [0, 1, 0, -FaceSize / 2],
                              [0, 0, 0, 1]]),

                 3: np.array([[-1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, -FaceSize / 2],
                              [0, 0, 0, 1]]),
                 4: np.array([[0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [-1, 0, 0, -FaceSize / 2],
                              [0, 0, 0, 1]])
                 }

    tc = MarkerLength / 2  # mm translation in x and y of corners from center of face
    all_aruco_points = []
    origin_points = np.array([[-tc, tc, 0, 1], [tc, tc, 0, 1], [tc, -tc, 0, 1], [-tc, -tc, 0, 1]]).transpose()

    # first row
    for i in [0, 1, 2, 3, 4]:
        aruco_corners = np.dot(np.linalg.inv(flip_dict[i]),
                               origin_points)  # np.dot(flip_dict[i], origin_points)  # marker -> cube
        all_aruco_points.append(aruco_corners[0:3].transpose())

    all_aruco_points = np.array(all_aruco_points, dtype=np.float32)
    return all_aruco_points


# board detection
class CubeDetection:
    def __init__(self, cameraMatrix, distCoeffs, marker_dict=aruco.Dictionary_get(aruco.DICT_4X4_250),
                 ids=np.array([[0], [1], [2], [3], [4]]), markerLength=60, faceSize=80):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.success = False
        self.marker_dict = marker_dict
        # self.board = aruco.Board_create(aruco_points(), marker_dict, np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]))
        self.board = aruco.Board_create(aruco_points(markerLength, faceSize), marker_dict, ids)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    def detect(self, img, threshold=0):

        trans = None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.marker_dict, parameters=self.parameters,
                                                              cameraMatrix=self.cameraMatrix,
                                                              distCoeff=self.distCoeffs)
        # aruco.refineDetectedMarkers(gray, self.board, corners, ids, rejectedImgPoints)

        if ids is not None and ids.shape[0] > threshold:
            # board optimisation
            self.success, rotation, translation_ = aruco.estimatePoseBoard(corners, ids, self.board,
                                                                           self.cameraMatrix,
                                                                           self.distCoeffs, None, None)
            if self.success:
                rvec = rotation.copy()
                tvec = translation_.copy() / 1000
                aruco.drawDetectedMarkers(img, corners, ids)

                rot = cv2.Rodrigues(rvec)[0]
                trans = np.identity(4)
                trans[0:3, 0:3] = rot
                trans[0:3, 3] = tvec.squeeze()
                aruco.drawAxis(img, self.cameraMatrix, self.distCoeffs, rvec, tvec, length=0.1)

        return trans


# sphere detection
class SphereDetection:
    def __init__(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        self.minArea = 40
        self.maxArea = 1000
        self.minCircularity = 0.4
        self.maxCircularity = 1

        self.eps = 0.01
        self.minInertiaRatio = 0
        self.maxInertiaRatio = 1

        self.minConvexity = 0.7
        self.maxConvexity = 1

        self._3dlist = []

    def nearest_nonzero_idx(self, a, x, y):
        idx = np.argwhere(a[:,:, 2])  # find non-zero indicies
        return idx[((idx - [x, y]) ** 2).sum(1).argmin()]

    def detect0(self, img, depth):
        _2dlist = []
        MD = None
        residual = None

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_green = np.array([20, 150, 30])  # Set the lower limit of the green threshold
        upper_green = np.array([40, 255, 255])  # Set the upper limit of the green threshold
        mask = cv2.inRange(hsv, lower_green, upper_green)  # Set the mask value range
        cv2.imshow('mask', mask)
        cv2.waitKey(1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        locations = []
        for contourIdx in range(len(contours)):
            moms = cv2.moments(contours[contourIdx])
            area = moms['m00']

            '''filter by area'''
            if area < self.minArea or area >= self.maxArea:
                continue

            '''filter by circularity'''
            perimeter = cv2.arcLength(contours[contourIdx], True)
            if perimeter == 0:
                continue
            ratio = 4 * np.pi * area / (perimeter * perimeter)
            if ratio < self.minCircularity or ratio > self.maxCircularity:
                continue

            '''filter by convexity'''
            hull = cv2.convexHull(contours[contourIdx])
            contour_area = cv2.contourArea(contours[contourIdx])
            hull_area = cv2.contourArea(hull)
            if abs(hull_area) < np.finfo(float).eps:
                continue
            hull_ratio = contour_area / hull_area
            if hull_ratio < self.minConvexity or hull_ratio > self.maxConvexity:
                continue

            location = np.array([moms['m10'] / moms['m00'], moms['m01'] / moms['m00']])
            locations.append(location)
            img = cv2.circle(img, (int(location[0]), int(location[1])), radius=5, color=(0, 0, 255), thickness=-1)
            # cv2.putText(img, ('%s | %s | %s | %s'%(area, ratio, hull_area, hull_ratio)), (int(location[0]), int(location[1])), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # pose solving
        locations_3d = []
        if len(locations) == 4:
            # print('-------------')
            for i in range(len(locations)):
                # print(depth[int(locations[i][1]), int(locations[i][0]), :])
                if not np.all(depth[int(locations[i][1]), int(locations[i][0]), :] == 0):
                    locations_3d.append(depth[int(locations[i][1]), int(locations[i][0]), :])  # y, x
                # else:
                #     ind = self.nearest_nonzero_idx(depth, int(locations[i][1]), int(locations[i][0]))
                #     locations_3d.append(depth[ind[0], ind[1], :])

        print('num: ', len(locations), ' 3d num: ', len(locations_3d))
        if len(locations) == 4 and len(locations_3d) == 4:
            '''method 2: SVD'''
            id_0 = None
            id_1 = None
            # print('----------------')
            for k in range(0, len(locations_3d)):
                remain = locations_3d.copy()
                remain.pop(k)
                angle1 = np.dot(remain[0] - locations_3d[k], remain[1] - locations_3d[k])/(np.linalg.norm(remain[0] - locations_3d[k]) * np.linalg.norm(remain[1] - locations_3d[k]))
                angle2 = np.dot(remain[0] - locations_3d[k], remain[2] - locations_3d[k])/(np.linalg.norm(remain[0] - locations_3d[k]) * np.linalg.norm(remain[2] - locations_3d[k]))
                angle3 = np.dot(remain[1] - locations_3d[k], remain[2] - locations_3d[k])/(np.linalg.norm(remain[1] - locations_3d[k]) * np.linalg.norm(remain[2] - locations_3d[k]))
                # print('%s, %s, %s, %s' % (k, angle1, angle2, angle3))
                if abs(angle1) < 0.17:
                    id_0 = locations_3d[k]
                    if np.linalg.norm(remain[0] - locations_3d[k]) > np.linalg.norm(remain[1] - locations_3d[k]):
                        id_1 = remain[0]
                        id_2 = remain[1]
                        id_3 = remain[2]
                    else:
                        id_1 = remain[1]
                        id_2 = remain[0]
                        id_3 = remain[2]

                elif abs(angle2) < 0.17:
                    id_0 = locations_3d[k]
                    if np.linalg.norm(remain[0] - locations_3d[k]) > np.linalg.norm(remain[2] - locations_3d[k]):
                        id_1 = remain[0]
                        id_2 = remain[2]
                        id_3 = remain[1]
                    else:
                        id_1 = remain[2]
                        id_2 = remain[0]
                        id_3 = remain[1]

                elif abs(angle3) < 0.17:
                    id_0 = locations_3d[k]
                    if np.linalg.norm(remain[1] - locations_3d[k]) > np.linalg.norm(remain[2] - locations_3d[k]):
                        id_1 = remain[1]
                        id_2 = remain[2]
                        id_3 = remain[0]
                    else:
                        id_1 = remain[2]
                        id_2 = remain[1]
                        id_3 = remain[0]

            if id_0 is not None and id_1 is not None:
                # locations_M = np.array([[0, 0, 0],[0, 0.06, 0],[0.04, 0, 0],[-0.05, -0.03, 0]])
                locations_M = np.array([[0, 0, 0],[0, 0, -0.06],[-0.04, 0, 0],[0.05, 0, 0.03]])
                locations_D = np.array([id_0, id_1, id_2, id_3])

                mean_M = np.mean(locations_M, axis=0)
                mean_D = np.mean(locations_D, axis=0)
                p = locations_M - mean_M
                q = locations_D - mean_D
                W = np.dot(q.transpose(), p)
                U, s, VT = np.linalg.svd(W)
                R = np.dot(U, VT)
                t = mean_D - np.dot(R, mean_M)
                MD = np.identity(4)
                MD[0:3, 0:3] = R
                MD[0:3, 3] = t
                locations_M_4d = np.concatenate((locations_M, np.ones((4,1))), axis=1).transpose()
                # locations_M_4d = np.array([[0, 0, 0, 1],[0, 0, -0.06, 1],[-0.04, 0, 0, 1],[0.05, 0, 0.03, 1]]).transpose()
                # locations_M_4d = np.array([[0, 0, 0, 1], [0, 0.06, 0, 1], [0.04, 0, 0, 1], [-0.05, -0.03, 0, 1]]).transpose()
                residual = np.mean(np.dot(MD, locations_M_4d)[0:3, :].transpose()-locations_D)
            # else:
            #     bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('color image2', bgr_img)  # BGR
            #     cv2.waitKey(0)
        return MD, residual

    def detect(self, img, depth):
        _2dlist = []
        MD = None
        residual = None

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_green = np.array([20, 150, 50])  # Set the lower limit of the green threshold
        upper_green = np.array([40, 255, 255])  # Set the upper limit of the green threshold
        mask = cv2.inRange(hsv, lower_green, upper_green)  # Set the mask value range
        cv2.imshow('mask', mask)
        cv2.waitKey(1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        locations = []
        for contourIdx in range(len(contours)):
            moms = cv2.moments(contours[contourIdx])
            area = moms['m00']

            '''filter by area'''
            if area < self.minArea or area >= self.maxArea:
                continue

            '''filter by circularity'''
            perimeter = cv2.arcLength(contours[contourIdx], True)
            if perimeter == 0:
                continue
            ratio = 4 * np.pi * area / (perimeter * perimeter)
            if ratio < self.minCircularity or ratio > self.maxCircularity:
                continue

            '''filter by convexity'''
            hull = cv2.convexHull(contours[contourIdx])
            contour_area = cv2.contourArea(contours[contourIdx])
            hull_area = cv2.contourArea(hull)
            if abs(hull_area) < np.finfo(float).eps:
                continue
            hull_ratio = contour_area / hull_area
            if hull_ratio < self.minConvexity or hull_ratio > self.maxConvexity:
                continue

            location = np.array([moms['m10'] / moms['m00'], moms['m01'] / moms['m00']])
            locations.append(location)
            img = cv2.circle(img, (int(location[0]), int(location[1])), radius=5, color=(0, 0, 255), thickness=-1)
            # cv2.putText(img, ('%s | %s | %s | %s'%(area, ratio, hull_area, hull_ratio)), (int(location[0]), int(location[1])), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # pose solving
        locations_3d = []
        if len(locations) == 4:
            # print('-------------')
            for i in range(len(locations)):
                # print(depth[int(locations[i][1]), int(locations[i][0]), :])
                if not np.all(depth[int(locations[i][1]), int(locations[i][0]), :] == 0):
                    locations_3d.append(depth[int(locations[i][1]), int(locations[i][0]), :])  # y, x
                # else:
                #     ind = self.nearest_nonzero_idx(depth, int(locations[i][1]), int(locations[i][0]))
                #     locations_3d.append(depth[ind[0], ind[1], :])

        print('detected num: ', len(locations), ' valid num: ', len(locations_3d))
        if len(locations) == 4 and len(locations_3d) == 4:
            '''method 2: SVD'''
            id_0 = None
            id_1 = None
            # print('----------------')
            set = False
            for k in range(0, len(locations_3d)):
                remain = locations_3d.copy()
                remain.pop(k)
                angle1 = np.dot(remain[0] - locations_3d[k], remain[1] - locations_3d[k])/(np.linalg.norm(remain[0] - locations_3d[k]) * np.linalg.norm(remain[1] - locations_3d[k]))
                angle2 = np.dot(remain[0] - locations_3d[k], remain[2] - locations_3d[k])/(np.linalg.norm(remain[0] - locations_3d[k]) * np.linalg.norm(remain[2] - locations_3d[k]))
                angle3 = np.dot(remain[1] - locations_3d[k], remain[2] - locations_3d[k])/(np.linalg.norm(remain[1] - locations_3d[k]) * np.linalg.norm(remain[2] - locations_3d[k]))
                # print('%s, %s, %s, %s' % (k, angle1, angle2, angle3))
                if angle1 < -0.7:
                    id_0 = locations_3d[k]
                    set = True
                    if np.linalg.norm(remain[0] - locations_3d[k]) > np.linalg.norm(remain[1] - locations_3d[k]):
                        id_1 = remain[0]
                        id_2 = remain[1]
                        id_3 = remain[2]
                    else:
                        id_1 = remain[1]
                        id_2 = remain[0]
                        id_3 = remain[2]

                elif angle2 < -0.7:
                    set = True
                    id_0 = locations_3d[k]
                    if np.linalg.norm(remain[0] - locations_3d[k]) > np.linalg.norm(remain[2] - locations_3d[k]):
                        id_1 = remain[0]
                        id_2 = remain[2]
                        id_3 = remain[1]
                    else:
                        id_1 = remain[2]
                        id_2 = remain[0]
                        id_3 = remain[1]

                elif angle3 < -0.7:
                    set = True
                    id_0 = locations_3d[k]
                    if np.linalg.norm(remain[1] - locations_3d[k]) > np.linalg.norm(remain[2] - locations_3d[k]):
                        id_1 = remain[1]
                        id_2 = remain[2]
                        id_3 = remain[0]
                    else:
                        id_1 = remain[2]
                        id_2 = remain[1]
                        id_3 = remain[0]
                if set:
                    break

            if id_0 is not None and id_1 is not None:
                locations_M = np.array([[0, 0, -0.00534],[0, -0.025, 0],[0, 0.02, 0],[0.03, 0, 0]])
                locations_D = np.array([id_0, id_1, id_2, id_3])

                mean_M = np.mean(locations_M, axis=0)
                mean_D = np.mean(locations_D, axis=0)
                p = locations_M - mean_M
                q = locations_D - mean_D
                W = np.dot(q.transpose(), p)
                U, s, VT = np.linalg.svd(W)
                R = np.dot(U, VT)
                t = mean_D - np.dot(R, mean_M)
                MD = np.identity(4)
                MD[0:3, 0:3] = R
                MD[0:3, 3] = t
                locations_M_4d = np.concatenate((locations_M, np.ones((4,1))), axis=1).transpose()
                # locations_M_4d = np.array([[0, 0, 0, 1],[0, 0, -0.06, 1],[-0.04, 0, 0, 1],[0.05, 0, 0.03, 1]]).transpose()
                # locations_M_4d = np.array([[0, 0, 0, 1], [0, 0.06, 0, 1], [0.04, 0, 0, 1], [-0.05, -0.03, 0, 1]]).transpose()
                residual = np.mean(np.dot(MD, locations_M_4d)[0:3, :].transpose()-locations_D)
            # else:
            #     bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('color image2', bgr_img)  # BGR
            #     cv2.waitKey(0)
        return MD, residual


# kalman filter
def rotationMatrixToEuler(R_x):
    sy = math.sqrt(R_x[0, 0] * R_x[0, 0] + R_x[1, 0] * R_x[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R_x[2, 1], R_x[2, 2])
        y = math.atan2(-R_x[2, 0], sy)
        z = math.atan2(R_x[1, 0], R_x[0, 0])
    else:
        x = math.atan2(-R_x[1, 2], R_x[1, 1])
        y = math.atan2(-R_x[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def format_conv(measured_pose):
    rot = measured_pose[0:3, 0:3]
    trans = measured_pose[0:3, 3]#*1000
    Eu = R.from_matrix(rot).as_euler('zxy')

    # Eu = rotationMatrixToEuler(rot)
    x = np.float(trans[0])
    y = np.float(trans[1])
    z = np.float(trans[2])
    a = np.float(Eu[0])
    b = np.float(Eu[1])
    c = np.float(Eu[2])
    f = np.array([x, y, z, a, b, c], dtype=np.float32)
    return f


def EulerToRotationMatrix(theta):
    mat = R.from_euler('zxy', theta.reshape(3)).as_matrix()
    return mat


def filter_location(nState=3, nMeasure=3):  # x y z
    kalman = cv2.KalmanFilter(nState, nMeasure)

    transitionM = np.eye(nState)
    kalman.transitionMatrix = np.array(transitionM, dtype=np.float32)

    measureM = np.eye(nState)
    kalman.measurementMatrix = np.array(measureM, dtype=np.float32)

    kalman.processNoiseCov = 1e-2 * np.identity(nState, np.float32)
    kalman.measurementNoiseCov = 1e-3 * np.identity(nMeasure, np.float32)

    return kalman



def filter_pose(nState=18, nMeasure=6):  # xyz, dx dy dz, ddx ddy ddz ; ... # x y z thetax y z
    kalman = cv2.KalmanFilter(nState, nMeasure, 0)

    measureM = np.zeros((nMeasure, nState))
    transitionM = np.eye(nState)

    kalman.processNoiseCov = 1e-5 * np.identity(nState, np.float32)
    kalman.measurementNoiseCov = 1e-4 * np.identity(nMeasure, np.float32)
    kalman.errorCovPost = np.identity(nState, np.float32)

    dt = np.array(1 / 30, dtype=np.float32)
    transitionM[0, 3] = dt
    transitionM[1, 4] = dt
    transitionM[2, 5] = dt
    transitionM[3, 6] = dt
    transitionM[4, 7] = dt
    transitionM[5, 8] = dt
    transitionM[9, 12] = dt
    transitionM[10, 13] = dt
    transitionM[11, 14] = dt
    transitionM[12, 15] = dt
    transitionM[13, 16] = dt
    transitionM[14, 17] = dt
    transitionM[0, 6] = (1 / 2) * (np.square(dt))
    transitionM[1, 7] = (1 / 2) * (np.square(dt))
    transitionM[2, 8] = (1 / 2) * (np.square(dt))
    transitionM[9, 15] = (1 / 2) * (np.square(dt))
    transitionM[10, 16] = (1 / 2) * (np.square(dt))
    transitionM[11, 17] = (1 / 2) * (np.square(dt))
    kalman.transitionMatrix = np.array(transitionM, dtype=np.float32)

    measureM[0, 0] = 1
    measureM[1, 1] = 1
    measureM[2, 2] = 1
    measureM[3, 9] = 1
    measureM[4, 10] = 1
    measureM[5, 11] = 1
    kalman.measurementMatrix = np.array(measureM, dtype=np.float32)

    return kalman

def filter_pose_const(nState=6, nMeasure=6):  # xyz, dx dy dz, ddx ddy ddz ; ... # x y z thetax y z
    kalman = cv2.KalmanFilter(nState, nMeasure, 0)

    transitionM = np.eye(nState)
    kalman.transitionMatrix = np.array(transitionM, dtype=np.float32)
    # -5,-4
    kalman.processNoiseCov = 1e-4 * np.identity(nState, np.float32)
    kalman.measurementNoiseCov = 1e-3 * np.identity(nMeasure, np.float32)
    kalman.errorCovPost = 0.1*np.identity(nState, np.float32)

    measureM = np.eye(nMeasure, nState)
    kalman.measurementMatrix = np.array(measureM, dtype=np.float32)

    return kalman

''' sphere fitting for hip centre estimation '''


def sphere_fitting(points):
    n = points.shape[0]
    radius_ = points[:, 0] * points[:, 0] \
              + points[:, 1] * points[:, 1] \
              + points[:, 2] * points[:, 2]
    Z = np.c_[radius_, points, np.ones(n)]
    # matrix of moments
    M = Z.transpose().dot(Z) / n

    # pratt fit
    P = np.zeros([5, 5])
    P[4, 0] = P[0, 4] = -2
    P[1, 1] = P[2, 2] = P[3, 3] = 1
    # Taubin fit
    T = np.zeros([5, 5])
    T[0, 0] = 4 * M[0, 4]
    T[0, 1] = T[1, 0] = 2 * M[0, 3]
    T[0, 2] = T[2, 0] = 2 * M[0, 2]
    T[0, 3] = T[3, 0] = 2 * M[0, 1]
    T[1, 1] = T[2, 2] = T[3, 3] = 1

    # hyperaccurate
    H = 2 * T - P
    if (np.sum(np.isnan(M)) > 0 or np.sum(np.isnan(H)) > 0):
        coeff = np.zeros(4)
        status = False
        return points, coeff, status
    # MA = yHA
    eigvals, eigvecs = linalg.eig(M, H)
    eigvals[np.where(eigvals < 0)] = np.inf
    sort_idx = np.argsort(np.abs(eigvals))
    min_eig_var_idx = sort_idx[0]
    _coeff = eigvecs[:, min_eig_var_idx]
    coeff = np.zeros(4)
    coeff[0] = - _coeff[1] / (2 * _coeff[0])
    coeff[1] = - _coeff[2] / (2 * _coeff[0])
    coeff[2] = - _coeff[3] / (2 * _coeff[0])
    coeff[3] = np.sqrt(
        (_coeff[1] * _coeff[1] + _coeff[2] * _coeff[2] +
         _coeff[3] * _coeff[3] - 4 * _coeff[0] * _coeff[4])
        / (4 * _coeff[0] * _coeff[0])
    )
    status = True
    if (np.sum(np.isnan(coeff)) > 0):
        status = False
    return coeff, status
