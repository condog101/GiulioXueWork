import os, time, copy, cv2
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import cv2.aruco as aruco
import ctypes
from scipy.spatial.transform import Rotation as R
import scipy

""" ICP registration """


def draw_registration_result_t(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()
    source_temp.transform(transformation)

    # Convert open3d.cpu.pybind.t.geometry.PointCloud → open3d.cpu.pybind.geometry.PointCloud
    src = source_temp.to_legacy()
    trg = target_temp.to_legacy()
    # Paint points in new format
    src.paint_uniform_color([1, 0.706, 0])
    trg.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([src, trg])


def draw_t(source, target):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.point["colors"] = ([1, 0.706, 0])
    target_temp.point["colors"] = ([0, 0.651, 0.929])

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    try:
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    except:
        print("Normals estimation failed")
    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# Past registration
def reg(src_cloud, trg_cloud, mean_align=False, init=None, show=False, threshold=0.05):
    # if an initial transform is provided
    if mean_align:
        if init is None:
            sx, sy, sz = np.mean(np.array(src_cloud.points), axis=0)
            tx, ty, tz = np.mean(np.array(trg_cloud.points), axis=0)
            mean_align = np.asarray([[1, 0, 0, tx - sx],
                                     [0, 1, 0, ty - sy],
                                     [0, 0, 1, tz - sz],
                                     [0.0, 0.0, 0.0, 1.0]])
        else:
            src_cloud_2 = copy.deepcopy(src_cloud).transform(init)
            sx, sy, sz = np.mean(np.array(src_cloud_2.points), axis=0)
            tx, ty, tz = np.mean(np.array(trg_cloud.points), axis=0)
            mean_align = np.asarray([[1, 0, 0, tx - sx],
                                     [0, 1, 0, ty - sy],
                                     [0, 0, 1, tz - sz],
                                     [0.0, 0.0, 0.0, 1.0]])
            mean_align = np.dot(mean_align, init)

        reg_p2p = o3d.pipelines.registration.registration_icp(src_cloud, trg_cloud, threshold, mean_align,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # else, do global transform first
    else:
        # ------------- ransac global alignment --------------
        if init is not None:
            src_cloud_2 = copy.deepcopy(src_cloud).transform(init)
        else:
            src_cloud_2 = copy.deepcopy(src_cloud)

        voxel_size = 0.004
        source_down, source_fpfh = preprocess_point_cloud(src_cloud_2, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(trg_cloud, voxel_size)

        distance_threshold = voxel_size
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

        if init is not None:
            reg_p2p = o3d.pipelines.registration.registration_icp(src_cloud, trg_cloud, threshold,
                                                                  np.dot(result_ransac.transformation, init),
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint())
        else:
            reg_p2p = o3d.pipelines.registration.registration_icp(src_cloud, trg_cloud, threshold,
                                                                  result_ransac.transformation,
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # print('Fitness: ', reg_p2p.fitness, ' Inlier_rmse: ', reg_p2p.inlier_rmse)

    T = reg_p2p.transformation

    if show:
        draw_registration_result_t(src_cloud, trg_cloud, reg_p2p.transformation)

    return reg_p2p.fitness, reg_p2p.inlier_rmse, T



#########################################################################################################

def registration(src_cloud, trg_cloud, init='mean', show=False, max_correspondence_distance=0.2):
    if init == 'mean':
        sx, sy, sz = np.mean(np.array(src_cloud.points), axis=0)
        tx, ty, tz = np.mean(np.array(trg_cloud.points), axis=0)
        # Initial alignment or source to target transform.
        init_source_to_target = np.asarray([[-1, 0, 0, tx - sx],
                                            [0, -1, 0, ty - sy],
                                            [0, 0, 1, tz - sz],
                                            [0.0, 0.0, 0.0, 1.0]], dtype="float32")

    #source = o3d.t.geometry.PointCloud(np.asarray(src_cloud.points))
    #target = o3d.t.geometry.PointCloud(np.asarray(trg_cloud.points))

    source = o3d.t.geometry.PointCloud.from_legacy(src_cloud)
    target = o3d.t.geometry.PointCloud.from_legacy(trg_cloud)

    # source.point["positions"] = source.point["positions"].to(o3d.core.Dtype.Float32)
    # target.point["positions"] = target.point["positions"].to(o3d.core.Dtype.Float32)

    if show:
        draw_t(source, target)
        draw_registration_result_t(source, target, init_source_to_target)

    # Search distance for Nearest Neighbour Search[Hybrid-Search is used].
    # max_correspondence_distance = 0.005

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    '''
    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(
                                                    o3d.t.pipelines.registration.robust_kernel.RobustKernel(
                                                    o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss, 0.08)
                                                    )
                                                    '''

    # Convergence-Criteria for Vanilla ICP
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                                                   relative_rmse=0.000001,
                                                                   max_iteration=10000)
    # Down-sampling voxel-size.
    # voxel_size = 0.0001

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    # save_loss_log = True

    try:
        # http://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html#Understanding-ICP-APIprint(type(source))

        '''
        # GPU
        source = source.cuda()
        target = target.cuda()
        s = time.perf_counter_ns()
        reg_t_icp = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance,
                                                            init_source_to_target, estimation, criteria)
        #                                                    voxel_size, save_loss_log)
        icp_time = (time.perf_counter_ns() - s)/1000000.0
        print("Time taken by ICP - GPU [ms]: ", icp_time)
        print("Inlier Fitness - GPU: ", reg_t_icp.fitness)
        print("Inlier RMSE - GPU: ", reg_t_icp.inlier_rmse)
        '''
        # print("Vanilla ICP")
        # plot_rmse(registration_icp)

        # CPU
        #source = source.cuda()
        #target = target.cuda()

        s = time.perf_counter_ns()
        reg_t_icp = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance,
                                                     init_source_to_target, estimation, criteria, 0.05)

        print('# correspondences: %d' % (np.sum(reg_t_icp.correspondences_.cpu().numpy() != -1)))
        print('Inlier RMSE: %f' % (reg_t_icp.inlier_rmse))
        print()

        '''
        icp_time = (time.perf_counter_ns() - s) / 1000000.0
        print("Time taken by ICP - CPU [ms]: ", icp_time)        
        print("Inlier Fitness - CPU: ", reg_t_icp.fitness)
        print("Inlier RMSE - CPU: ", reg_t_icp.inlier_rmse)
        '''

        # print("Vanilla ICP")
        # plot_rmse(registration_icp)

    except Exception as e:
        print(e)
        print("EXCEPTION")
        return

    if show:
        draw_registration_result_t(source, target, reg_t_icp.transformation)

    return reg_t_icp.fitness, reg_t_icp.inlier_rmse, reg_t_icp.transformation, np.sum(reg_t_icp.correspondences_.cpu().numpy() != -1)


#########################################################################################################


""" BICP registration """
c_float_p = ctypes.POINTER(ctypes.c_float)


class BICP:
    def __init__(self, src_data, src_hip):
        # load point cloud data
        os.environ['path'] += r';C:\Users\HP\source\repos\pcl\x64\Release'
        self.lib = ctypes.cdll.LoadLibrary("pclDll.dll")
        self.src_size = src_data.shape[0]
        self.src_data = src_data.copy().astype(np.float32).ctypes.data_as(c_float_p)
        self.src_hip = src_hip.copy().astype(np.float32).ctypes.data_as(c_float_p)

    def align(self, trg_data, trg_hip):
        trg_size = trg_data.shape[0]
        trg_data = trg_data.copy().astype(np.float32).ctypes.data_as(c_float_p)
        trg_hip = trg_hip.copy().astype(np.float32).ctypes.data_as(c_float_p)

        T_arr = (ctypes.c_float * 16)()
        self.lib.BICP.argtypes = [c_float_p, ctypes.c_int, c_float_p,
                                  c_float_p, ctypes.c_int, c_float_p,
                                  ctypes.c_int, ctypes.c_float, ctypes.c_float, c_float_p]
        self.lib.BICP.restype = ctypes.c_int

        converged = self.lib.BICP(self.src_data, self.src_size, self.src_hip,
                                  trg_data, trg_size, trg_hip,
                                  300, 0.01, 0.000001, T_arr)
        T = np.asarray(list(T_arr)).reshape((4, 4)).transpose()
        # print('converged: ', converged, '\n', T)
        return T


""" rcICP registration """


def TransformPoints(pts_in, M, crop=True):  # [N, 3/4] -> [N, 3/4]
    pts = pts_in.copy()
    assert pts.shape[1] == 3 or pts.shape[1] == 4
    if pts.shape[1] == 3:
        pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1).transpose()  # [N, 4] -> [4, N]
    else:
        pts = pts.transpose()
    pts_trans = np.dot(M, pts)  # points to search (already aligned)
    if crop:
        return pts_trans.transpose()[:, 0:3]
    else:
        return pts_trans.transpose()


def solve_svd(A, b):
    # compute svd of A
    U, s, VT = np.linalg.svd(A)
    len = np.shape(s)[0]
    sigma_pinv = np.zeros(A.shape).transpose()
    sigma_pinv[:len, :len] = np.diag(1 / s)
    c = np.dot(U.transpose(), b)
    # w = np.divide(c[0:len], s.reshape((-1,1)))
    w = np.dot(sigma_pinv, c)
    x = np.dot(VT.transpose(), w)
    return x


def rcICP(src, trg, X, o4=1 / 2, n_SUPP=10):
    # initial transformation to roughly align
    # print("First align two clouds together in translation...")
    M = np.identity(4)
    # init_trans = np.mean(trg[:, 0:3], axis=0) - np.mean(src[:, 0:3], axis=0)
    # M[0:3, 3] = M[0:3, 3] + init_trans

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src[:, 0:3])
    source.paint_uniform_color([1, 0, 0])
    vis.add_geometry(source)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(trg[:, 0:3])
    target.paint_uniform_color([0, 0, 1])
    vis.add_geometry(target)

    vis.update_geometry(source)
    vis.update_geometry(target)
    vis.poll_events()
    vis.update_renderer()

    # Build a kdtree out of the points in trg
    print("Building KdTree in Target")
    tree = KDTree(trg[:, 0:3], leaf_size=40)

    # Initialisation
    count = 0
    list_dist = [1000]
    ratio = 0.0

    # Initialisation for global convergence
    n_disturb = 0
    best = []
    global_minimum = 1000
    global_M = np.identity(4)

    # ICP iteration (until improvement is less than 0.01% or larger than 1)
    print("Starting iteration...")
    while ratio < 0.9999 and count < 100:
        # find closest point in trg by kdtree
        p = TransformPoints(src.copy(), M)  # N x 3
        dist, nearest_ind = tree.query(p[:, 0:3], k=1)
        q = trg[nearest_ind.squeeze(), 0:3]

        dis_vect = np.subtract(p.copy(), q.copy())
        v = dis_vect / np.linalg.norm(dis_vect, axis=1, keepdims=True)

        # Compute point to point distances
        point2point = np.abs(np.sum(np.subtract(p.copy(), q.copy()) * v, axis=1))
        # Cull outliers
        median_3x = 3.0 * np.median(np.abs(point2point))
        inliners = (np.abs(point2point) <= median_3x)
        index = np.where(inliners == True)[0]
        dist_sum = np.sum(point2point[inliners])

        old_mean = dist_sum / len(index)

        ''' Optimisation here '''

        s = TransformPoints(src, X)
        d = TransformPoints(q, X)
        # v4D = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1).transpose()[0:3,:]  # 4, N
        # Xinv = np.linalg.inv(X)
        v4D = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1)  # N, 4
        vT = np.dot(v4D, np.linalg.inv(X))[:, 0:3]  # N, 3

        ''' 2.1 Solve the linear component U and compute Micp_U '''
        # Construct A and b:
        A = np.zeros(shape=(len(index), 3))
        b = np.zeros(shape=(len(index), 1))

        ind = 0
        for i in index:
            A[ind, :] = np.array([-vT[i, 0] * (1 - o4 * s[i, 2]),
                                  -vT[i, 1] * (1 - o4 * s[i, 2]),
                                  -vT[i, 0] * s[i, 0] * o4 - vT[i, 1] * s[i, 1] * o4 - vT[i, 2]]).reshape([1, -1])
            b[ind] = np.sum(d[i] * vT[i]) - np.sum(s[i] * vT[i])  # distance, should be
            ind = ind + 1

        x = solve_svd(A, b)
        phi_1, phi_2, phi_3 = x[:, 0]

        count += 1
        ratio = old_mean / list_dist[-1]

        # Update M1 if we improved (otherwise, but NOT only then, we will terminate)
        if ratio < 1.0:
            UQ = np.array([[1 - phi_3 * o4, 0, phi_1 * o4, -phi_1],
                           [0, 1 - phi_3 * o4, phi_2 * o4, -phi_2],
                           [0, 0, 1, -phi_3],
                           [0, 0, 0, 1]])
            M = np.dot(np.linalg.inv(X), np.dot(UQ, X))
            best_M = M.copy()
            list_dist.append(old_mean)

            # display
            source.points = o3d.utility.Vector3dVector(TransformPoints(src.copy(), M))
            vis.update_geometry(source)
            vis.update_geometry(target)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1 / 2)

        # skip local minimum
        else:
            # draw_registration_result(source=visual_o3d(src[:, 0:3]), target=visual_o3d(trg[:, 0:3]), transformation=M)
            n_disturb = n_disturb + 1
            best.append(list_dist[-1])
            # if global_minimum > list_dist[-1] and count is not 2:
            if global_minimum > list_dist[-1]:
                global_minimum = list_dist[-1].copy()
                global_M = best_M.copy()
                # print('min', global_minimum, '\n', global_M)

            if n_disturb > n_SUPP:
                break

            ratio = 0
            rand = 0.005 * (2 * np.random.rand(3) - 1)

            M[0, 3] = M[0, 3] + rand[0]
            M[1, 3] = M[1, 3] + rand[1]
            M[2, 3] = M[2, 3] + rand[2]
            list_dist.append(1000)
            print('************ Restart from disturbed minimum')

    print('best', best)
    if n_disturb == 0:
        global_M = best_M
        global_minimum = list_dist[-1]

    return global_M, global_minimum


def logR(R):
    omega = scipy.linalg.logm(R)
    omega = np.array([omega[2, 1], omega[0, 2], omega[1, 0]])  # [w1,w2,w3]'
    return omega


def AXXB(data_frames):
    length = len(data_frames)
    alpha = np.zeros((length, 3))  # [N, 3]
    beta = np.zeros((length, 3))  # [N, 3]
    # data_frames[0]['A'] = np.array([[-0.989992 ,-0.141120, 0, 0],
    #        [0.141120, -0.989992, 0,  0],
    #        [0, 0, 1, 0],
    #        [0, 0, 0, 1]])
    # data_frames[0]['B'] = np.array([[-0.989992, -0.138307, 0.028036, -26.9559],
    #                 [0.138307, -0.911449, 0.387470, -96.1332],
    #                 [-0.028036, 0.387470, 0.921456, 19.4872],
    #                 [0, 0, 0 ,1]])
    # data_frames[1]['A'] = np.array([[0.07073, 0.000000, 0.997495, -400.000],
    #         [0.000000, 1.000000, 0.000000, 0.000000],
    #         [-0.997495, 0.000000, 0.070737 ,400.000],
    #         [0, 0, 0, 1]])
    # data_frames[1]['B'] = np.array([[0.070737, 0.198172, 0.997612, -309.543],
    #                 [-0.198172, 0.963323, -0.180936, 59.0244],
    #                 [-0.977612, -0.180936, 0.107415, 291.177],
    #                 [0, 0, 0, 1]])
    for data in range(length):
        RA1 = data_frames[data]['D1D2'][0:3, 0:3]
        RB1 = data_frames[data]['M1M2'][0:3, 0:3]
        alpha[data, :] = logR(RA1)
        beta[data, :] = logR(RB1)

    r, e = R.align_vectors(alpha, beta)
    Rx = r.as_matrix()

    C = np.zeros((length * 3, 3))  # [3N, 3]
    D = np.zeros((length * 3, 1))  # [3N, 1]
    for j in range(length):
        C[j * 3:j * 3 + 3, :] = data_frames[j]['D1D2'][0:3, 0:3] - np.eye(3)
        D[j * 3:j * 3 + 3, 0] = np.dot(Rx, data_frames[j]['M1M2'][0:3, 3]) - data_frames[j]['D1D2'][0:3, 3]
    Px = np.dot(np.dot(np.linalg.inv(np.dot(C.transpose(), C)), C.transpose()), D)
    Tx = np.identity(4)
    Tx[0:3, 0:3] = Rx
    Tx[0:3, 3] = Px.squeeze()

    print(Tx)

    return Tx
