'''
Model definition
'''
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
from utils.settings import *
import utils.tf_util as tf_util

''' seg net '''


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_XYZ') as sc:
        assert (K == 3)
        weights = tf.compat.v1.get_variable('weights', [256, 3 * K],
                                            initializer=tf.constant_initializer(
                                                0.0),
                                            dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [3 * K],
                                           initializer=tf.constant_initializer(
                                               0.0),
                                           dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, K * K],
                                            initializer=tf.constant_initializer(
                                                0.0),
                                            dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K * K],
                                           initializer=tf.constant_initializer(
                                               0.0),
                                           dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def get_model(point_cloud, is_training=tf.constant(False, dtype=tf.bool), scope='newseg', bn_decay=None,
              reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_point = point_cloud.get_shape()[1].value
        end_points = {}

        with tf.compat.v1.variable_scope('transform_net1') as sc:
            transform = input_transform_net(
                point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        with tf.compat.v1.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        point_feat = tf.expand_dims(net_transformed, [2])
        print(point_feat)

        net = tf_util.conv2d(point_feat, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net_hx = tf_util.conv2d(net, 1024, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv5', bn_decay=bn_decay)

        global_feat = tf_util.max_pool2d(net_hx, [num_point, 1],
                                         padding='VALID', scope='maxpool')
        print(global_feat)

        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        concat_feat = tf.concat([point_feat, global_feat_expand], 3)
        print(concat_feat)

        net = tf_util.conv2d(concat_feat, 512, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv8', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv9', bn_decay=bn_decay)

        net = tf_util.conv2d(net, 1, [1, 1],
                             padding='VALID', stride=[1, 1], activation_fn=None,
                             scope='conv10')
        net = tf.squeeze(net, [2])  # BxNxC
        net = tf.squeeze(net, [2])  # BxNxC
        net = tf.sigmoid(net)
        return net


''' roi net '''


def SSDHook(feature_map, hook_id):
    """
    Takes input feature map, output the predictions tensor
    hook_id is for variable_scope unqie string ID
    """
    with tf.compat.v1.variable_scope('ssd_hook_' + hook_id):
        # Note we have linear activation (i.e. no activation function)
        net_conf = slim.conv2d(feature_map, NUM_PRED_CONF, [
                               3, 3], activation_fn=None, scope='conv_conf')
        net_conf = tf.compat.v1.layers.flatten(net_conf)

        net_loc = slim.conv2d(feature_map, NUM_PRED_LOC, [
                              3, 3], activation_fn=None, scope='conv_loc')
        net_loc = tf.compat.v1.layers.flatten(net_loc)

    return net_conf, net_loc


def AlexNet(x, scope='newroi', channels=32, reuse=tf.compat.v1.AUTO_REUSE, ret=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Classification and localization predictions
        preds_conf = []  # conf -> classification b/c confidence loss -> classification loss
        preds_loc = []

        # Use batch normalization for all convolution layers
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': True},
                            weights_regularizer=slim.l2_regularizer(scale=REG_SCALE)):
            ''' alex 1 '''
            net = slim.conv2d(
                x, channels, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

            ''' alex 2 '''
            net = slim.conv2d(net, channels * 2, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

            # added for SSD
            feat1 = net
            net_conf, net_loc = SSDHook(net, 'conv2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)
            ###

            ''' alex 3, 4, 5 '''
            net = slim.conv2d(net, channels * 2, [3, 3], scope='conv3')
            net = slim.conv2d(net, channels * 2, [3, 3], scope='conv4')
            net = slim.conv2d(net, channels, [3, 3], scope='conv5')

            ''' 8 '''
            net = slim.conv2d(net, channels, [1, 1], scope='conv8')
            net = slim.conv2d(net, channels * 2, [3, 3], 2, scope='conv8_2')

            # added for SSD
            feat2 = net
            net_conf, net_loc = SSDHook(net, 'conv8_2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

        # Concatenate all preds together into 1 vector, for both classification and localization predictions
        final_pred_conf = tf.concat(preds_conf, axis=1)
        probs = tf.sigmoid(final_pred_conf)

        final_pred_loc = tf.concat(preds_loc, axis=1)

        if ret:
            return feat1, feat2
        else:
            return probs, final_pred_loc


def AlexNet2(x, scope='newroi', channels=32, reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Classification and localization predictions
        preds_conf = []  # conf -> classification b/c confidence loss -> classification loss
        preds_loc = []

        # Use batch normalization for all convolution layers
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': True},
                            weights_regularizer=slim.l2_regularizer(scale=REG_SCALE)):
            ''' alex 1 '''
            net = slim.conv2d(
                x, channels, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

            ''' alex 2 '''
            net = slim.conv2d(net, channels*2, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            # net_act1 = net

            # # added for SSD
            # net_conf, net_loc = SSDHook(net, 'conv2')
            # preds_conf.append(net_conf)
            # preds_loc.append(net_loc)
            # ###

            ''' alex 3, 4, 5 '''
            net = slim.conv2d(net, channels*2, [3, 3], scope='conv3')
            net = slim.conv2d(net, channels*2, [3, 3], scope='conv4')
            net = slim.conv2d(net, channels, [3, 3], 2, scope='conv5')

            net_conf, net_loc = SSDHook(net, 'conv5')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

            ''' 6 '''
            net = slim.conv2d(net, channels, [1, 1], scope='conv6')
            net = slim.conv2d(net, channels*2, [3, 3], 2, scope='conv6_2')

            # added for SSD
            net_conf, net_loc = SSDHook(net, 'conv6_2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

            ''' 7 '''
            net = slim.conv2d(net, channels, [1, 1], scope='conv7')
            net = slim.conv2d(net, channels*2, [3, 3], 2, scope='conv7_2')

            # added for SSD
            net_conf, net_loc = SSDHook(net, 'conv7_2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

        # Concatenate all preds together into 1 vector, for both classification and localization predictions
        final_pred_conf = tf.concat(preds_conf, axis=1)
        probs = tf.sigmoid(final_pred_conf)

        final_pred_loc = tf.concat(preds_loc, axis=1)
        return probs, final_pred_loc


''' nms processing '''


def calc_iou(box_a, box_b):
    """
    Calculate the Intersection Over Union of two boxes
    Each box specified by upper left corner and lower right corner:
    (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner

    Returns IOU value
    """
    # Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
    # http://math.stackexchange.com/a/99576
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap

    # Calculate union
    area_box_a = np.abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_box_b = np.abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou


def nms(y_pred_conf, y_pred_loc, prob, extend=1):
    class_boxes = {}  # class -> [(x1, y1, x2, y2, prob), (...), ...]
    class_boxes[1] = []

    y_idx = 0
    for fm_size in FM_SIZES:
        fm_h, fm_w = fm_size  # feature map height and width
        for row in range(fm_h):
            for col in range(fm_w):
                # Only perform calculations if class confidence > CONF_THRESH and not background class
                if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0.:
                    # Calculate absolute coordinates of predicted bounding box
                    xc, yc = col + 0.5, row + 0.5  # center of current feature map cell
                    center_coords = np.array([xc, yc, xc, yc])
                    abs_box_coords = center_coords + extend * y_pred_loc[
                        y_idx * 4: y_idx * 4 + 4]  # predictions are offsets to center of fm cell

                    # Calculate predicted box coordinates in actual image
                    scale = np.array(
                        [IMG_W / fm_w, IMG_H / fm_h, IMG_W / fm_w, IMG_H / fm_h])
                    box_coords = abs_box_coords * scale
                    box_coords = [int(x) for x in box_coords]
                    # Compare this box to all previous boxes of this class
                    cls = y_pred_conf[y_idx]
                    cls_prob = prob[y_idx]
                    box = (*box_coords, cls, cls_prob)

                    '''debug'''
                    box_coords = [int(round(x)) for x in box[:4]]
                    cls = int(box[4])
                    cls_prob = box[5]

                    # # Annotate image
                    # rgb = cv2.rectangle(rgb, tuple(box_coords[:2]), tuple(box_coords[2:]), (0, 255, 0))
                    # label_str = '%s %.2f' % ('knee', cls_prob)
                    # rgb = cv2.putText(rgb, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0, 255, 0), 1,
                    #                   cv2.LINE_AA)
                    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('debug', bgr)
                    # cv2.waitKey(0)

                    if len(class_boxes[cls]) == 0:
                        # if empty, directly append
                        class_boxes[cls].append(box)
                    else:
                        # did this box suppress other box(es)?
                        suppressed = False
                        # did this box overlap with other box(es)?
                        overlapped = False
                        # for other_box in class_boxes[cls]:
                        #     if box[5] > other_box[5]:
                        #         class_boxes[cls].remove(other_box)
                        #         suppressed = True
                        # if suppressed:
                        #     class_boxes[cls].append(box)
                        for other_box in class_boxes[cls]:
                            iou = calc_iou(box[:4], other_box[:4])
                            # if overlapped and possibility higher
                            if iou > NMS_IOU_THRESH:
                                overlapped = True
                                # If current box has higher confidence than other box
                                if box[5] > other_box[5]:  # if overlapped and has higher possibility
                                    class_boxes[cls].remove(other_box)
                                    suppressed = True
                            # else:
                            #     class_boxes[cls].append(box)  # if empty, directly append
                        if suppressed or not overlapped:
                            class_boxes[cls].append(box)

                y_idx += 1

    maxprob = 0
    boxes = None
    for cls in class_boxes.keys():
        for class_box in class_boxes[cls]:
            if maxprob < class_box[-1]:
                boxes = np.array(class_box)
                maxprob = class_box[-1]
    boxes = np.asarray(boxes)
    return boxes
