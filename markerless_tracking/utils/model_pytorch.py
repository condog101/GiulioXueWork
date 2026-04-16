"""
PyTorch model definitions for markerless tracking.
Replaces the TensorFlow-based model.py and tf_util.py.

Contains:
- AlexNet2: SSD-style RGB bounding box detector
- PointNetSeg: PointNet-based point cloud segmentation
- nms / calc_iou: numpy post-processing (unchanged from TF version)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.settings import (
    NUM_PRED_CONF, NUM_PRED_LOC, FM_SIZES, IMG_H, IMG_W,
    CONF_THRESH, NMS_IOU_THRESH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TFSamePadConv2d(nn.Module):
    """Conv2d that replicates TensorFlow 'SAME' padding (asymmetric for stride>1).
    No bias — used with batch norm which absorbs the bias."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=False)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x):
        ih, iw = x.shape[2], x.shape[3]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + self.kernel_size[0] - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + self.kernel_size[1] - iw, 0)
        # TF pads more on right/bottom
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                       pad_h // 2, pad_h - pad_h // 2])
        return self.conv(x)


# ---------------------------------------------------------------------------
# AlexNet2 — RGB ROI detection (SSD-style)
# ---------------------------------------------------------------------------

class SSDHook(nn.Module):
    """SSD detection head: 3x3 conv + BN for confidence and localization.
    No activation (original uses activation_fn=None inside slim arg_scope
    that still applies batch_norm)."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv_conf = nn.Conv2d(in_channels, NUM_PRED_CONF, 3, padding=1, bias=False)
        self.bn_conf = nn.BatchNorm2d(NUM_PRED_CONF, eps=1e-3, momentum=0.1)
        self.conv_loc = nn.Conv2d(in_channels, NUM_PRED_LOC, 3, padding=1, bias=False)
        self.bn_loc = nn.BatchNorm2d(NUM_PRED_LOC, eps=1e-3, momentum=0.1)

    def forward(self, x):
        # Permute NCHW -> NHWC before flatten to match TF's flatten ordering.
        # Critical for loc (C=4): NMS expects 4 values per spatial position adjacent.
        conf = self.bn_conf(self.conv_conf(x)).permute(0, 2, 3, 1).contiguous().flatten(1)
        loc = self.bn_loc(self.conv_loc(x)).permute(0, 2, 3, 1).contiguous().flatten(1)
        return conf, loc


class AlexNet2(nn.Module):
    """
    Modified AlexNet with SSD multi-scale detection heads.

    Input:  [B, 3, 512, 512]  (NCHW)
    Output: probs [B, N], preds_loc [B, N*4]

    NOTE: The original TF code runs batch-norm with is_training=True even at
    inference time. To preserve this behaviour, call model.train() before
    inference (do NOT call model.eval()).

    slim.batch_norm defaults: scale=False (no gamma), center=True (beta).
    slim.conv2d with normalizer_fn disables conv biases.
    """

    def __init__(self, channels=32):
        super().__init__()
        ch = channels

        # --- backbone (all convs bias=False, BN absorbs bias) ---
        self.conv1 = nn.Conv2d(3, ch, 11, stride=4, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.1)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(ch, ch * 2, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(ch * 2, eps=1e-3, momentum=0.1)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(ch * 2, ch * 2, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch * 2, eps=1e-3, momentum=0.1)

        self.conv4 = nn.Conv2d(ch * 2, ch * 2, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ch * 2, eps=1e-3, momentum=0.1)

        # stride-2 with TF SAME padding
        self.conv5 = TFSamePadConv2d(ch * 2, ch, 3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.1)

        self.conv6 = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.1)

        self.conv6_2 = TFSamePadConv2d(ch, ch * 2, 3, stride=2)
        self.bn6_2 = nn.BatchNorm2d(ch * 2, eps=1e-3, momentum=0.1)

        self.conv7 = nn.Conv2d(ch * 2, ch, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.1)

        self.conv7_2 = TFSamePadConv2d(ch, ch * 2, 3, stride=2)
        self.bn7_2 = nn.BatchNorm2d(ch * 2, eps=1e-3, momentum=0.1)

        # --- SSD heads ---
        self.ssd_hook1 = SSDHook(ch)       # after conv5 (15x15)
        self.ssd_hook2 = SSDHook(ch * 2)   # after conv6_2 (8x8)
        self.ssd_hook3 = SSDHook(ch * 2)   # after conv7_2 (4x4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        conf1, loc1 = self.ssd_hook1(x)

        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn6_2(self.conv6_2(x)))

        conf2, loc2 = self.ssd_hook2(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn7_2(self.conv7_2(x)))

        conf3, loc3 = self.ssd_hook3(x)

        final_conf = torch.cat([conf1, conf2, conf3], dim=1)
        probs = torch.sigmoid(final_conf)

        final_loc = torch.cat([loc1, loc2, loc3], dim=1)
        return probs, final_loc


# ---------------------------------------------------------------------------
# PointNet Segmentation
# ---------------------------------------------------------------------------

class TransformNet(nn.Module):
    """
    Spatial Transformer Network (T-Net) for PointNet.
    Produces a KxK transformation matrix.
    """

    def __init__(self, K=3):
        super().__init__()
        self.K = K

        self.conv1 = nn.Conv1d(K, 64, 1)
        self.bn1 = nn.BatchNorm1d(64, eps=1e-3, momentum=0.1)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-3, momentum=0.1)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024, eps=1e-3, momentum=0.1)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-3, momentum=0.1)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256, eps=1e-3, momentum=0.1)

        self.fc3 = nn.Linear(256, K * K)

        # Initialize fc3 to output identity by default
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc3.bias.data += torch.eye(K).flatten()

    def forward(self, x):
        # x: [B, K, N]
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]  # [B, 1024]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # [B, K*K]
        # Add identity
        identity = torch.eye(self.K, device=x.device, dtype=x.dtype).flatten().unsqueeze(0)
        x = x + identity
        return x.view(B, self.K, self.K)


class PointNetSeg(nn.Module):
    """
    PointNet segmentation network for point cloud.

    Input:  [B, N, 3]  (point cloud)
    Output: [B, N]     (per-point sigmoid probability)

    NOTE: The original TF code runs with is_training=False at inference,
    so call model.eval() before inference.
    """

    def __init__(self, num_point=2000):
        super().__init__()
        self.num_point = num_point

        # Input transform (3x3)
        self.input_transform = TransformNet(K=3)

        # First two convolutions (after input transform)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64, eps=1e-3, momentum=0.1)

        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-3, momentum=0.1)

        # Feature transform (64x64)
        self.feature_transform = TransformNet(K=64)

        # Post-transform convolutions
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64, eps=1e-3, momentum=0.1)

        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-3, momentum=0.1)

        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024, eps=1e-3, momentum=0.1)

        # Segmentation head (input: 64 point_feat + 1024 global = 1088)
        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.bn6 = nn.BatchNorm1d(512, eps=1e-3, momentum=0.1)

        self.conv7 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256, eps=1e-3, momentum=0.1)

        self.conv8 = nn.Conv1d(256, 128, 1)
        self.bn8 = nn.BatchNorm1d(128, eps=1e-3, momentum=0.1)

        self.conv9 = nn.Conv1d(128, 128, 1)
        self.bn9 = nn.BatchNorm1d(128, eps=1e-3, momentum=0.1)

        self.conv10 = nn.Conv1d(128, 1, 1)

    def forward(self, point_cloud):
        # point_cloud: [B, N, 3]
        B, N, _ = point_cloud.shape
        x = point_cloud.permute(0, 2, 1)  # [B, 3, N]

        # Input transform
        transform = self.input_transform(x)              # [B, 3, 3]
        x = torch.bmm(point_cloud, transform)            # [B, N, 3]
        x = x.permute(0, 2, 1)                           # [B, 3, N]

        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))               # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))               # [B, 64, N]

        # Feature transform
        feat_transform = self.feature_transform(x)        # [B, 64, 64]
        x = x.permute(0, 2, 1)                            # [B, N, 64]
        x = torch.bmm(x, feat_transform)                  # [B, N, 64]
        point_feat = x.permute(0, 2, 1)                   # [B, 64, N]

        # Global feature
        x = F.relu(self.bn3(self.conv3(point_feat)))      # [B, 64, N]
        x = F.relu(self.bn4(self.conv4(x)))               # [B, 128, N]
        x = F.relu(self.bn5(self.conv5(x)))               # [B, 1024, N]
        global_feat = x.max(dim=2, keepdim=True)[0]       # [B, 1024, 1]
        global_feat = global_feat.expand(-1, -1, N)       # [B, 1024, N]

        # Concatenate point + global features
        x = torch.cat([point_feat, global_feat], dim=1)   # [B, 1088, N]

        # Segmentation head
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.conv10(x)                                # [B, 1, N]
        x = x.squeeze(1)                                  # [B, N]
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# NMS post-processing (pure numpy, unchanged from TF version)
# ---------------------------------------------------------------------------

def calc_iou(box_a, box_b):
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap
    area_box_a = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_box_b = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_box_a + area_box_b - intersection
    return intersection / union


def nms(y_pred_conf, y_pred_loc, prob, extend=1):
    class_boxes = {1: []}
    y_idx = 0
    for fm_size in FM_SIZES:
        fm_h, fm_w = fm_size
        for row in range(fm_h):
            for col in range(fm_w):
                if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0.:
                    xc, yc = col + 0.5, row + 0.5
                    center_coords = np.array([xc, yc, xc, yc])
                    abs_box_coords = center_coords + extend * y_pred_loc[
                        y_idx * 4: y_idx * 4 + 4]
                    scale = np.array(
                        [IMG_W / fm_w, IMG_H / fm_h, IMG_W / fm_w, IMG_H / fm_h])
                    box_coords = abs_box_coords * scale
                    box_coords = [int(x) for x in box_coords]
                    cls = y_pred_conf[y_idx]
                    cls_prob = prob[y_idx]
                    box = (*box_coords, cls, cls_prob)

                    box_coords = [int(round(x)) for x in box[:4]]
                    cls = int(box[4])
                    cls_prob = box[5]

                    if len(class_boxes[cls]) == 0:
                        class_boxes[cls].append(box)
                    else:
                        suppressed = False
                        overlapped = False
                        for other_box in class_boxes[cls]:
                            iou = calc_iou(box[:4], other_box[:4])
                            if iou > NMS_IOU_THRESH:
                                overlapped = True
                                if box[5] > other_box[5]:
                                    class_boxes[cls].remove(other_box)
                                    suppressed = True
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
