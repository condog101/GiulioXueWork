#!/usr/bin/env python3
"""
One-time weight conversion script: TensorFlow checkpoints -> PyTorch .pt files.

Run this in the old TF/Python 3.6 conda environment (xue_env):
    conda run -n xue_env python convert_weights.py

Produces:
    ckpt/rgb/alexnet2.pt
    ckpt/pcl/pointnet_seg.pt
"""
import os
import sys
import numpy as np

# Need TF to read checkpoints
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# We save as numpy archives (.npz) since the old env may not have PyTorch.
# A second step (or the new env) converts .npz -> .pt using torch.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_tf_var(ckpt_path, name):
    """Load a single variable from a TF checkpoint."""
    return tf.train.load_variable(ckpt_path, name)


def convert_rgb_model(ckpt_dir):
    """Convert AlexNet2 RGB model weights from TF checkpoint to PyTorch state_dict."""
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    print(f"Converting RGB model from: {ckpt_path}")

    state_dict = {}

    # Backbone conv layers
    # slim.conv2d: no biases (absorbed by BN), weights are [H,W,Cin,Cout]
    # slim.batch_norm: scale=False (no gamma), center=True (beta)
    # PyTorch: conv weight [Cout,Cin,H,W], BN weight=gamma(1.0), bias=beta

    backbone_layers = {
        'conv1': 'conv1',
        'conv2': 'conv2',
        'conv3': 'conv3',
        'conv4': 'conv4',
        'conv5': 'conv5.conv',  # TFSamePadConv2d wraps nn.Conv2d as .conv
        'conv6': 'conv6',
        'conv6_2': 'conv6_2.conv',
        'conv7': 'conv7',
        'conv7_2': 'conv7_2.conv',
    }

    bn_layers = {
        'conv1': 'bn1',
        'conv2': 'bn2',
        'conv3': 'bn3',
        'conv4': 'bn4',
        'conv5': 'bn5',
        'conv6': 'bn6',
        'conv6_2': 'bn6_2',
        'conv7': 'bn7',
        'conv7_2': 'bn7_2',
    }

    for tf_name, pt_conv in backbone_layers.items():
        # Conv weights: [H,W,Cin,Cout] -> [Cout,Cin,H,W]
        w = load_tf_var(ckpt_path, f'newroi/{tf_name}/weights')
        state_dict[f'{pt_conv}.weight'] = np.transpose(w, (3, 2, 0, 1))

        # BN: no gamma (scale=False), load beta, moving_mean, moving_variance
        pt_bn = bn_layers[tf_name]
        beta = load_tf_var(ckpt_path, f'newroi/{tf_name}/BatchNorm/beta')
        moving_mean = load_tf_var(ckpt_path, f'newroi/{tf_name}/BatchNorm/moving_mean')
        moving_var = load_tf_var(ckpt_path, f'newroi/{tf_name}/BatchNorm/moving_variance')
        num_features = beta.shape[0]

        state_dict[f'{pt_bn}.weight'] = np.ones(num_features, dtype=np.float32)  # gamma=1
        state_dict[f'{pt_bn}.bias'] = beta
        state_dict[f'{pt_bn}.running_mean'] = moving_mean
        state_dict[f'{pt_bn}.running_var'] = moving_var
        state_dict[f'{pt_bn}.num_batches_tracked'] = np.array(0, dtype=np.int64)

    # SSD hooks
    ssd_hooks = {
        'ssd_hook_conv5': 'ssd_hook1',
        'ssd_hook_conv6_2': 'ssd_hook2',
        'ssd_hook_conv7_2': 'ssd_hook3',
    }

    for tf_hook, pt_hook in ssd_hooks.items():
        for head in ['conf', 'loc']:
            tf_prefix = f'newroi/{tf_hook}/conv_{head}'
            pt_conv = f'{pt_hook}.conv_{head}'
            pt_bn = f'{pt_hook}.bn_{head}'

            # Conv weights
            w = load_tf_var(ckpt_path, f'{tf_prefix}/weights')
            state_dict[f'{pt_conv}.weight'] = np.transpose(w, (3, 2, 0, 1))

            # BN (no gamma)
            beta = load_tf_var(ckpt_path, f'{tf_prefix}/BatchNorm/beta')
            moving_mean = load_tf_var(ckpt_path, f'{tf_prefix}/BatchNorm/moving_mean')
            moving_var = load_tf_var(ckpt_path, f'{tf_prefix}/BatchNorm/moving_variance')
            num_features = beta.shape[0]

            state_dict[f'{pt_bn}.weight'] = np.ones(num_features, dtype=np.float32)
            state_dict[f'{pt_bn}.bias'] = beta
            state_dict[f'{pt_bn}.running_mean'] = moving_mean
            state_dict[f'{pt_bn}.running_var'] = moving_var
            state_dict[f'{pt_bn}.num_batches_tracked'] = np.array(0, dtype=np.int64)

    return state_dict


def convert_pcl_model(ckpt_dir):
    """Convert PointNet PCL model weights from TF checkpoint to PyTorch state_dict."""
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    print(f"Converting PCL model from: {ckpt_path}")

    state_dict = {}

    def conv_weight_to_pt(w):
        """TF conv2d [1, kw, Cin, Cout] -> PyTorch Conv1d [Cout, Cin, 1].
        For the first conv layers where kw=3 and Cin=1, this collapses
        the spatial dim into channels: [1, 3, 1, 64] -> [64, 3, 1]."""
        # Squeeze height dim (always 1)
        w = w.squeeze(axis=0)  # [kw, Cin, Cout]
        # If kw > 1, it spans the input coordinates (treated as channels in Conv1d)
        if w.shape[0] > 1:
            # [kw, Cin, Cout] where Cin=1 -> [kw, Cout] -> [Cout, kw, 1]
            w = w.squeeze(axis=1)  # [kw, Cout]
            w = w.T[:, :, np.newaxis]  # [Cout, kw, 1]
        else:
            # [1, Cin, Cout] -> [Cin, Cout]
            w = w.squeeze(axis=0)  # [Cin, Cout]
            w = w.T[:, :, np.newaxis]  # [Cout, Cin, 1]
        return w

    def fc_weight_to_pt(w):
        """TF FC [in, out] -> PyTorch Linear [out, in]."""
        return w.T

    def load_conv_bn(tf_prefix, pt_conv, pt_bn):
        """Load conv + BN layer pair for PointNet (has both gamma and beta)."""
        w = load_tf_var(ckpt_path, f'{tf_prefix}/weights')
        b = load_tf_var(ckpt_path, f'{tf_prefix}/biases')
        state_dict[f'{pt_conv}.weight'] = conv_weight_to_pt(w)
        state_dict[f'{pt_conv}.bias'] = b

        gamma = load_tf_var(ckpt_path, f'{tf_prefix}/bn/gamma')
        beta = load_tf_var(ckpt_path, f'{tf_prefix}/bn/beta')
        # EMA shadow variables for running mean/var
        ema_mean = load_tf_var(ckpt_path,
            f'{tf_prefix}/bn/{tf_prefix}/bn/moments/Squeeze/ExponentialMovingAverage')
        ema_var = load_tf_var(ckpt_path,
            f'{tf_prefix}/bn/{tf_prefix}/bn/moments/Squeeze_1/ExponentialMovingAverage')

        state_dict[f'{pt_bn}.weight'] = gamma
        state_dict[f'{pt_bn}.bias'] = beta
        state_dict[f'{pt_bn}.running_mean'] = ema_mean
        state_dict[f'{pt_bn}.running_var'] = ema_var
        state_dict[f'{pt_bn}.num_batches_tracked'] = np.array(0, dtype=np.int64)

    def load_fc_bn(tf_prefix, pt_fc, pt_bn):
        """Load FC + BN layer pair."""
        w = load_tf_var(ckpt_path, f'{tf_prefix}/weights')
        b = load_tf_var(ckpt_path, f'{tf_prefix}/biases')
        state_dict[f'{pt_fc}.weight'] = fc_weight_to_pt(w)
        state_dict[f'{pt_fc}.bias'] = b

        gamma = load_tf_var(ckpt_path, f'{tf_prefix}/bn/gamma')
        beta = load_tf_var(ckpt_path, f'{tf_prefix}/bn/beta')
        ema_mean = load_tf_var(ckpt_path,
            f'{tf_prefix}/bn/{tf_prefix}/bn/moments/Squeeze/ExponentialMovingAverage')
        ema_var = load_tf_var(ckpt_path,
            f'{tf_prefix}/bn/{tf_prefix}/bn/moments/Squeeze_1/ExponentialMovingAverage')

        state_dict[f'{pt_bn}.weight'] = gamma
        state_dict[f'{pt_bn}.bias'] = beta
        state_dict[f'{pt_bn}.running_mean'] = ema_mean
        state_dict[f'{pt_bn}.running_var'] = ema_var
        state_dict[f'{pt_bn}.num_batches_tracked'] = np.array(0, dtype=np.int64)

    # --- Input Transform Net (K=3) ---
    tnet1_prefix = 'newseg/transform_net1'
    load_conv_bn(f'{tnet1_prefix}/tconv1', 'input_transform.conv1', 'input_transform.bn1')
    load_conv_bn(f'{tnet1_prefix}/tconv2', 'input_transform.conv2', 'input_transform.bn2')
    load_conv_bn(f'{tnet1_prefix}/tconv3', 'input_transform.conv3', 'input_transform.bn3')
    load_fc_bn(f'{tnet1_prefix}/tfc1', 'input_transform.fc1', 'input_transform.bn4')
    load_fc_bn(f'{tnet1_prefix}/tfc2', 'input_transform.fc2', 'input_transform.bn5')

    # Transform XYZ final linear (biases stored as zeros, identity added in graph)
    w = load_tf_var(ckpt_path, f'{tnet1_prefix}/transform_XYZ/weights')
    b = load_tf_var(ckpt_path, f'{tnet1_prefix}/transform_XYZ/biases')
    identity_3 = np.eye(3).flatten().astype(np.float32)
    state_dict['input_transform.fc3.weight'] = fc_weight_to_pt(w)
    state_dict['input_transform.fc3.bias'] = b + identity_3  # Add identity!

    # --- Main conv layers ---
    load_conv_bn('newseg/conv1', 'conv1', 'bn1')
    load_conv_bn('newseg/conv2', 'conv2', 'bn2')

    # --- Feature Transform Net (K=64) ---
    tnet2_prefix = 'newseg/transform_net2'
    load_conv_bn(f'{tnet2_prefix}/tconv1', 'feature_transform.conv1', 'feature_transform.bn1')
    load_conv_bn(f'{tnet2_prefix}/tconv2', 'feature_transform.conv2', 'feature_transform.bn2')
    load_conv_bn(f'{tnet2_prefix}/tconv3', 'feature_transform.conv3', 'feature_transform.bn3')
    load_fc_bn(f'{tnet2_prefix}/tfc1', 'feature_transform.fc1', 'feature_transform.bn4')
    load_fc_bn(f'{tnet2_prefix}/tfc2', 'feature_transform.fc2', 'feature_transform.bn5')

    # Transform feat final linear
    w = load_tf_var(ckpt_path, f'{tnet2_prefix}/transform_feat/weights')
    b = load_tf_var(ckpt_path, f'{tnet2_prefix}/transform_feat/biases')
    identity_64 = np.eye(64).flatten().astype(np.float32)
    state_dict['feature_transform.fc3.weight'] = fc_weight_to_pt(w)
    state_dict['feature_transform.fc3.bias'] = b + identity_64  # Add identity!

    # --- Post-transform convolutions ---
    load_conv_bn('newseg/conv3', 'conv3', 'bn3')
    load_conv_bn('newseg/conv4', 'conv4', 'bn4')
    load_conv_bn('newseg/conv5', 'conv5', 'bn5')

    # --- Segmentation head ---
    load_conv_bn('newseg/conv6', 'conv6', 'bn6')
    load_conv_bn('newseg/conv7', 'conv7', 'bn7')
    load_conv_bn('newseg/conv8', 'conv8', 'bn8')
    load_conv_bn('newseg/conv9', 'conv9', 'bn9')

    # conv10: no BN, no activation
    w = load_tf_var(ckpt_path, 'newseg/conv10/weights')
    b = load_tf_var(ckpt_path, 'newseg/conv10/biases')
    state_dict['conv10.weight'] = conv_weight_to_pt(w)
    state_dict['conv10.bias'] = b

    return state_dict


def save_as_npz(state_dict, output_path):
    """Save state dict as .npz (compatible with any Python version)."""
    np.savez(output_path, **state_dict)
    print(f"Saved {len(state_dict)} tensors to {output_path}")


def npz_to_pt(npz_path, pt_path):
    """Convert .npz to PyTorch .pt file. Run this in the PyTorch environment."""
    try:
        import torch
    except ImportError:
        print(f"PyTorch not available. Saved as .npz at {npz_path}")
        print(f"In the PyTorch env, run: python convert_weights.py --npz-to-pt")
        return False

    data = np.load(npz_path)
    state_dict = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype == np.int64:
            state_dict[key] = torch.tensor(arr)
        else:
            state_dict[key] = torch.from_numpy(arr.copy())
    torch.save(state_dict, pt_path)
    print(f"Saved PyTorch state dict to {pt_path}")
    return True


def main():
    os.chdir(SCRIPT_DIR)

    if '--npz-to-pt' in sys.argv:
        # Second pass: convert .npz to .pt (run in PyTorch env)
        npz_to_pt('ckpt/rgb/alexnet2.npz', 'ckpt/rgb/alexnet2.pt')
        npz_to_pt('ckpt/pcl/pointnet_seg.npz', 'ckpt/pcl/pointnet_seg.pt')
        return

    # Convert RGB model
    rgb_state = convert_rgb_model('ckpt/rgb/')
    save_as_npz(rgb_state, 'ckpt/rgb/alexnet2.npz')

    # Convert PCL model
    pcl_state = convert_pcl_model('ckpt/pcl/')
    save_as_npz(pcl_state, 'ckpt/pcl/pointnet_seg.npz')

    # Try to also save as .pt if torch is available
    npz_to_pt('ckpt/rgb/alexnet2.npz', 'ckpt/rgb/alexnet2.pt')
    npz_to_pt('ckpt/pcl/pointnet_seg.npz', 'ckpt/pcl/pointnet_seg.pt')

    print("\nDone! Weight conversion complete.")
    print("If .pt files were not created (no PyTorch in this env),")
    print("run in the PyTorch env:  python convert_weights.py --npz-to-pt")


if __name__ == '__main__':
    main()
