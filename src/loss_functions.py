import tensorflow as tf
import tensorflow_addons as tfa


def edge_detail_aggregate_loss(targets, network_output, weights=None, sparse=False):
    targets = targets[:, :, :, 0:2]
    laplacian_kernel = tf.constant(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32
    )
    laplacian_kernel = tf.stack([laplacian_kernel, laplacian_kernel], axis=-1)
    laplacian_kernel = tf.expand_dims(laplacian_kernel, -1)

    boundary_targets = tf.nn.conv2d(
        targets, laplacian_kernel, strides=1, padding="SAME"
    )
    boundary_targets = tf.clip_by_value(
        boundary_targets, clip_value_min=0, clip_value_max=1
    )
    boundary_targets = tf.cast(boundary_targets > 0.1, tf.float32)

    boundary_targets_x2 = tf.nn.conv2d(
        targets, laplacian_kernel, strides=2, padding="SAME"
    )
    boundary_targets_x2 = tf.clip_by_value(
        boundary_targets_x2, clip_value_min=0, clip_value_max=1
    )
    boundary_targets_x2_up = tf.keras.layers.UpSampling2D(size=(2, 2))(
        boundary_targets_x2
    )
    boundary_targets_x2_up = tf.cast(boundary_targets_x2_up > 0.1, tf.float32)

    boundary_targets_x4 = tf.nn.conv2d(
        targets, laplacian_kernel, strides=4, padding="SAME"
    )
    boundary_targets_x4 = tf.clip_by_value(
        boundary_targets_x4, clip_value_min=0, clip_value_max=1
    )
    boundary_targets_x4_up = tf.keras.layers.UpSampling2D(size=(4, 4))(
        boundary_targets_x4
    )
    boundary_targets_x4_up = tf.cast(boundary_targets_x4_up > 0.1, tf.float32)

    boundary_targets_pyramid = tf.concat(
        [boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up],
        axis=-1,
    )

    fuse_kernel = tf.constant([[6.0 / 10], [3.0 / 10], [1.0 / 10]], dtype=tf.float32)
    fuse_kernel = tf.reshape(fuse_kernel, [1, 1, 3, 1])

    boundary_targets_pyramid = tf.nn.conv2d(
        boundary_targets_pyramid, fuse_kernel, strides=1, padding="SAME"
    )

    boundary_targets_pyramid = tf.cast(boundary_targets_pyramid > 0.1, tf.float32)

    sigmoidfocalcrossentropy = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)(
        boundary_targets_pyramid, network_output
    )

    return sigmoidfocalcrossentropy


def mb_detail_aggregate_loss(targets, network_output, weights=None, sparse=False):
    target_mb = tf.slice(targets, [0, 0, 0, 2], [-1, -1, -1, 1])

    sigmoidfocalcrossentropy = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)(
        target_mb, network_output
    )
    return sigmoidfocalcrossentropy


def no_loss(targets, network_output, weights=None):
    return 1


def end_point_error(input_flow, target_flow):
    return tf.reduce_mean(
        tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(target_flow, input_flow)), axis=-1))
    )


def scale_output_to_target(output, target):
    b_target, h_target, w_target, _ = target.get_shape()
    output_scaled = tf.image.resize(
        output, (h_target, w_target), method=tf.image.ResizeMethod.BILINEAR
    )
    return end_point_error(output_scaled, target)
    

def multi_scale_end_point_error(targets, network_output, weights=None, sparse=False):
    target_flow = tf.slice(targets, [0, 0, 0, 0], [-1, -1, -1, 2])

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [1]
    assert len(weights) == len(network_output)

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * scale_output_to_target(output, target_flow, sparse)
    return loss
