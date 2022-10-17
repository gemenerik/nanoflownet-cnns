import math

import tensorflow as tf


__all__ = ["flownet2s"]
FILTER_FACTOR = 3 / 8


def FlowNet2s(input_height, input_width, training):
    inputs = tf.keras.Input((None, None, 2), batch_size=None, name="input_1")

    conv1 = tf.keras.layers.Conv2D(
        int(64 * FILTER_FACTOR),
        (7, 7),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv1",
    )(inputs)

    conv2 = tf.keras.layers.Conv2D(
        int(128 * FILTER_FACTOR),
        (5, 5),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv2",
    )(conv1)

    conv3 = tf.keras.layers.Conv2D(
        int(256 * FILTER_FACTOR),
        (5, 5),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv3",
    )(conv2)

    conv3_1 = tf.keras.layers.Conv2D(
        int(256 * FILTER_FACTOR),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv3_1",
    )(conv3)

    conv4 = tf.keras.layers.Conv2D(
        int(512 * FILTER_FACTOR),
        (3, 3),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv4",
    )(conv3_1)

    conv4_1 = tf.keras.layers.Conv2D(
        int(512 * FILTER_FACTOR),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv4_1",
    )(conv4)

    conv5 = tf.keras.layers.Conv2D(
        int(512 * FILTER_FACTOR),
        (3, 3),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv5",
    )(conv4_1)

    conv5_1 = tf.keras.layers.Conv2D(
        int(512 * FILTER_FACTOR),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
        name="conv5_1",
    )(conv5)

    # disabled deepest layers to maintain reasonable feature sizes, resulting in FlowNet2-xs

    # conv6 = tf.keras.layers.Conv2D(
    #     int(1024 * FILTER_FACTOR),
    #     (3, 3),
    #     strides=2,
    #     padding="SAME",
    #     activation="relu",
    #     use_bias=True,
    #     name="conv6",
    # )(conv5_1)

    # conv6_1 = tf.keras.layers.Conv2D(
    #     int(1024 * FILTER_FACTOR),
    #     (3, 3),
    #     strides=1,
    #     padding="SAME",
    #     activation="relu",
    #     use_bias=True,
    #     name="conv6_1",
    # )(conv6)

    # flow_pred_6 = tf.keras.layers.Conv2D(
    #     2,
    #     (3, 3),
    #     strides=1,
    #     padding="SAME",
    #     activation=None,
    #     use_bias=True,
    # )(conv6_1)

    # flow_pred_6_up = tf.keras.layers.Conv2DTranspose(
    #     2,
    #     (4, 4),
    #     strides=2,
    #     padding="SAME",
    #     activation=None,
    #     use_bias=True,
    # )(flow_pred_6)

    # upconv5 = tf.keras.layers.Conv2DTranspose(
    #     int(512 * FILTER_FACTOR),
    #     (4, 4),
    #     strides=2,
    #     padding="SAME",
    #     activation="relu",
    #     use_bias=True,
    # )(conv6_1)

    # concat5 = tf.keras.layers.concatenate([conv5_1, upconv5, flow_pred_6_up], axis=-1)

    flow_pred_5 = tf.keras.layers.Conv2D(
        2,
        (3, 3),
        strides=1,
        padding="SAME",
        activation=None,
        use_bias=True,
        name="flow_5",
    )(conv5_1)

    flow_pred_5_up = tf.keras.layers.Conv2DTranspose(
        2,
        (4, 4),
        strides=2,
        padding="SAME",
        activation=None,
        use_bias=True,
    )(flow_pred_5)

    upconv4 = tf.keras.layers.Conv2DTranspose(
        int(256 * FILTER_FACTOR),
        (4, 4),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(conv5_1)

    concat4 = tf.keras.layers.concatenate([conv4_1, upconv4, flow_pred_5_up], axis=-1)

    flow_pred_4 = tf.keras.layers.Conv2D(
        2,
        (3, 3),
        strides=1,
        padding="SAME",
        activation=None,
        use_bias=True,
        name="flow_4",
    )(concat4)

    flow_pred_4_up = tf.keras.layers.Conv2DTranspose(
        2,
        (4, 4),
        strides=2,
        padding="SAME",
        activation=None,
        use_bias=True,
    )(flow_pred_4)

    upconv3 = tf.keras.layers.Conv2DTranspose(
        int(128 * FILTER_FACTOR),
        (4, 4),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(concat4)

    concat3 = tf.keras.layers.concatenate([conv3_1, upconv3, flow_pred_4_up], axis=-1)

    flow_pred_3 = tf.keras.layers.Conv2D(
        2,
        (3, 3),
        strides=1,
        padding="SAME",
        activation=None,
        use_bias=True,
        name="flow_3",
    )(concat3)

    flow_pred_3_up = tf.keras.layers.Conv2DTranspose(
        2,
        (4, 4),
        strides=2,
        padding="SAME",
        activation=None,
        use_bias=True,
    )(flow_pred_3)

    upconv2 = tf.keras.layers.Conv2DTranspose(
        int(64 * FILTER_FACTOR),
        (4, 4),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(concat3)

    concat2 = tf.keras.layers.concatenate([conv2, upconv2, flow_pred_3_up], axis=-1)

    flow_pred_2 = tf.keras.layers.Conv2D(
        2,
        (3, 3),
        strides=1,
        padding="SAME",
        activation=None,
        use_bias=True,
        name="output",
    )(concat2)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[flow_pred_2, flow_pred_3, flow_pred_4, flow_pred_5]
        if training
        else [flow_pred_2],
    )
    return model


def flownet2s(batch_size, input_height, input_width, training=True):
    model = FlowNet2s(input_height, input_width, training)
    return model
