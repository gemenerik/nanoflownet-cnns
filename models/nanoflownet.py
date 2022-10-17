import math
import tensorflow as tf


__all__ = ["nanoflownet"]

# batch_size = 1

FILTER_FACTOR = 2


def stdc_module(inputs, filters):
    convx_1 = tf.keras.layers.Conv2D(
        int(filters / 2),
        (1, 1),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(inputs)
    convx_2 = tf.keras.layers.SeparableConv2D(
        int(filters / 4),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_1)
    convx_3 = tf.keras.layers.SeparableConv2D(
        int(filters / 8),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_2)
    convx_4 = tf.keras.layers.SeparableConv2D(
        int(filters / 8),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_3)
    fusion = tf.keras.layers.Concatenate()([convx_1, convx_2, convx_3, convx_4])
    return fusion


def modified_strided_stdc_module(inputs, filters):
    convx_1 = tf.keras.layers.SeparableConv2D(
        int(filters / 2),
        (3, 3),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(inputs)
    pooled_convx1 = tf.keras.layers.AveragePooling2D((3, 3), padding="SAME", strides=2)(
        inputs
    )
    pooled_convx1 = tf.keras.layers.Conv2D(
        int(filters / 2),
        (1, 1),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(pooled_convx1)
    convx_2 = tf.keras.layers.SeparableConv2D(
        int(filters / 4),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_1)
    convx_3 = tf.keras.layers.SeparableConv2D(
        int(filters / 4),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_2)
    fusion = tf.keras.layers.Concatenate()([pooled_convx1, convx_2, convx_3])
    return fusion

def original_strided_stdc_module(inputs, filters):
    convx_1 = tf.keras.layers.SeparableConv2D(
        int(filters / 2),
        (1, 1),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(inputs)
    pooled_convx1 = tf.keras.layers.AveragePooling2D(
        (3, 3), padding="SAME", strides=2
    )(convx_1)
    convx_2 = tf.keras.layers.SeparableConv2D(
        int(filters / 4),
        (3, 3),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_1)
    convx_3 = tf.keras.layers.SeparableConv2D(
        int(filters / 8),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_2)
    convx_4 = tf.keras.layers.SeparableConv2D(
        int(filters / 8),
        (3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(convx_3)
    fusion = tf.keras.layers.Concatenate()([pooled_convx1, convx_2, convx_3, convx_4])
    return fusion


def stage(inputs, filters):
    stdcm1 = modified_strided_stdc_module(inputs, filters)

    stdcm2 = stdc_module(stdcm1, filters)

    return stdcm2


def attention_refinement_module(inputs, filters):
    conv = tf.keras.layers.SeparableConv2D(
        filters,
        kernel_size=(3, 3),
        strides=1,
        use_bias=True,
        padding="SAME",
        activation="relu",
    )(inputs)
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()(conv)
    global_pooling = tf.keras.layers.Reshape((1, 1, filters))(global_pooling)
    conv_attention = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        use_bias=True,
        padding="SAME",
        activation="sigmoid",
    )(global_pooling)
    out = tf.keras.layers.Multiply()([conv, conv_attention])
    return out


def feature_fusion_module(context_inputs, spatial_inputs, filters):
    featurescat = tf.keras.layers.Concatenate()([context_inputs, spatial_inputs])
    features = tf.keras.layers.Conv2D(
        filters, kernel_size=(1, 1), padding="SAME", use_bias=True, activation="relu"
    )(featurescat)

    attention = tf.keras.layers.GlobalAveragePooling2D()(features)
    attention = tf.keras.layers.Reshape((1, 1, filters))(attention)
    attention = tf.keras.layers.Conv2D(
        filters // 4,
        kernel_size=(1, 1),
        padding="SAME",
        use_bias=True,
        activation="relu",
    )(attention)
    attention = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding="SAME",
        use_bias=True,
        activation="sigmoid",
    )(attention)
    features_attention = tf.keras.layers.Multiply()([features, attention])
    features_out = tf.keras.layers.Add()([features_attention, features])
    return features_out


def flow_head(inputs, filters, output_filters, output_name):
    conv_out = tf.keras.layers.SeparableConv2D(
        filters,
        kernel_size=(3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(inputs)

    conv_out = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=(1, 1),
        strides=1,
        padding="SAME",
        use_bias=True,
        name=output_name,
    )(conv_out)
    return conv_out


def NanoFlowNet(input_height, input_width, training):
    inputs = tf.keras.Input((None, None, 2), batch_size=None, name="input_1")
    conv2 = tf.keras.layers.SeparableConv2D(
        int(8 * FILTER_FACTOR),
        (3, 3),
        strides=2,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(inputs)

    stage3 = stage(conv2, 32 * FILTER_FACTOR)
    stage3_out = flow_head(stage3, 8 * FILTER_FACTOR, 1, "stage3")
    stage3_out = tf.keras.layers.UpSampling2D(
        size=(4, 4),
        data_format="channels_last",
        interpolation="bilinear",
        name="stage3_out",
    )(stage3_out)

    stage4 = stage(stage3, 64 * FILTER_FACTOR)

    stage5 = stage(stage4, 128 * FILTER_FACTOR)

    avg = tf.keras.layers.GlobalAveragePooling2D()(stage5)
    avg = tf.keras.layers.Reshape((1, 1, 128 * FILTER_FACTOR))(avg)
    avg = tf.keras.layers.Conv2D(
        int(16 * FILTER_FACTOR),
        kernel_size=(1, 1),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(avg)

    stage5_arm = attention_refinement_module(stage5, 16 * FILTER_FACTOR)
    stage5_arm_sum = tf.keras.layers.Multiply()([stage5_arm, avg])
    stage5_arm_up = tf.keras.layers.UpSampling2D(
        size=(2, 2), data_format="channels_last", interpolation="bilinear"
    )(stage5_arm_sum)

    stage5_arm_up = tf.keras.layers.SeparableConv2D(
        int(16 * FILTER_FACTOR),
        kernel_size=(3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(stage5_arm_up)

    stage5_out = flow_head(stage5_arm_up, 8 * FILTER_FACTOR, 2, "stage5_out")

    stage4_arm = attention_refinement_module(stage4, 16 * FILTER_FACTOR)
    stage4_arm_sum = tf.keras.layers.Add()([stage4_arm, stage5_arm_up])
    stage4_arm_up = tf.keras.layers.UpSampling2D(
        size=(2, 2), data_format="channels_last", interpolation="bilinear"
    )(stage4_arm_sum)
    stage4_arm_up = tf.keras.layers.SeparableConv2D(
        int(16 * FILTER_FACTOR),
        kernel_size=(3, 3),
        strides=1,
        padding="SAME",
        activation="relu",
        use_bias=True,
    )(stage4_arm_up)
    stage4_out = flow_head(stage4_arm_up, 8 * FILTER_FACTOR, 2, "stage4_out")

    ffm = feature_fusion_module(stage4_arm_up, stage3, 32 * FILTER_FACTOR)
    conv_out = flow_head(ffm, 32 * FILTER_FACTOR, 2, "output")
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_out, stage4_out, stage5_out, stage3_out]
        if training
        else [conv_out],
    )
    return model


def nanoflownet(batch_size, input_height, input_width, training=True):
    model = NanoFlowNet(input_height, input_width, training)
    return model
