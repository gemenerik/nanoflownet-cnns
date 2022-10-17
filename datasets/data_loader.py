import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline

import datasets


def return_dali_data_loader(
    dataset,
    data,
    epoch_size,
    batch_size,
    workers,
    target_height=None,
    target_width=None,
    mosaicMode=False,
    colorMode=False,
    split_value=-1,
    split_file=None,
    arch=None,
):
    (
        (image0_train_names, image0_val_names),
        (image1_train_names, image1_val_names),
        (flow_train_names, flow_val_names),
        (mb_train_names, mb_val_names),
        (dataset_height, dataset_width),
    ) = datasets.__dict__[dataset](
        data,
        split=split_file if split_file else split_value,
    )

    if (
        len(image0_train_names) == 0
    ):  # instead feed a boolean to return_dali_dataloader function
        full_validation_mode = True
        epoch_size = len(image0_val_names) / batch_size
    else:
        full_validation_mode = False
        if epoch_size == 0:
            epoch_size = len(image0_train_names) / batch_size
        else:
            epoch_size = epoch_size

    def create_image_pipeline(
        batch_size,
        num_threads,
        device_id,
        image0_list,
        image1_list,
        flow_list,
        mb_list,
        valBool,
        mosaicMode,
        colorMode,
        target_height,
        target_width,
        dataset_height,
        dataset_width,
        shuffleBool,
        arch,
    ):
        pipeline = Pipeline(
            batch_size,
            num_threads,
            device_id,
            seed=1,
        )
        with pipeline:
            """READ FILES"""
            image0, _ = fn.readers.file(
                file_root=data,
                files=image0_list,
                shuffle_after_epoch=shuffleBool,
                name="Reader",
                seed=1,
            )
            image1, _ = fn.readers.file(
                file_root=data,
                files=image1_list,
                shuffle_after_epoch=shuffleBool,
                seed=1,
            )
            flo = fn.readers.numpy(
                file_root=data,
                files=flow_list,
                shuffle_after_epoch=shuffleBool,
                seed=1,
            )
            mb, _ = fn.readers.file(
                file_root=data,
                files=mb_list,
                shuffle_after_epoch=shuffleBool,
                seed=1,
            )

            """ DECODE AND RESHAPE """
            image0 = fn.decoders.image(image0, device="cpu")
            image0 = fn.reshape(image0.gpu(), layout="HWC")
            image1 = fn.decoders.image(image1, device="cpu")
            image1 = fn.reshape(image1.gpu(), layout="HWC")
            flo = fn.reshape(flo.gpu(), layout="HWC")
            mb = fn.decoders.image(
                mb, device="cpu", output_type=types.DALIImageType.GRAY
            )
            mb = fn.reshape(mb.gpu(), layout="HWC")
            mb = fn.cast(mb, dtype=types.FLOAT)

            flo = fn.cast(flo, dtype=types.DALIDataType.FLOAT)

            if target_height and target_width:
                resize_factor = min(510 / target_height, 960 / target_width)
                image0 = fn.resize(
                    image0,
                    resize_x=int(dataset_width / resize_factor),
                    resize_y=int(dataset_height / resize_factor),
                )
                image1 = fn.resize(
                    image1,
                    resize_x=int(dataset_width / resize_factor),
                    resize_y=int(dataset_height / resize_factor),
                )
                flo = fn.resize(
                    flo,
                    resize_x=int(dataset_width / resize_factor),
                    resize_y=int(dataset_height / resize_factor),
                )
                mb = fn.resize(
                    mb,
                    resize_x=int(dataset_width / resize_factor),
                    resize_y=int(dataset_height / resize_factor),
                )

                cropped_input_height = (
                    int(dataset_height / resize_factor) // 32 * 32
                    if arch == "flownet2s"
                    else int(dataset_height / resize_factor) // 16 * 16
                )
                cropped_input_width = (
                    int(dataset_width / resize_factor) // 32 * 32
                    if arch == "flownet2s"
                    else int(dataset_width / resize_factor) // 16 * 16
                )

                image0 = fn.crop(
                    image0,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                    seed=42,
                )
                image1 = fn.crop(
                    image1,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                    seed=42,
                )
                mb = fn.crop(
                    mb,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                    seed=42,
                )
                flo = fn.crop(
                    flo,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                    seed=42,
                )
            else:
                raise NotImplementedError

            if valBool:
                pass
            else:
                """RANDOM SYMMETRIC HORIZONTAL MIRROR"""
                coin1 = fn.random.coin_flip(
                    dtype=types.DALIDataType.BOOL, seed=10, probability=0.5
                )
                coin1_n = coin1 ^ True
                image0 = fn.flip(image0) * coin1 + image0 * coin1_n
                image1 = fn.flip(image1) * coin1 + image1 * coin1_n
                mb = fn.flip(mb) * coin1 + mb * coin1_n
                flo_u = fn.slice(flo, axis_names="C", start=0, end=1)
                flo_u = -1 * fn.flip(flo_u) * coin1 + flo_u * coin1_n
                flo_v = fn.slice(flo, axis_names="C", start=1, end=2)
                flo_v = fn.flip(flo_v) * coin1 + flo_v * coin1_n
                flo = fn.cat(flo_u, flo_v, axis_name="C")

                """RANDOM SYMMETRIC VERTICAL MIRROR"""
                coin2 = fn.random.coin_flip(
                    dtype=types.DALIDataType.BOOL, seed=20, probability=0.5
                )
                coin2_n = coin2 ^ True
                image0 = (
                    fn.flip(image0, horizontal=0, vertical=1) * coin2 + image0 * coin2_n
                )
                image1 = (
                    fn.flip(image1, horizontal=0, vertical=1) * coin2 + image1 * coin2_n
                )
                mb = fn.flip(mb, horizontal=0, vertical=1) * coin2 + mb * coin2_n
                flo_u = fn.slice(flo, axis_names="C", start=0, end=1)
                flo_u = (
                    fn.flip(flo_u, horizontal=0, vertical=1) * coin2 + flo_u * coin2_n
                )
                flo_v = fn.slice(flo, axis_names="C", start=1, end=2)
                flo_v = (
                    -1 * fn.flip(flo_v, horizontal=0, vertical=1) * coin2
                    + flo_v * coin2_n
                )
                flo = fn.cat(flo_u, flo_v, axis_name="C")

                """ RANDOM ASYMMETRIC SCALING """
                #
                #   Transformation matrices based on toy example by Michał Zientkiewicz 
                #

                scale = fn.random.uniform(range=(0.9, 1.1), shape=[2])

                w = cropped_input_width  # known size of the input video
                h = cropped_input_height
                grid = 1  # spacing of flow vectors

                fw = cropped_input_width // grid  # size of the motion vector field
                fh = cropped_input_height // grid

                x = np.arange(fw).reshape([1, fw]) + 0.5  # (0.5, 0.5) is a pixel center
                y = np.arange(fh).reshape([fh, 1]) + 0.5
                xy = np.stack([x, np.zeros_like(x)], axis=2) + np.stack(
                    [np.zeros_like(y), y], axis=2
                )
                xy = (
                    xy * grid
                )  # now go from optical flow grid to source image coordinates

                m = fn.transforms.scale(scale=scale, center=[w / 2, h / 2])
                image1 = fn.warp_affine(
                    image1, matrix=m, inverse_map=False, fill_value=0
                )

                # optical flow augmentation goes here
                of3 = fn.coord_transform(flo + xy, MT=m) - xy
                flo = of3

                # take a crop to avoid padded parts
                image0 = fn.resize(
                    image0,
                    resize_x=cropped_input_width * 1 / 0.9,
                    resize_y=cropped_input_height * 1 / 0.9,
                )
                image1 = fn.resize(
                    image1,
                    resize_x=cropped_input_width * 1 / 0.9,
                    resize_y=cropped_input_height * 1 / 0.9,
                )
                mb = fn.resize(
                    mb,
                    resize_x=cropped_input_width * 1 / 0.9,
                    resize_y=cropped_input_height * 1 / 0.9,
                )
                flo = fn.resize(
                    flo,
                    resize_x=cropped_input_width * 1 / 0.9,
                    resize_y=cropped_input_height * 1 / 0.9,
                )

                # correct flow for zoom of 1/0.9 on both images (thus flow becomes 1/0.9 times larger)
                flo = flo * 1 / 0.9

                image0 = fn.crop(
                    image0,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                image1 = fn.crop(
                    image1,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                mb = fn.crop(
                    mb,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                flo = fn.crop(
                    flo,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )

                """ RANDOM SYMMETRIC ROTATION """
                angle_deg = fn.random.uniform(range=(-15, 15))
                angle = angle_deg * 3.14159265359 / 180
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)

                image0 = fn.resize(
                    image0,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )
                image1 = fn.resize(
                    image1,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )
                mb = fn.resize(
                    mb,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )
                flo = fn.resize(
                    flo,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )

                # correct flow
                flo_u = fn.slice(flo, axis_names="C", start=0, end=1)
                flo_v = fn.slice(flo, axis_names="C", start=1, end=2)
                x_scale_factor = (
                    cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle)
                ) / cropped_input_width
                y_scale_factor = (
                    cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle)
                ) / cropped_input_height
                flo_u = flo_u * x_scale_factor
                flo_v = flo_v * y_scale_factor
                flo = fn.cat(flo_u, flo_v, axis_name="C")

                image0 = fn.rotate(
                    image0,
                    angle=angle_deg,
                    fill_value=0,
                    axis=0,
                    keep_size=False,
                )
                image1 = fn.rotate(
                    image1,
                    angle=angle_deg,
                    fill_value=0,
                    axis=0,
                    keep_size=False,
                )
                mb = fn.rotate(
                    mb,
                    angle=angle_deg,
                    fill_value=0,
                    axis=0,
                    keep_size=False,
                )
                flo = fn.rotate(
                    flo,
                    angle=angle_deg,
                    fill_value=0,
                    axis=0,
                    keep_size=True,
                )
                flo_u1 = fn.slice(flo, axis_names="C", start=0, end=1)
                flo_v1 = fn.slice(flo, axis_names="C", start=1, end=2)

                flo_u2 = cos_angle * flo_u1 - sin_angle * flo_v1
                flo_v2 = sin_angle * flo_u1 + cos_angle * flo_v1
                flo = fn.cat(flo_u2, flo_v2, axis_name="C")

                image0 = fn.crop(
                    image0,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                image1 = fn.crop(
                    image1,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                mb = fn.crop(
                    mb,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                flo = fn.crop(
                    flo,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )

                """ RANDOM ASYMMETRIC ROTATION """
                #
                #   Transformation matrices based on toy example by Michał Zientkiewicz 
                #

                angle_deg = fn.random.uniform(range=(-3, 3))
                angle = angle_deg * 3.14159265359 / 180

                w = cropped_input_width  # known size of the input video
                h = cropped_input_height
                grid = 1  # spacing of flow vectors

                fw = cropped_input_width // grid  # size of the motion vector field
                fh = cropped_input_height // grid

                x = np.arange(fw).reshape([1, fw]) + 0.5  # (0.5, 0.5) is a pixel center
                y = np.arange(fh).reshape([fh, 1]) + 0.5
                xy = np.stack([x, np.zeros_like(x)], axis=2) + np.stack(
                    [np.zeros_like(y), y], axis=2
                )
                xy = (
                    xy * grid
                )  # now go from optical flow grid to source image coordinates

                m = fn.transforms.rotation(angle=angle_deg, center=[w / 2, h / 2])
                image1 = fn.warp_affine(image1, matrix=m, inverse_map=False)

                # optical flow augmentation goes here
                of3 = fn.coord_transform(flo + xy, MT=m) - xy
                flo = of3

                # take a crop to avoid padded parts
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)

                image0 = fn.resize(
                    image0,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )
                image1 = fn.resize(
                    image1,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )
                mb = fn.resize(
                    mb,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )
                flo = fn.resize(
                    flo,
                    resize_x=cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle),
                    resize_y=cropped_input_height * math.abs(cos_angle)
                    + cropped_input_width * math.abs(sin_angle),
                )

                # correct flow
                flo_u = fn.slice(flo, axis_names="C", start=0, end=1)
                flo_v = fn.slice(flo, axis_names="C", start=1, end=2)
                x_scale_factor = (
                    cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle)
                ) / cropped_input_width
                y_scale_factor = (
                    cropped_input_width * math.abs(cos_angle)
                    + cropped_input_height * math.abs(sin_angle)
                ) / cropped_input_height
                flo_u = flo_u * x_scale_factor
                flo_v = flo_v * y_scale_factor
                flo = fn.cat(flo_u, flo_v, axis_name="C")

                image0 = fn.crop(
                    image0,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                image1 = fn.crop(
                    image1,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                mb = fn.crop(
                    mb,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                flo = fn.crop(
                    flo,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )

                """ RANDOM ASYMMETRIC TRANSLATION """
                #
                #   Transformation matrices based on toy example by Michał Zientkiewicz 
                #
                
                dxdy = fn.random.uniform(range=(-2, 2), shape=2)

                w = cropped_input_width  # known size of the input video
                h = cropped_input_height
                grid = 1  # spacing of flow vectors

                fw = cropped_input_width // grid  # size of the motion vector field
                fh = cropped_input_height // grid

                x = np.arange(fw).reshape([1, fw]) + 0.5  # (0.5, 0.5) is a pixel center
                y = np.arange(fh).reshape([fh, 1]) + 0.5
                xy = np.stack([x, np.zeros_like(x)], axis=2) + np.stack(
                    [np.zeros_like(y), y], axis=2
                )
                xy = (
                    xy * grid
                )  # now go from optical flow grid to source image coordinates

                m = fn.transforms.translation(offset=dxdy)
                image1 = fn.warp_affine(
                    image1, matrix=m, inverse_map=False, fill_value=0
                )

                # optical flow augmentation goes here
                of3 = fn.coord_transform(flo + xy, MT=m) - xy
                flo = of3

                # take a crop to avoid padded parts
                image0 = fn.resize(
                    image0,
                    resize_x=cropped_input_width + 4 * 2,
                    resize_y=cropped_input_height + 4 * 2,
                )
                image1 = fn.resize(
                    image1,
                    resize_x=cropped_input_width + 4 * 2,
                    resize_y=cropped_input_height + 4 * 2,
                )
                mb = fn.resize(
                    mb,
                    resize_x=cropped_input_width + 4 * 2,
                    resize_y=cropped_input_height + 4 * 2,
                )
                flo = fn.resize(
                    flo,
                    resize_x=cropped_input_width + 4 * 2,
                    resize_y=cropped_input_height + 4 * 2,
                )

                # correct flow
                flo_u = fn.slice(flo, axis_names="C", start=0, end=1)
                flo_v = fn.slice(flo, axis_names="C", start=1, end=2)
                x_scale_factor = 1 + (4 * 2) / cropped_input_width
                y_scale_factor = 1 + (4 * 2) / cropped_input_height
                flo_u = flo_u * x_scale_factor
                flo_v = flo_v * y_scale_factor
                flo = fn.cat(flo_u, flo_v, axis_name="C")

                image0 = fn.crop(
                    image0,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                image1 = fn.crop(
                    image1,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                mb = fn.crop(
                    mb,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )
                flo = fn.crop(
                    flo,
                    crop=(cropped_input_height, cropped_input_width),
                    crop_pos_x=0.5,
                    crop_pos_y=0.5,
                )

                """ Adjust hue, saturation and brightness """
                symmetric_coin = fn.random.coin_flip(
                    dtype=types.DALIDataType.BOOL, seed=60, probability=0.8
                )
                asymmetric_coin = symmetric_coin ^ True

                brightness = (
                    fn.random.uniform(range=[0.8, 1.2], seed=25) * symmetric_coin
                    + fn.random.uniform(range=[0.95, 1.05], seed=26) * asymmetric_coin
                )
                contrast = (
                    fn.random.uniform(range=[0.6, 1.4], seed=27) * symmetric_coin
                    + fn.random.uniform(range=[0.95, 1.05], seed=28) * asymmetric_coin
                )
                hue = fn.random.uniform(range=[-2.86, 2.86], seed=29)
                saturation = fn.random.uniform(range=[0.9, 1.1], seed=30)

                image0 = (
                    fn.color_twist(
                        image0,
                        brightness=brightness,
                        contrast=contrast,
                        hue=hue,
                        saturation=saturation,
                    )
                    * symmetric_coin
                    + image0 * asymmetric_coin
                )
                image1 = fn.color_twist(
                    image1,
                    brightness=brightness,
                    contrast=contrast,
                    hue=hue,
                    saturation=saturation,
                )

            """ NORMALIZATION """
            flo = fn.crop_mirror_normalize(
                flo,
                mean=[0.00000001, 0.00000001],
                std=[1, 1],
                output_layout="HWC",
            )

            if not colorMode:
                image0 = fn.color_space_conversion(
                    image0, image_type=types.RGB, output_type=types.GRAY
                )
                image1 = fn.color_space_conversion(
                    image1, image_type=types.RGB, output_type=types.GRAY
                )

            images = fn.cat(image0, image1, axis_name="C")
            images = fn.cast(images, dtype=types.DALIDataType.FLOAT)
            images = (
                fn.crop_mirror_normalize(
                    images,
                    mean=[128, 128],
                    std=[128, 128],
                    output_layout="HWC",
                )
                if not colorMode
                else fn.crop_mirror_normalize(
                    images,
                    mean=[128, 128, 128, 128, 128, 128],
                    std=[128, 128, 128, 128, 128, 128],
                    output_layout="HWC",
                )
            )
            mb = fn.cast(mb, dtype=types.DALIDataType.BOOL)
            mb = fn.cast(mb, dtype=types.DALIDataType.FLOAT)

            targets = fn.cat(flo, mb, axis_name="C")

            pipeline.set_outputs(images, targets)
        return pipeline

    class DALILoader:
        def __init__(
            self,
            batch_size,
            image0_names,
            image1_names,
            flow_names,
            mb_names,
            disableAugmentations,
            num_threads,
            device_id,
            mosaicMode,
            colorMode,
            target_height,
            target_width,
            shuffleBool,
            arch,
        ):
            self.pipeline = create_image_pipeline(
                batch_size,
                num_threads,
                device_id,
                image0_names,
                image1_names,
                flow_names,
                mb_names,
                disableAugmentations,
                mosaicMode,
                colorMode,
                target_height,
                target_width,
                dataset_height,
                dataset_width,
                shuffleBool,
                arch,
            )
            self.pipeline.build()
            self.epoch_size = self.pipeline.epoch_size("Reader") / batch_size

            resize_factor = min(510 / target_height, 960 / target_width)

            channels = 2 if not colorMode else 6

            self.dali_iterator = dali_tf.DALIDataset(
                pipeline=self.pipeline,
                batch_size=batch_size,
                output_shapes=(
                    (
                        batch_size,
                        int(dataset_height / resize_factor) // 32 * 32,
                        int(dataset_width / resize_factor) // 32 * 32,
                        channels,
                    ),
                    (
                        batch_size,
                        int(dataset_height / resize_factor) // 32 * 32,
                        int(dataset_width / resize_factor) // 32 * 32,
                        3,
                    ),
                )
                if arch == "flownet2s"
                else (
                    (
                        batch_size,
                        int(dataset_height / resize_factor) // 16 * 16,
                        int(dataset_width / resize_factor) // 16 * 16,
                        channels,
                    ),
                    (
                        batch_size,
                        int(dataset_height / resize_factor) // 16 * 16,
                        int(dataset_width / resize_factor) // 16 * 16,
                        3,
                    ),
                ),
                output_dtypes=(tf.float32, tf.float32),
                device_id=0,
            )

        def __len__(self):
            return int(self.epoch_size)

        def __iter__(self):
            return self.dali_iterator.__iter__()

        def reset(self):
            return self.dali_iterator.reset()

    if full_validation_mode:
        train_loader = None
    else:
        train_loader = DALILoader(
            batch_size=batch_size,
            num_threads=workers,
            device_id=0,
            image0_names=image0_train_names,
            image1_names=image1_train_names,
            flow_names=flow_train_names,
            mb_names=mb_train_names,
            disableAugmentations=True if dataset == "flying_things_3d" else False,
            mosaicMode=mosaicMode,
            colorMode=colorMode,
            target_height=target_height,
            target_width=target_width,
            shuffleBool=True,
            arch=arch,
        )

    val_loader = DALILoader(
        batch_size=batch_size,
        num_threads=workers,
        device_id=0,
        image0_names=image0_val_names,
        image1_names=image1_val_names,
        flow_names=flow_val_names,
        mb_names=mb_val_names,
        disableAugmentations=True,
        mosaicMode=mosaicMode,
        colorMode=colorMode,
        target_height=target_height,
        target_width=target_width,
        shuffleBool=False,
        arch=arch,
    )
    return train_loader, val_loader, epoch_size
