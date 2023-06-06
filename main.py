import datetime
import os

import numpy as np
import tensorflow as tf
from tqdm.keras import TqdmCallback

import models
from datasets.data_loader import return_dali_data_loader
from src import flow_vis
from src.multiscaleloss import (
    edge_detail_aggregate_loss,
    mb_detail_aggregate_loss,
    multiscaleEPE,
    no_loss,
)
from src.parser import return_parsed_args


def return_callbacks(
    save_path,
    args,
    val_loader,
    model,
    sintel_clean_train_loader,
    sintel_clean_train_length,
    sintel_final_train_loader,
    sintel_final_train_length,
):
    # Define callbacks
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        save_path + "/best_model.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
        period=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path + "/checkpoint.h5",
        monitor="loss",
        verbose=1,
        save_best_only=False,
        mode="auto",
        period=1,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=save_path, histogram_freq=1
    )

    def log_images(epoch, logs):
        if epoch % 10 == 0:
            data = next(iter(val_loader))
            model_prediction = model.predict(data[0])
            if args.execution_mode:
                output = model_prediction
            else:
                output = model_prediction[0]
            gt_data = data[1]

            flow_pred = []
            flow_gt = []
            for k in range(len(output)):
                output_flow = output[k, :, :, 0:2]
                flo_bgr = flow_vis.flow_to_color(output_flow, convert_to_bgr=True)
                flow_pred.append(flo_bgr)

                gt_flow = gt_data[k, :, :, 0:2]
                gt_flo_bgr = flow_vis.flow_to_color(gt_flow, convert_to_bgr=True)
                flow_gt.append(gt_flo_bgr)

    image_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)

    class ExtraValidation(tf.keras.callbacks.Callback):
        """Log evaluation metrics of an extra validation set. This callback
        is useful for model training scenarios where multiple validation sets
        are used for evaluation (as Keras by default, provides functionality for
        evaluating on a single validation set only).
        The evaluation metrics are also logged to TensorBoard.
        Args:
            validation_data: A tf.data.Dataset pipeline used to evaluate the
                model, essentially an extra validation dataset.
            tensorboard_path: Path to the TensorBoard logging directory.
            validation_freq: Number of epochs to wait before performing
                subsequent evaluations.
        """

        def __init__(
            self,
            validation_data,
            tensorboard_path,
            validation_freq=1,
            dataset_name="MPI Sintel clean",
            dataset_eval_length=None,
        ):
            super(ExtraValidation, self).__init__()

            self.validation_data = validation_data
            self.dataset_name = dataset_name
            self.dataset_eval_length = dataset_eval_length
            self.tensorboard_path = tensorboard_path

            self.tensorboard_writer = tf.summary.create_file_writer(
                f"{self.tensorboard_path}/mpi_sintel_val"
            )

            self.validation_freq = validation_freq

        def on_epoch_end(self, epoch, logs=None):
            # evaluate at an interval of `validation_freq` epochs
            if (epoch + 1) % self.validation_freq == 0:
                # gather metric names form model
                metric_names = [
                    "{}_{}".format("epoch", metric.name)
                    for metric in self.model.metrics
                ]
                # TODO: fix `model.evaluate` memory leak on TPU
                # gather the evaluation metrics
                scores = self.model.evaluate(
                    self.validation_data, verbose=1, steps=self.dataset_eval_length
                )

                if args.execution_mode:
                    results = [(metric_names[0], scores)]
                else:
                    results = zip(metric_names, scores)

                # gather evaluation metrics to TensorBoard
                with self.tensorboard_writer.as_default():
                    for metric_name, score in results:
                        tf.summary.scalar(metric_name, score, step=epoch)

    callbacks = [
        checkpoint_best,
        checkpoint,
        image_callback,
        TqdmCallback(verbose=1, dynamic_ncols=True),
        ExtraValidation(
            sintel_clean_train_loader.dali_iterator,
            save_path,
            validation_freq=10,
            dataset_name="MPI Sintel train clean",
            dataset_eval_length=sintel_clean_train_length,
        ),
        ExtraValidation(
            sintel_final_train_loader.dali_iterator,
            save_path,
            validation_freq=10,
            dataset_name="MPI Sintel train final",
            dataset_eval_length=sintel_final_train_length,
        ),
        tensorboard_callback,
    ]
    return callbacks


def main():
    # Parse command line arguments
    args = return_parsed_args()

    data = (
        "/workspace/FlyingChairs2/"
        if args.dataset == "flying_chairs2"
        else "/workspace/flowData/FlyingThings3D/"
        if args.dataset == "flying_things_3d"
        else "/workspace/flowData/MPI-Sintel/training/"
        if args.dataset == "mpi_sintel_clean"
        else None
    )
    data_sintel_train = "/workspace/flowData/MPI-Sintel/training/"
    data_sintel_test = "/workspace/flowData/MPI-Sintel/test/"

    # Fix random seed
    np.random.seed(132)

    # Create output directory
    save_path = "{},{}epochs{},b{},lr{}".format(
        args.arch,
        args.epochs,
        ",epochSize" + str(args.epoch_size) if args.epoch_size > 0 else "",
        args.batch_size,
        args.lr,
    )
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join("results/", args.dataset, save_path)
    print("=> will save everything to {}".format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up data loader
    train_loader, val_loader, args.epoch_size = return_dali_data_loader(
        args.dataset,
        data,
        args.epoch_size,
        args.batch_size,
        args.workers,
        args.input_height,
        args.input_width,
        args.mosaic,
        args.color,
        args.split_value,
        args.split_file,
        args.arch,
    )
    _, sintel_clean_train_loader, sintel_clean_train_length = return_dali_data_loader(
        "mpi_sintel_clean",
        data_sintel_train,
        0,
        args.batch_size,
        args.workers,
        args.input_height,
        args.input_width,
        args.mosaic,
        args.color,
        -1,
        None,
        args.arch,
    )  # split set to -1 to force all data to be used for validation
    _, sintel_final_train_loader, sintel_final_train_length = return_dali_data_loader(
        "mpi_sintel_final",
        data_sintel_train,
        0,
        args.batch_size,
        args.workers,
        args.input_height,
        args.input_width,
        args.mosaic,
        args.color,
        -1,
        None,
        args.arch,
    )

    # Set up model
    model = models.__dict__[args.arch](
        args.batch_size, args.input_height, args.input_width
    )
    if args.pretrained:
        model.load_weights(args.pretrained)
    if args.execution_mode:
        model = tf.keras.Model(inputs=model.input, outputs=model.output[0])


    with tf.device("/gpu:0"):
        # Compile model
        model.summary()
        if args.arch == "flownet2s":
            loss_functions = [
                multiscaleEPE,
                multiscaleEPE,
                multiscaleEPE,
                multiscaleEPE,
            ]
        else:
            if args.detail_guidance == "off":
                loss_functions = [multiscaleEPE, multiscaleEPE, multiscaleEPE, no_loss]
            elif args.detail_guidance == "motion_boundaries":
                loss_functions = [
                    multiscaleEPE,
                    multiscaleEPE,
                    multiscaleEPE,
                    mb_detail_aggregate_loss,
                ]
            elif args.detail_guidance == "edge_detect":
                loss_functions = [
                    multiscaleEPE,
                    multiscaleEPE,
                    multiscaleEPE,
                    edge_detail_aggregate_loss,
                ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss=loss_functions if not args.execution_mode else [multiscaleEPE],
            loss_weights=args.multiscale_weights
            if (args.arch == "nanoflownet" and not args.execution_mode)
            else [0.3200, 0.1600, 0.0800, 0.0400]
            if not args.execution_mode
            else 1,
        )

        # Train model
        model.fit(
            train_loader.dali_iterator,
            epochs=args.epochs,
            steps_per_epoch=args.epoch_size,
            shuffle=True,
            validation_data=val_loader.dali_iterator,
            validation_steps=int(len(val_loader) / args.batch_size),
            verbose=0,
            callbacks=return_callbacks(
                save_path,
                args,
                val_loader,
                model,
                sintel_clean_train_loader,
                sintel_clean_train_length,
                sintel_final_train_loader,
                sintel_final_train_length,
            ),
        )

        # Save model
        model.save(save_path + "/modelsave", save_format="tf")
        model.save(save_path + "/modelsave/nanoflownet.h5", save_format="h5")

        """SAVE TFLITE MODELS"""
        # Convert to TensorFlow lite
        input_shape = (
            (args.input_height, args.input_width, 3)
            if args.color
            else (args.input_height, args.input_width, 1)
        )
        input_1, input_2 = tf.keras.Input(shape=input_shape), tf.keras.Input(
            shape=input_shape
        )
        cat_layer = tf.keras.layers.Concatenate()([input_1, input_2])
        cat_model = tf.keras.Model(inputs=[input_1, input_2], outputs=cat_layer)
        out = model(cat_model.output)[0]
        multi_input_model = tf.keras.Model(
            inputs=[cat_model.input[0], cat_model.input[1]], outputs=out
        )
        multi_input_model.summary()
        converter = tf.lite.TFLiteConverter.from_keras_model(multi_input_model)
        tflite_model = converter.convert()
        with open(f"{save_path}/modelsave/nanoflownet_unquantized.tflite", "wb") as f:
            f.write(tflite_model)


if __name__ == "__main__":
    main()
