import argparse
import models
import datasets


def return_parsed_args():
    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )
    dataset_names = sorted(name for name in datasets.__all__)

    parser = argparse.ArgumentParser(
        description="NanoFlowNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        default="flying_chairs2",
        choices=dataset_names,
        help="dataset type : " + " | ".join(dataset_names),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-s",
        "--split-file",
        default=None,
        type=str,
        help="test-val split file",
    )

    group.add_argument(
        "--split-value",
        default=0.8,
        type=float,
        help="test-val split proportion between 0 (only test) and 1 (only train), "
        "will be overwritten if a split file is set",
    )

    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="nanoflownet",
        choices=model_names,
        help="model architecture, overwritten if pretrained is specified: "
        + " | ".join(model_names),
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=12,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )

    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--epoch-size",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch size (will match dataset size if set to 0)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=8,
        type=int,
        metavar="N",
        help="mini-batch size",
    )
    parser.add_argument(
        "--input-height",
        default=112,
        type=int,
        metavar="N",
        help="mini-batch size",
    )

    parser.add_argument(
        "--input-width",
        default=160,
        type=int,
        metavar="N",
        help="mini-batch size",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )

    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum for sgd, alpha parameter for adam",
    )

    parser.add_argument(
        "--beta",
        default=0.999,
        type=float,
        metavar="M",
        help="beta parameter for adam",
    )

    parser.add_argument(
        "--epsilon",
        default=1e-08,
        type=float,
        metavar="M",
        help="epsilon for adam",
    )

    parser.add_argument(
        "--multiscale-weights",
        "-w",
        default=[1, 0.5, 0.25, 1],
        type=float,
        nargs=4,
        help="training weights for each loss, [output_loss, auxiliary loss (1/8), auxiliary loss (1/16), detail guidance loss]",
        metavar=("W1", "W2", "W3", "W4"),
    )

    parser.add_argument(
        "--mosaic",
        action="store_true",
        help="enable input mosaicing",
    )

    parser.add_argument(
        "--color",
        action="store_true",
        help="enable color mode instead of grayscale",
    )

    parser.add_argument(
        "--detail-guidance",
        default="motion_boundaries",
        choices=["off", "motion_boundaries", "edge_detect"],
        help="choose detail guidance mode",
    )

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        help="path to pre-trained model",
    )

    parser.add_argument(
        "--no-date",
        action="store_true",
        help="don't append date timestamp to folder",
    )

    parser.add_argument(
        "--milestones",
        default=[
            145,
            215,
            290,
            360,
        ],
        metavar="N",
        nargs="*",
        help="epochs at which learning rate is divided by 2",
    )

    parser.add_argument(
        "--execution-mode",
        action="store_true",
        help="disable additional non-flow outputs of net",
    )

    global args, best_EPE
    best_EPE = -1
    n_iter = 0

    args = parser.parse_args()
    return args
