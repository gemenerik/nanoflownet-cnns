import glob
import itertools
import os.path
from pathlib import Path

#
# taken from Source code for torchvision.datasets._optical_flow
#


def append_files(root_path, split="train", pass_name="clean", camera="left"):
    # verify_str_arg(split, "split", valid_values=("train", "test"))
    split = split.upper()

    # verify_str_arg(pass_name, "pass_name", valid_values=("clean", "final", "both"))
    passes = {
        "clean": ["frames_cleanpass"],
        "final": ["frames_finalpass"],
        "both": ["frames_cleanpass", "frames_finalpass"],
    }[pass_name]

    # verify_str_arg(camera, "camera", valid_values=("left", "right", "both"))
    cameras = ["left", "right"] if camera == "both" else [camera]

    root_path = Path(root_path)  # / "FlyingThings3D"

    directions = ("into_future", "into_past")
    image0_list = []
    image1_list = []
    flow_list = []
    motion_boundary_list = []
    for pass_name, camera, direction in itertools.product(passes, cameras, directions):
        image_dirs = sorted(glob.glob(str(root_path / pass_name / split / "*/*")))
        image_dirs = sorted(Path(image_dir) / camera for image_dir in image_dirs)

        flow_dirs = sorted(glob.glob(str(root_path / "optical_flow" / split / "*/*")))
        flow_dirs = sorted(
            Path(flow_dir) / direction / camera for flow_dir in flow_dirs
        )

        motion_boundary_dirs = sorted(
            glob.glob(str(root_path / "motion_boundaries" / split / "*/*"))
        )
        motion_boundary_dirs = sorted(
            Path(motion_boundary_dir) / direction / camera
            for motion_boundary_dir in motion_boundary_dirs
        )

        if not image_dirs or not flow_dirs:
            raise FileNotFoundError(
                "Could not find the FlyingThings3D flow images. "
                "Please make sure the directory structure is correct."
            )

        for image_dir, flow_dir, motion_boundary_dir in zip(
            image_dirs, flow_dirs, motion_boundary_dirs
        ):
            images = sorted(glob.glob(str(image_dir / "*.png")))
            flows = sorted(glob.glob(str(flow_dir / "*.npy")))
            motion_boundaries = sorted(glob.glob(str(motion_boundary_dir / "*.pgm")))
            for i in range(len(flows) - 1):
                if direction == "into_future":
                    image0_list += [os.path.relpath(images[i], root_path)]
                    image1_list += [os.path.relpath(images[i + 1], root_path)]
                    flow_list += [os.path.relpath(flows[i], root_path)]
                    motion_boundary_list += [
                        os.path.relpath(motion_boundaries[i], root_path)
                    ]
                elif direction == "into_past":
                    image0_list += [os.path.relpath(images[i + 1], root_path)]
                    image1_list += [os.path.relpath(images[i], root_path)]
                    flow_list += [os.path.relpath(flows[i + 1], root_path)]
                    motion_boundary_list += [
                        os.path.relpath(motion_boundaries[i + 1], root_path)
                    ]
    return image0_list, image1_list, flow_list, motion_boundary_list


def flying_things_3d(dir, split=None):
    image0_train, image1_train, flow_train, mb_train = append_files(dir, "train")
    image0_val, image1_val, flow_val, mb_val = append_files(dir, "test")

    return (
        (image0_train, image0_val),
        (image1_train, image1_val),
        (flow_train, flow_val),
        (mb_train, mb_val),
        (540, 960),
    )
