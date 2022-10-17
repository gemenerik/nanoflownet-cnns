import glob
import os.path

from .traintestsplit import split2list


def make_dataset(dataset_dir, split=None, dataset_type="clean"):
    print(dataset_dir)
    flow_dir = "flow"
    assert os.path.isdir(os.path.join(dataset_dir, flow_dir))
    img_dir = dataset_type
    assert os.path.isdir(os.path.join(dataset_dir, img_dir))
    mb_dir = "motion_boundaries"
    assert os.path.isdir(os.path.join(dataset_dir, mb_dir))

    image0 = []
    image1 = []
    flow = []
    motion_boundaries = []
    for flow_map in sorted(
        glob.glob(os.path.join(dataset_dir, flow_dir, "*", "*.npy"))
    ):
        flow_map = os.path.relpath(
            flow_map, os.path.join(dataset_dir, flow_dir)
        )

        scene_dir, filename = os.path.split(flow_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split("_")
        frame_nb = int(frame_nb)
        img1 = os.path.join(
            img_dir, scene_dir, "{}_{:04d}.png".format(prefix, frame_nb)
        )
        img2 = os.path.join(
            img_dir, scene_dir, "{}_{:04d}.png".format(prefix, frame_nb + 1)
        )
        mb = os.path.join(
            mb_dir, scene_dir, "{}_{:04d}.png".format(prefix, frame_nb)
        )
        flow_map = os.path.join(flow_dir, flow_map)
        if not (
            os.path.isfile(os.path.join(dataset_dir, img1))
            and os.path.isfile(os.path.join(dataset_dir, img2)) and os.path.isfile(os.path.join(dataset_dir, mb))
        ):
            continue
        image0.append(img1)
        image1.append(img2)
        flow.append(flow_map)
        motion_boundaries.append(mb)

    return (
        split2list(image0, split, default_split=0.87),
        split2list(image1, split, default_split=0.87),
        split2list(flow, split, default_split=0.87),
        split2list(motion_boundaries, split, default_split=0.87),
        (436, 1024),
    )


def mpi_sintel_clean(root, split=None):
    train_list = make_dataset(root, split, "clean")
    return train_list


def mpi_sintel_final(root, split=None):
    train_list = make_dataset(root, split, "final")

    return train_list
