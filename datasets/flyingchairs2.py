import os.path
import glob

def append_files(dir, val=False):
    image0, image1, flow, mb = [], [], [], []
    dir = dir + "train/" if not val else dir + "val/"
    flo_list = sorted(glob.glob(os.path.join(dir, "*-flow_01.npy")))
    for flow_map in flo_list:
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-12]
        img1 = root_filename + "-img_0.png"
        img2 = root_filename + "-img_1.png"
        motion_boundary = root_filename + "-mb_01.png"

        subdir = "train/" if not val else "val/"
        image0.append(subdir + img1)
        image1.append(subdir + img2)
        flow.append(subdir + flow_map)
        mb.append(subdir + motion_boundary)
    return image0, image1, flow, mb


def flying_chairs2(dir, split=None):
    image0_train, image1_train, flow_train, mb_train = append_files(
        dir, False
    )
    image0_val, image1_val, flow_val, mb_val = append_files(dir, True)

    return (
        (image0_train, image0_val),
        (image1_train, image1_val),
        (flow_train, flow_val),
        (mb_train, mb_val),
        (384, 512),
    )
