# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/main/docs/image_apis.md#computer-vision-mode

# import setup_path
import airsim

import pprint
import os
import time
import math
import tempfile
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

client.simSetCameraFov("0", 87)  # AI-deck FOV
# client.simSetFocusDistance("0", 35)

airsim.wait_key("Press any key to get camera parameters")
for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
    print("CameraInfo %d:" % camera_name)
    pp.pprint(camera_info)

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")
print("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

airsim.wait_key("Press any key to get images")
for x in range(3):  # do few times
    z = x * -20 - 5  # some random number
    client.simSetVehiclePose(
        airsim.Pose(
            airsim.Vector3r(z, z, z), airsim.to_quaternion(x / 3.0, 0, x / 3.0)
        ),
        True,
    )

    # capture "before image"
    responses_0 = client.simGetImages(
        [
            airsim.ImageRequest("0", airsim.ImageType.Scene),  # input frame 1
            airsim.ImageRequest("0", airsim.ImageType.OpticalFlowVis),  # optical flow between second image of last pair, and first of current pair, not useful
        ]
    )

    # move the quadcopter with random permutations
    z_x = np.random.randint(-1, 1)/5
    z_y = np.random.randint(-1, 1)/5
    z_z = np.random.randint(-1, 1)/5
    z_theta = np.random.randint(-1, 1)/50
    z_phi = np.random.randint(-1, 1)/50
    z_psi = np.random.randint(-1, 1)/50

    client.simSetVehiclePose(
        airsim.Pose(
            airsim.Vector3r(z+z_x, z+z_y, z+z_z), airsim.to_quaternion(x / 3.0 + z_theta, 0 + z_phi, x / 3.0 + z_psi)
        ),
        True,
    )

    responses_1 = client.simGetImages(
        [
            airsim.ImageRequest("0", airsim.ImageType.Scene),  # input frame 2
            airsim.ImageRequest("0", airsim.ImageType.OpticalFlowVis),  # optical flow ground truth
        ]
    )

    for i, response in enumerate(responses_0 + responses_1):
        filename = os.path.join(tmp_dir, str(x) + "_" + str(i))
        if response.pixels_as_float:
            print(
                "Type %d, size %d, pos %s"
                % (
                    response.image_type,
                    len(response.image_data_float),
                    pprint.pformat(response.camera_position),
                )
            )
            airsim.write_pfm(
                os.path.normpath(filename + ".pfm"), airsim.get_pfm_array(response)
            )
        else:
            print(
                "Type %d, size %d, pos %s"
                % (
                    response.image_type,
                    len(response.image_data_uint8),
                    pprint.pformat(response.camera_position),
                )
            )
            airsim.write_file(
                os.path.normpath(filename + ".png"), response.image_data_uint8
            )

    pose = client.simGetVehiclePose()
    pp.pprint(pose)

    time.sleep(3)

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(
    airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True
)
