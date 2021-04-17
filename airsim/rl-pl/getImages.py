import airsim
from airsim.types import ImageRequest  # pip install airsim
import numpy as np
import os
# for car use CarClient()
client = airsim.CarClient(ip="127.0.0.1")

responses = client.simGetImages ( [
        # RGB array bytes
        airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
        # floating point depth planner
        airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, True),
        # floating point depth perspective
        airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True),
        # floating point depth vision
        airsim.ImageRequest(0, airsim.ImageType.DepthVis, True),
        # floating point disparity normalized
        airsim.ImageRequest(0, airsim.ImageType.DisparityNormalized, True),
        # floating point segmentation vision
        airsim.ImageRequest(0, airsim.ImageType.Segmentation, True)
    ]
)

r0 = responses[0]
r1 = responses[1]
r2 = responses[2]
r3 = responses[3]
r4 = responses[4]
r5 = responses[5]

img0 = (np.frombuffer(r0.image_data_uint8, dtype=np.uint8)).reshape(r0.height, r0.width, 3)
img1 = np.flipud(airsim.list_to_2d_float_array(r1.image_data_float, r1.width, r1.height))
img2 = np.flipud(airsim.list_to_2d_float_array(r2.image_data_float, r2.width, r2.height))
img3 = np.flipud(airsim.list_to_2d_float_array(r3.image_data_float, r3.width, r3.height))
img4 = np.flipud(airsim.list_to_2d_float_array(r4.image_data_float, r4.width, r4.height))
img5 = np.flipud(airsim.list_to_2d_float_array(r5.image_data_float, r5.width, r5.height))

airsim.write_png(os.path.normpath("rgb" + '.png'), img0)
airsim.write_pfm(os.path.normpath("depthPlanner" + '.fpm'), img1)
airsim.write_pfm(os.path.normpath("depthPerspective" + '.fpm'), img2)
airsim.write_pfm(os.path.normpath("depthVisionCenter" + '.fpm'), img3)
airsim.write_pfm(os.path.normpath("disparityNormalized" + '.fpm'), img4)
airsim.write_pfm(os.path.normpath("segmentation" + '.fpm'), img5)
