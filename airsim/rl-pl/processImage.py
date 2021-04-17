import airsim
from airsim.types import ImageRequest  # pip install airsim
import numpy as np
import os
import time
# for car use CarClient()
client = airsim.CarClient(ip="127.0.0.1")

# try:
#     while True:
responses = client.simGetImages ( [
        airsim.ImageRequest(0, airsim.ImageType.DisparityNormalized, True)
    ]
)

r1 = responses[0]

imgMat = airsim.list_to_2d_float_array(r1.image_data_float, r1.width, r1.height)
img1 = np.flipud(imgMat)
airsim.write_pfm(os.path.normpath("depthPlanner" + '.fpm'), img1)

print(imgMat[:,0])
print(imgMat[:,-1])

imgMatShape = imgMat.shape
print(imgMatShape)
print('sum of col 1', np.sum(imgMat[:,0]))

imgR = np.sum(imgMat, axis=0)
imgRShape = imgR.shape

print('from np.sum', imgR)

print(imgRShape)
# print(imgR)

imgRRe = np.reshape(imgR, (16, 24))
print(imgRRe.shape)
# print(imgRRe)
print(sum(imgRRe[0]))

imgRReR = np.sum(imgRRe, axis=1)
print(imgRReR)
time.sleep(1)

# except KeyboardInterrupt as e:
#     exit()
