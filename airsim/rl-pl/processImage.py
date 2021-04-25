import airsim
from airsim.types import ImageRequest  # pip install airsim
import numpy as np
import os
import time
# for car use CarClient()
client = airsim.CarClient(ip="127.0.0.1")

try:
    while True:
        responses = client.simGetImages ( [
                airsim.ImageRequest(0, airsim.ImageType.DisparityNormalized, True)
            ]
        )

        r1 = responses[0]

        imgMat = airsim.list_to_2d_float_array(r1.image_data_float, r1.width, r1.height)
        # print(imgMat.shape)
        # img1 = np.flipud(imgMat)
        # airsim.write_pfm(os.path.normpath("depthPlanner" + '.fpm'), img1)

        # print(imgMat[:,0])
        # print(imgMat[:,-1])

        # imgMatShape = imgMat.shape
        # print(imgMatShape)
        # print('sum of col 1', np.sum(imgMat[:,0]))

        imgR = np.max(imgMat, axis=0)
        imgRShape = imgR.shape
        print(imgRShape)
        # print('from np.sum', imgR)
        # print(imgRShape)
        # print(imgR)
        steerMap = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.0,
                    -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625]
        diff = np.max(imgR) - np.min(imgR)
        print(diff)
        mVal = np.argmax(imgR)

        if(diff > 0.01):
            print('idx  =', int(mVal/24))
            print(steerMap[int(mVal/24)])
        else:
            print('under threshold')

        print('maxIdx', mVal)
        print('minIdx', np.argmin(imgR))

        # imgRRe = np.reshape(imgR, (16, 24))
        # # print(imgRRe.shape)
        # # print(imgRRe)
        # # print(sum(imgRRe[0]))

        # imgRReR = np.max(imgRRe, axis=1)
        # print(imgRReR.shape)
        # print(np.max(imgRReR) - np.min(imgRReR))
        # print('maxIdx', np.argmax(imgRReR))
        # print('minIdx', np.argmin(imgRReR))


        time.sleep(1)

except KeyboardInterrupt as e:
    exit()
