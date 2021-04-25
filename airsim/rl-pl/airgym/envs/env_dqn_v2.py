import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image

###############################################################################

cols = 17
rows = 9
# selects the middle element in the last row
origin = (rows * cols) - math.ceil(float(cols) / 2)
actionMatrix = [[None for i in range(cols)] for j in range(rows)]

gasbreak = 1.0  # pos means gas, zero means break
for i in range(rows):
    steer = -0.5
    for j in range(cols):
        actionMatrix[i][j] = (steer, gasbreak)
        steer += 1.0 / (cols - 1)
    gasbreak -= 1.0 / (rows - 1)

# used dure cost evaluation to ensure that agent can be penalized when if it
# takes the wrong action when its is about to collide with an object
steerMap = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5,
            0.5, # This middle value ensures that the agent turns when an obsticle is in the center
            -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625]



###############################################################################

class AirSimCarEnvDQNV2(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(int(rows * cols))

        # image being requested
        self.image_request = [
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, True),
            airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, True)
        ]

        # accessing airsim API to control the car
        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.state["pose"] = None
        self.state["prev_pose"] = None
        self.state["collision"] = None
        self.state["prev_steer"] = 0.0
        self.state["steer"] = 0.0
        self.state["prev_disNorSteer"] = 0.0
        self.dispNormMaxPool = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

        # reset the custom states
        self.state["prev_steer"] = 0.0
        self.state["steer"] = 0.0
        self.state["prev_disNorSteer"] = 0.0
        self.dispNormMaxPool = None

        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    # need to define more states for finer control
    def _do_action(self, action):
        # print(action, action // cols, action % cols)
        steer, gasbreak = actionMatrix[action // cols][action % cols]
        
        # save the previous steering action previous steer state
        # update the current steer action to the new one acquired from
        #   the action matrix.
        self.state['prev_steer'] = self.state['steer']
        self.state['steer'] = steer

        self.car_controls.steering = steer
        if gasbreak == 0:
            self.car_controls.brake = 1.0
            self.car_controls.throttle = 0.0
        else:
            self.car_controls.brake = 0.0
            self.car_controls.throttle = gasbreak

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.convert("L"))

        # process the disparity normalized image
        dn = responses[1]
        idn = airsim.list_to_2d_float_array(dn.image_data_float, dn.width, dn.height)
        # max pool each column so the (384, 128) image becomes a (384,) vector
        self.dispNormMaxPool = np.max(idn, axis=0)


        return im_final.reshape([self.image_shape[0], self.image_shape[1], 1])

    def _get_obs(self):
        responses = self.car.simGetImages(self.image_request)
        image = self.transform_obs(responses)

        self.car_state = self.car.getCarState()
        collision = self.car.simGetCollisionInfo().has_collided

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = collision

        return image

    def _compute_reward(self):
        MAX_SPEED = 15
        MIN_SPEED = 2
        thresh_dist = 2
        beta = 3

        z = 0
        pts = [
            np.array([0, -1, z]),
            np.array([130, -1, z]),
            np.array([130, 125, z]),
            np.array([0, 125, z]),
            np.array([0, -1, z]),
            np.array([130, -1, z]),
            np.array([130, -128, z]),
            np.array([0, -128, z]),
            np.array([0, -1, z]),
        ]
        pd = self.state["pose"].position
        car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))) / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print('dist:', dist)
        # print('speed:', self.car_state.speed)
        
        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = math.exp(-beta * dist) - 0.5
            reward_speed = ((self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) - 0.5
            reward = reward_dist + reward_speed
            # print('reward(dist, speed):', reward_dist, reward_speed)

        #######################################################################

        # modify the reward based on the steering action taken by the agent
        # if the previous steering action was opposite to the one expected from
        # the the one obtained from analyzing the dispariity normalized image
        dnmpMax = np.max(self.dispNormMaxPool)
        dnmpMin = np.min(self.dispNormMaxPool)
        # compute teh difference between the largest and smallest
        # this is used to proide a threshold before an action is performed
        diff = dnmpMax - dnmpMin
        # find the index of the max value and uses that to index into the steer map
        # to ensure that a steering operation is necessary and if the the agent should
        # be penalized or not
        # The reward should only be adjusted if the agent is wondering close to
        #   objects it can run into
        mVal = np.argmax(self.dispNormMaxPool)
        dnSteer = steerMap[int(mVal/24)]
        if(diff > 0.01):
            preDisDiff = abs(self.state["prev_disNorSteer"] - self.state["prev_steer"])
            curDisDiff = abs(dnSteer - self.state["steer"])
            reward -= curDisDiff
            if(preDisDiff < curDisDiff):
                reward -= preDisDiff
        else:
            # no need to modify the reward, the agent is not expected to perform a specific action
            reward -= 0

        #######################################################################

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed < 0.2:
                done = 1
        if self.state["collision"]:
            done = 1

        # save the current disparity normalized to the preve state to use
        # in the next round
        self.state["prev_disNorSteer"] = dnSteer
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(origin)
        return self._get_obs()
