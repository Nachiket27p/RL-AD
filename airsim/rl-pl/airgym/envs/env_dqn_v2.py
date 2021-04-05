import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image

cols = 9
rows = 9
actionMatrix = [[None for i in range(cols)] for j in range(rows)]

gasbreak = -1.0  # negative means break. pos means gas
for i in range(rows):
    steer = -1.0
    for j in range(cols):
        actionMatrix[i][j] = (steer, gasbreak)
        steer += 0.25
    gasbreak += 0.25


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
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthVis, True, False
        )

        # accessing airsim API to control the car
        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.state["pose"] = None
        self.state["prev_pose"] = None
        self.state["collision"] = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    # need to define more states for finer control
    def _do_action(self, action):
        steer, gasbreak = actionMatrix[action // rows][action % cols]
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

        return im_final.reshape([self.image_shape[0], self.image_shape[1], 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses)

        self.car_state = self.car.getCarState()
        collision = self.car.simGetCollisionInfo().has_collided

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = collision

        return image

    def _compute_reward(self):
        MAX_SPEED = 50
        MIN_SPEED = 5
        thresh_dist = 3.5
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
                np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1])))
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = math.exp(-beta * dist) - 0.5
            reward_speed = ((self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) - 0.5
            reward = reward_dist + reward_speed

        # print(reward)
        done = 0
        if reward < -2:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed < 0.1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(40)
        return self._get_obs()
