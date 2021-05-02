import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# use this to adjust the image size being collected from the environment
# Ensure you read the 'README.md' to understand how to match the image size
imageX = 384
imageY = 128

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-dqn-v2",
                ip_address="127.0.0.1",
                image_shape=(imageX, imageY, 1),
            )
        )
    ]
)
# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

modelPath = './logs_dqn_car_v2/models/timeSteps100000/model/dqn_airsim_car_v2_policy/dqn_airsim_car_v2_policy'
model = DQN.load(modelPath, env=env)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
