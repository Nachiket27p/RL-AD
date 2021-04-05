from gym.envs.registration import register

register(
    id="airsim-car-sample-v0", entry_point="airgym.envs:AirSimCarEnv",
)

register(
    id="airsim-car-dqn-v2", entry_point="airgym.envs:AirSimCarEnvDQNV2",
)

register(
    id="airsim-car-qrdqn-v0", entry_point="airgym.envs:AirSimCarEnvQRDQN",
)
