from gym.envs.registration import register
register(
    id='DroneNavigation-v0',
    entry_point='drone_envs.envs:DroneNavigationV0'
)

register(
    id='DroneNavigation-v1',
    entry_point='drone_envs.envs:DroneNavigationV1'
)

register(
    id='DroneNavigationMultiFormation-v0',
    entry_point=lambda** kwargs: DroneNavigationMulti(**kwargs)
)