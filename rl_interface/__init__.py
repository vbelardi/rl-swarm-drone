from gymnasium.envs.registration import register

register(
    id="rl_interface/GridWorld-v0",
    entry_point="rl_interface.rl_interface:GridWorldEnv",
)
