from connexion_env import ConnexionEnv
env = ConnexionEnv()
obs, info = env.reset()
print(obs.shape, env.action_size)
mask = env.get_valid_action_mask()
print(mask.sum(), "valid actions")
