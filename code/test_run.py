from connexion_env import ConnexionEnv

env = ConnexionEnv()
obs, info = env.reset()
print("obs shape:", obs.shape)
print("action_size:", env.action_size)

for i in range(10):
    mask = env.get_valid_action_mask()
    # 유효한 액션 아무거나 하나 골라봄
    import numpy as np
    valid_actions = np.where(mask)[0]
    a = int(valid_actions[0])
    obs, r, done, trunc, info = env.step(a)
    print(i, "reward:", r, "done:", done)
    if done:
        obs, info = env.reset()
