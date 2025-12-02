# eval_random_vs_ppo.py
import numpy as np
import torch
from torch.distributions import Categorical

from connexion_env import ConnexionEnv
from train_ppo import PolicyValueNet   # 같은 파일/클래스 이름 기준

def load_policy(path: str, env: ConnexionEnv):
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_size
    model = PolicyValueNet(obs_dim, action_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def ppo_policy_action(model, env, obs):
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_t)
        mask_np = env.get_valid_action_mask().astype(np.bool_)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        from train_ppo import masked_categorical
        dist = masked_categorical(logits, mask)
        action = dist.sample()
    return int(action.item())

def random_policy_action(env):
    mask = env.get_valid_action_mask()
    valid_actions = np.where(mask)[0]
    return int(np.random.choice(valid_actions))

def play_one_game(model, first_is_ppo: bool = True):
    env = ConnexionEnv()
    obs, _ = env.reset()
    done = False

    # player0 = PPO or Random (first_is_ppo에 따라)
    current_player = env.current_player  # 0 또는 1

    # 마지막 reward는 _final_reward에서 player0 기준이므로
    # player0가 PPO인지 Random인지 기억해야 한다.
    ppo_as_player0 = first_is_ppo

    last_reward = 0.0
    while not done:
        if current_player == 0:  # player0 차례
            if ppo_as_player0:
                action = ppo_policy_action(model, env, obs)
            else:
                action = random_policy_action(env)
        else:  # player1 차례
            if ppo_as_player0:
                action = random_policy_action(env)
            else:
                action = ppo_policy_action(model, env, obs)

        obs, reward, done, trunc, info = env.step(action)
        last_reward = reward
        current_player = env.current_player

    # _final_reward는 player0 관점에서 리턴하므로,
    # ppo_as_player0가 True면 reward 그대로,
    # False면 반대로 뒤집어 주면 "ppo 관점" reward가 됨
    if not ppo_as_player0:
        last_reward = -last_reward

    return last_reward  # +1 승, 0 무, -1 패

def main():
    env = ConnexionEnv()
    model = load_policy("ppo_connexion.pt", env)

    n_games = 50
    results = []
    for i in range(n_games):
        # 선후공 역할 바꿔가며
        first_is_ppo = (i % 2 == 0)
        r = play_one_game(model, first_is_ppo=first_is_ppo)
        results.append(r)
        print(f"Game {i}: reward(ppo) = {r}")

    results = np.array(results)
    win = np.sum(results > 0)
    draw = np.sum(results == 0)
    lose = np.sum(results < 0)
    print(f"PPO vs Random over {n_games} games:")
    print(f"  win={win}, draw={draw}, lose={lose}")

if __name__ == "__main__":
    main()
