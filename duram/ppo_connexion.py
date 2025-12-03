# ppo_connexion.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple

from connexion_env import ConnexionEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

# ───────────────────────── 하이퍼파라미터 ─────────────────────────

learning_rate = 3e-4
gamma = 0.99
lmbda = 0.95
eps_clip = 0.1
K_epoch = 4
T_horizon = 32

entropy_coef = 0.01
value_coef = 0.5
reward_scale = 0.1  # 너무 큰 점수 → 살짝 줄여서 학습 안정화


# ───────────────────────── PPO 모델 ─────────────────────────

class PPO(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.data = []

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_pi = nn.Linear(256, act_dim)
        self.fc_v = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def pi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [obs_dim] 또는 [batch, obs_dim]
        return: [act_dim] 또는 [batch, act_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_pi(x)
        prob = F.softmax(logits, dim=-1)
        # 배치 단일일 땐 [act_dim]으로, 배치일 땐 [B, act_dim] 유지
        if prob.size(0) == 1:
            return prob.squeeze(0)
        return prob

    def v(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [obs_dim] 또는 [batch, obs_dim]
        return: [batch] (scalar value per state)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)        # [B, 1]
        return v.squeeze(-1)    # [B]

    def put_data(self, transition):
        # transition: (s, a, r, s_prime, prob_a, done)
        self.data.append(transition)

    def make_batch(self):
        """
        self.data 에 쌓인 rollout을 텐서로 묶는다.
        shapes:
          s:        [T, obs_dim]
          a:        [T]
          r:        [T]
          s_prime:  [T, obs_dim]
          done_mask:[T]   (1.0: not done, 0.0: done)
          prob_a:   [T]   (old π(a|s))
        """
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0.0 if done else 1.0
            done_lst.append(done_mask)

        s = torch.tensor(np.array(s_lst), dtype=torch.float32, device=device)
        a = torch.tensor(a_lst, dtype=torch.int64, device=device)
        r = torch.tensor(r_lst, dtype=torch.float32, device=device)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float32, device=device)
        done_mask = torch.tensor(done_lst, dtype=torch.float32, device=device)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float32, device=device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        if len(self.data) == 0:
            return

        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # -------- 1) TD target & advantage (GAE) : 전부 torch로 계산 --------
        with torch.no_grad():
            v_s = self.v(s)              # [T]
            v_s_prime = self.v(s_prime)  # [T]

            # done_mask: 1.0 (미종료), 0.0 (종료) → 다음 상태 가치 반영 여부
            td_target = r + gamma * v_s_prime * done_mask  # [T]
            delta = td_target - v_s                        # [T]

            # GAE
            T = len(r)
            advantage = torch.zeros_like(delta)
            gae = 0.0
            for t in reversed(range(T)):
                gae = delta[t] + gamma * lmbda * gae * done_mask[t]
                advantage[t] = gae

        # advantage 정규화
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        td_target = td_target.detach()  # critic target은 고정

        # -------- 2) PPO 업데이트 --------
        for _ in range(K_epoch):
            pi = self.pi(s)  # [T, act_dim]
            if pi.dim() == 1:
                pi = pi.unsqueeze(0)  # 혹시라도 T=1일 때 대비

            # a: [T] → [T, 1]로 인덱싱
            pi_a = pi.gather(1, a.unsqueeze(1)).squeeze(1)  # [T]
            old_prob_a = prob_a  # [T]

            ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(old_prob_a + 1e-8))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            v_s = self.v(s)  # [T]
            critic_loss = F.smooth_l1_loss(v_s, td_target)

            entropy = -(pi * (pi + 1e-8).log()).sum(dim=1).mean()

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ───────────────────────── Action 샘플링 (마스크 적용) ─────────────────────────

def sample_valid_action(raw_prob: torch.Tensor, valid_mask: np.ndarray) -> Tuple[int, float]:
    """
    raw_prob: [act_dim] (softmax 결과, 아직 마스크 미적용)
    valid_mask: [act_dim] bool

    - invalid action은 확률 0
    - 남은 확률들 normalize해서 샘플링
    - PPO용 old_prob_a는 '마스크 적용 전(raw_prob)의 값'을 사용
    """
    prob_np = raw_prob.detach().cpu().numpy().astype(np.float64)

    if valid_mask.shape[0] != prob_np.shape[0]:
        raise ValueError("valid_mask length and prob length mismatch")

    prob_np[~valid_mask] = 0.0
    total = prob_np.sum()

    if total <= 0.0 or not np.isfinite(total):
        # 모든 확률이 0이면 valid 중에서 균등 랜덤
        valid_indices = np.flatnonzero(valid_mask)
        a = int(np.random.choice(valid_indices))
    else:
        prob_np /= total
        a = int(np.random.choice(len(prob_np), p=prob_np))

    old_prob_a = float(raw_prob[a].item())
    return a, old_prob_a


# ───────────────────────── 학습 루프 ─────────────────────────

def train_ppo(num_episodes: int = 1000) -> PPO:
    env = ConnexionEnv(seed=0, opponent="sample")
    obs, info = env.reset()

    obs_dim = env.obs_dim
    act_dim = env.action_size
    print(f"obs_dim: {obs_dim} act_dim: {act_dim}")

    model = PPO(obs_dim, act_dim)
    env = ConnexionEnv(seed=123, opponent="sample")
    print_interval = 50
    score_window = []

    for n_epi in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        steps_in_ep = 0

        while not done:
            for _ in range(T_horizon):
                s = torch.from_numpy(obs).float().to(device)

                with torch.no_grad():
                    raw_prob = model.pi(s)

                valid_mask = env.get_valid_action_mask()
                if not valid_mask.any():
                    done = True
                    break

                a, old_prob_a = sample_valid_action(raw_prob, valid_mask)

                obs_next, r_env, done, truncated, info = env.step(a)

                # reward scaling (학습 안정화를 위해 env 점수를 조금 줄여서 사용)
                r = r_env * reward_scale

                model.put_data((obs, a, r, obs_next, old_prob_a, done))

                obs = obs_next
                episode_return += r_env
                steps_in_ep += 1

                if done:
                    break

            model.train_net()

        score_window.append(episode_return)
        if len(score_window) > print_interval:
            score_window.pop(0)

        if n_epi % print_interval == 0:
            avg_ret = sum(score_window) / len(score_window)
            print(
                f"[Episode {n_epi}] return={episode_return:.3f}, "
                f"avg(last {len(score_window)}): {avg_ret:.3f}, steps={steps_in_ep}"
            )

    torch.save(model.state_dict(), "ppo_connexion.pt")
    print("Saved model to ppo_connexion.pt")

    return model


# ───────────────────────── 테스트 ─────────────────────────

def test_run(model: PPO, num_episodes: int = 200):
    env = ConnexionEnv(seed=123, opponent="sample")
    wins = 0
    draws = 0
    losses = 0

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            s = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                raw_prob = model.pi(s)

            valid_mask = env.get_valid_action_mask()
            prob_np = raw_prob.detach().cpu().numpy()
            prob_np[~valid_mask] = -1e9  # invalid은 아주 작은 값으로

            a = int(np.argmax(prob_np))

            obs, r_env, done, truncated, info = env.step(a)
            total_reward += r_env
            steps += 1

        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

        print(f"[Test Episode {ep}] return={total_reward:.3f}, steps={steps}")

    print(f"Test result: {wins}W {draws}D {losses}L / {num_episodes} episodes")


if __name__ == "__main__":
    model = train_ppo(num_episodes=10000)

    print("\n=== Test run with trained policy ===")
    test_run(model, num_episodes=200)
