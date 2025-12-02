# train_ppo.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from connexion_env import ConnexionEnv  # 네가 만든 env 모듈 이름에 맞게 수정


# ==============================
# 1. Policy+Value 네트워크 정의
# ==============================

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        return: logits (batch, action_dim), value (batch,)
        """
        x = self.shared(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


# ==============================
# 2. Masked Categorical 유틸
# ==============================

def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """
    logits: (batch, action_dim)
    mask: (batch, action_dim) bool or 0/1 tensor (True/1이면 valid)

    invalid action은 -1e9로 눌러서 softmax에서 거의 0 확률로.
    """
    # mask를 float로 캐스팅해서 invalid는 매우 작은 값으로
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return Categorical(logits=masked_logits)


# ==============================
# 3. GAE advantage 계산
# ==============================

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """
    rewards, values, dones: 길이 T의 1D numpy array 또는 torch tensor
    반환: advantages, returns
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t]  # 마지막 step은 bootstrap 안 한다고 가정
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv

    returns = advantages + values
    return advantages, returns


# ==============================
# 4. PPO 학습 루프
# ==============================

def train_ppo(
    total_timesteps=100_000,
    rollout_steps=1024,
    minibatch_size=256,
    ppo_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    learning_rate=3e-4,
    seed=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env 초기화
    env = ConnexionEnv()
    obs, info = env.reset(seed=seed)

    obs_dim = obs.shape[0]
    action_dim = env.action_size

    policy = PolicyValueNet(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # 학습 loop
    global_step = 0
    update_idx = 0

    while global_step < total_timesteps:
        # rollout 저장 버퍼
        obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        actions_buf = np.zeros(rollout_steps, dtype=np.int64)
        logprobs_buf = np.zeros(rollout_steps, dtype=np.float32)
        rewards_buf = np.zeros(rollout_steps, dtype=np.float32)
        dones_buf = np.zeros(rollout_steps, dtype=np.float32)
        values_buf = np.zeros(rollout_steps, dtype=np.float32)

        for t in range(rollout_steps):
            global_step += 1
            obs_buf[t] = obs

            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = policy(obs_tensor)
                # valid mask 가져오기
                valid_mask_np = env.get_valid_action_mask().astype(np.bool_)
                valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0).to(device)
                dist = masked_categorical(logits, valid_mask)
                action = dist.sample()
                logprob = dist.log_prob(action)

            actions_buf[t] = action.item()
            logprobs_buf[t] = logprob.item()
            values_buf[t] = value.item()
            dones_buf[t] = 0.0  # step()에서 done 받기 전까지 0

            # env step
            next_obs, reward, done, truncated, info = env.step(action.item())
            rewards_buf[t] = reward

            # 에피소드 끝나면 reset
            if done or truncated:
                dones_buf[t] = 1.0
                next_obs, info = env.reset()

            obs = next_obs

        # rollout 하나 끝 -> GAE 계산
        advantages, returns = compute_gae(
            rewards=rewards_buf,
            values=values_buf,
            dones=dones_buf,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # numpy -> torch
        obs_tensor = torch.from_numpy(obs_buf).float().to(device)
        actions_tensor = torch.from_numpy(actions_buf).long().to(device)
        old_logprobs_tensor = torch.from_numpy(logprobs_buf).float().to(device)
        advantages_tensor = torch.from_numpy(advantages).float().to(device)
        returns_tensor = torch.from_numpy(returns).float().to(device)

        # advantage 정규화 (stabilize)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # ==========================
        # PPO update (여러 epoch)
        # ==========================

        batch_size = rollout_steps
        indices = np.arange(batch_size)

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_logprobs = old_logprobs_tensor[mb_idx]
                mb_advantages = advantages_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]

                logits, values = policy(mb_obs)

                # 여기서도 마스크 적용 필요: 
                # 단, rollout 때의 valid_mask를 따로 저장해두지 않았으니
                # 실제 구현에서는 valid_mask도 버퍼에 같이 저장하는 게 가장 안전하다.
                # 스켈레톤에선 "logits로 그대로" dist를 만들지만,
                # 너가 구현할 땐 rollout 때 valid_mask를 같이 저장해와서 masked_categorical 써 주면 좋음.
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ratio
                ratio = torch.exp(logprobs - mb_old_logprobs)

                # PPO clip objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = ((values - mb_returns) ** 2).mean()

                # 전체 loss
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()

        update_idx += 1

        # 간단한 로그 출력
        avg_return = returns_tensor.mean().item()
        print(
            f"[Update {update_idx}] global_step={global_step} "
            f"avg_return={avg_return:.3f}"
        )

    # 학습 끝난 후 모델 저장
    torch.save(policy.state_dict(), "ppo_connexion.pt")
    print("Training finished. Model saved to ppo_connexion.pt")


if __name__ == "__main__":
    train_ppo()
