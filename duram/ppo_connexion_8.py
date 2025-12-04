# ppo_connexion.py 7 기준으로 gpt가 refactoring
 
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from connexion_env_7 import ConnexionEnv  # ← 너가 사용 중인 env 파일 이름에 맞게 수정

# ───────────────────────────── 설정값 ─────────────────────────────

SEED = 42

MAX_EPISODES = 10_000

# 상대 난이도 커리큘럼
RANDOM_PHASE_END = 2_000     # 0~2000: 완전 랜덤
GREEDY_PHASE_END = 7_000     # 2000~7000: 랜덤→Greedy 점점 증가
FINAL_GREEDY_PROB = 0.8      # 이후엔 Greedy 80%, Random 20%

# Smart Mask 커리큘럼 (얼마나 자주 Smart Mask만 쓸지)
SMART_PROB_MAX = 0.8
SMART_PROB_RAMP_END = 4_000  # 0~4000: 0 → 0.8까지 증가

# PPO 하이퍼파라미터
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95         # GAE lambda
K_EPOCH = 4
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 4_096
MINI_BATCH_SIZE = 256

ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────────── 유틸: 시드 고정 ─────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────────────────────────── 모델 정의 (ResNet 스타일) ─────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ActorCritic(nn.Module):
    """
    관측:
      - 보드: 64칸 × 16채널 one-hot → 1024
      - 내 손패: 5장 × 16 → 80
      - 상대 손패: 5장 × 16 → 80
      총 obs_dim = 1184 (env.obs_dim 사용)
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        # 보드(16×8×8)를 위한 인코더
        self.conv_in = nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
        )

        # 손패(내+상대) 160차원
        self.fc_hand = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
        )

        # Actor 헤드
        self.actor_conv = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.actor_bn = nn.BatchNorm2d(32)
        self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim)

        # Critic 헤드
        self.critic_conv = nn.Conv2d(128, 8, kernel_size=1, bias=False)
        self.critic_bn = nn.BatchNorm2d(8)
        self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def _encode(self, obs: torch.Tensor):
        """
        obs: (B, 1184)
        반환: board_feat: (B, 128, 8, 8), hand_feat: (B, 128)
        """
        bsz = obs.size(0)
        board_flat = obs[:, :1024]           # (B, 1024)
        hand_flat = obs[:, 1024:]           # (B, 160)

        # 1024 = 64 * 16  → (B, 64, 16) → (B, 16, 64) → (B, 16, 8, 8)
        board = board_flat.view(bsz, 64, 16).permute(0, 2, 1).contiguous()
        board = board.view(bsz, 16, 8, 8)

        x = F.relu(self.bn_in(self.conv_in(board)))
        x = self.res_blocks(x)              # (B, 128, 8, 8)
        h = self.fc_hand(hand_flat)         # (B, 128)
        return x, h

    def forward(self, obs: torch.Tensor):
        """
        반환:
          logits: (B, act_dim)
          value:  (B, 1)
        """
        x, h = self._encode(obs)
        bsz = obs.size(0)

        # Actor
        pol = F.relu(self.actor_bn(self.actor_conv(x)))  # (B, 32, 8, 8)
        pol = pol.view(bsz, -1)                          # (B, 32*8*8)
        pol = torch.cat([pol, h], dim=1)                 # (B, 32*8*8 + 128)
        logits = self.actor_fc(pol)                      # (B, act_dim)

        # Critic
        val_feat = F.relu(self.critic_bn(self.critic_conv(x)))  # (B, 8, 8, 8)
        val_feat = val_feat.view(bsz, -1)                       # (B, 8*8*8)
        val_feat = torch.cat([val_feat, h], dim=1)              # (B, 8*8*8 + 128)
        val_feat = F.relu(self.critic_fc1(val_feat))
        value = self.critic_fc2(val_feat)                       # (B, 1)

        return logits, value

    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        obs:  (1, obs_dim)
        mask: (1, act_dim) bool
        """
        logits, value = self.forward(obs)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        """
        PPO 업데이트용 평가 함수
        """
        logits, value = self.forward(obs)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value


# ───────────────────────────── 마스크 빌더 (Smart + Valid 혼합) ─────────────────────────────

def build_action_mask(env: ConnexionEnv, smart_prob: float) -> np.ndarray:
    """
    - 항상 규칙상 가능한 valid mask를 기반으로 하고
    - smart_prob 확률로 Smart Mask를 사용 (그 외엔 valid만)
    """
    valid = env.get_valid_action_mask()          # 규칙상 가능한 모든 수 (bool[n_actions])
    if valid.sum() == 0:
        return valid

    smart = env.get_smart_action_mask()          # 휴리스틱 기반 "좋은 수들"
    # 혹시나를 대비해서 valid와 AND
    smart = np.logical_and(smart, valid)

    if smart_prob <= 0.0 or smart.sum() == 0:
        return valid

    if np.random.rand() < smart_prob:
        return smart
    else:
        return valid


# ───────────────────────────── PPO 업데이트 ─────────────────────────────

def update_ppo(model: ActorCritic, optimizer, memory):
    obs = torch.tensor(np.array(memory["obs"]), dtype=torch.float32, device=DEVICE)
    acts = torch.tensor(np.array(memory["act"]), dtype=torch.long, device=DEVICE)
    old_logp = torch.tensor(np.array(memory["logp"]), dtype=torch.float32, device=DEVICE)
    rews = torch.tensor(np.array(memory["rew"]), dtype=torch.float32, device=DEVICE)
    masks = torch.tensor(np.array(memory["mask"]), dtype=torch.bool, device=DEVICE)
    vals = torch.tensor(np.array(memory["val"]), dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(np.array(memory["done"]), dtype=torch.bool, device=DEVICE)

    # ----- GAE 계산 -----
    vals_np = vals.detach().cpu().numpy()
    rews_np = rews.detach().cpu().numpy()
    dones_np = dones.detach().cpu().numpy()

    N = len(rews_np)
    adv_np = np.zeros(N, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(N)):
        next_v = vals_np[t + 1] if t + 1 < N else 0.0
        not_done = 0.0 if dones_np[t] else 1.0
        delta = rews_np[t] + GAMMA * next_v * not_done - vals_np[t]
        gae = delta + GAMMA * LAMBDA * not_done * gae
        adv_np[t] = gae

    advantages = torch.tensor(adv_np, dtype=torch.float32, device=DEVICE)
    returns = advantages + vals

    # 정규화
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ----- 미니배치 PPO 업데이트 -----
    N_total = obs.size(0)
    indices = np.arange(N_total)

    for _ in range(K_EPOCH):
        np.random.shuffle(indices)
        for start in range(0, N_total, MINI_BATCH_SIZE):
            idx = indices[start:start + MINI_BATCH_SIZE]
            if len(idx) == 0:
                continue

            batch_obs = obs[idx]
            batch_acts = acts[idx]
            batch_old_logp = old_logp[idx]
            batch_adv = advantages[idx]
            batch_returns = returns[idx]
            batch_masks = masks[idx]

            new_logp, entropy, values = model.evaluate_actions(
                batch_obs, batch_masks, batch_acts
            )

            ratio = torch.exp(new_logp - batch_old_logp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * batch_adv

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)
            entropy_loss = -entropy.mean()

            loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


# ───────────────────────────── 평가 함수 (랜덤 상대) ─────────────────────────────

def evaluate_policy(model: ActorCritic, env: ConnexionEnv, episodes: int = 200):
    print(f"\n=== Evaluate vs Random ({episodes} episodes) ===")
    model.eval()

    env.set_greedy_prob(0.0)  # 완전 랜덤 상대

    wins = losses = draws = 0
    total_diff = 0.0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}

        while not done:
            # 테스트에서는 Smart Mask 100% 사용 + argmax
            mask = env.get_smart_action_mask()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_t)
                logits = logits.masked_fill(~mask_t, -1e9)
                action = torch.argmax(logits, dim=1)

            obs, _, done, _, info = env.step(action.item())

        diff = float(info.get("diff", 0.0))
        total_diff += diff
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1

    win_rate = 100.0 * wins / episodes
    avg_diff = total_diff / episodes if episodes > 0 else 0.0

    print(f"W / D / L = {wins} / {draws} / {losses}")
    print(f"Win rate: {win_rate:.1f}%   Avg diff: {avg_diff:.2f}\n")

    model.train()


# ───────────────────────────── 학습 루프 ─────────────────────────────

def train():
    set_seed(SEED)
    env = ConnexionEnv(seed=SEED)

    print(f"Using device: {DEVICE}")
    print(f"obs_dim: {env.obs_dim}  act_dim: {env.action_size}")

    model = ActorCritic(env.obs_dim, env.action_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    memory = {
        "obs": [],
        "act": [],
        "logp": [],
        "rew": [],
        "mask": [],
        "done": [],
        "val": [],
    }

    score_q = deque(maxlen=100)
    diff_q = deque(maxlen=100)
    best_avg_diff = -1e9

    for ep in range(1, MAX_EPISODES + 1):
        # ── 1) 상대 난이도 설정 (greedy_prob) ──
        if ep <= RANDOM_PHASE_END:
            greedy_prob = 0.0  # 완전 랜덤
        elif ep <= GREEDY_PHASE_END:
            prog = (ep - RANDOM_PHASE_END) / (GREEDY_PHASE_END - RANDOM_PHASE_END)
            greedy_prob = 0.2 + (FINAL_GREEDY_PROB - 0.2) * prog  # 0.2 → 0.8
        else:
            greedy_prob = FINAL_GREEDY_PROB

        env.set_greedy_prob(float(greedy_prob))

        # ── 2) Smart Mask 사용 확률 설정 ──
        if ep <= SMART_PROB_RAMP_END:
            smart_prob = SMART_PROB_MAX * (ep / SMART_PROB_RAMP_END)
        else:
            smart_prob = SMART_PROB_MAX

        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        info = {}

        while not done:
            mask_np = build_action_mask(env, smart_prob)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask_np, dtype=torch.bool, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                action, logp, entropy, value = model.act(obs_t, mask_t)
            a = int(action.item())

            next_obs, rew, done, _, info = env.step(a)

            # 메모리에 저장
            memory["obs"].append(obs)
            memory["act"].append(a)
            memory["logp"].append(float(logp.item()))
            memory["rew"].append(float(rew))
            memory["mask"].append(mask_np)
            memory["done"].append(bool(done))
            memory["val"].append(float(value.item()))

            obs = next_obs
            ep_ret += rew

            # 일정 스텝마다 PPO 업데이트
            if len(memory["obs"]) >= UPDATE_TIMESTEP:
                update_ppo(model, optimizer, memory)
                for k in memory.keys():
                    memory[k].clear()

        score_q.append(ep_ret)
        if "diff" in info:
            diff_q.append(float(info["diff"]))

        # 로그 + 모델 저장
        if ep % 50 == 0:
            avg_ret = float(np.mean(score_q)) if score_q else 0.0
            avg_diff = float(np.mean(diff_q)) if diff_q else 0.0

            print(
                f"[Ep {ep:05d}] "
                f"greedy={greedy_prob:.2f}  smart={smart_prob:.2f}  "
                f"avg_ret={avg_ret:.3f}  avg_diff={avg_diff:.1f}"
            )

            # Greedy 상대 기준으로 평균 diff가 좋아지면 best_model 저장
            if avg_diff > best_avg_diff and ep > 500:
                best_avg_diff = avg_diff
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  --> New best_model.pt saved (avg_diff={best_avg_diff:.1f})")

    # 최종 모델 저장 + 평가
    torch.save(model.state_dict(), "final_model.pt")
    print("\nTraining finished. Saved final_model.pt")

    evaluate_policy(model, env, episodes=200)


if __name__ == "__main__":
    train()
