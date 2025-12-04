#!/usr/bin/env python3
import sys
import os
import random

# ★ 초기 실행 속도를 위해 라이브러리 지연 로딩

def main():
    model = None
    device = None
    torch = None
    np = None
    
    # ------------------------------------------------------------------
    # 1. 기본 설정
    # ------------------------------------------------------------------
    colors = ['R', 'G', 'B', 'Y']
    symbols = ['1', '2', '3', '4']
    
    valid_cells = []
    forbidden = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}
    for c in ['a','b','c','d','e','f']:
        for r in ['1','2','3','4','5','6']:
            for s in ['-', '+']:
                if (c+r+s) not in forbidden: valid_cells.append(c+r+s)
    
    cell_to_idx = {s: i for i, s in enumerate(valid_cells)}

    def parse_tile(s):
        if s == "X0": return None
        try: return colors.index(s[0]) * 4 + symbols.index(s[1])
        except: return None

    def get_tile_str(tid):
        return colors[tid//4] + symbols[tid%4]

    # 상태 변수
    my_hand = []
    opp_hand = []  # 모델 입력용
    board_state = [-1] * 64
    env_sim = None

    # ------------------------------------------------------------------
    # 2. 하이브리드 결정 로직 (Simple is Best)
    # ------------------------------------------------------------------
    def get_hybrid_action(model, env, obs, mask, device, torch, np):
        # 1. PPO 추천 (직관)
        obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            x, h = model.forward_shared(obs_t)
            b = x.size(0)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(b, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            logits = logits.masked_fill(~mask_t, -1e9)
            ppo_action = torch.argmax(logits, dim=1).item()

        # 2. Greedy 추천 (계산)
        # 복잡한 상대방 수비 로직 제거 -> 오직 내 득점에 집중
        greedy_action, greedy_score = find_best_greedy_move(env)
        
        # 3. PPO 수의 점수 확인
        ppo_score = simulate_score(env, ppo_action)
        
        # 4. 결정
        # Greedy가 10점 이상 더 높을 때만 개입 (PPO의 전략 존중)
        # 점수 차이가 작다면 PPO가 더 좋은 자리를 선점했을 확률이 높음
        if greedy_score > ppo_score + 10.0:
            return greedy_action
        else:
            return ppo_action

    def find_best_greedy_move(env):
        best_val = -9999
        best_action = -1
        
        # 가능한 모든 수 탐색
        for slot in range(len(env.agent_hand)):
            for c_idx in range(64):
                if env.board[c_idx] == -1:
                    action = slot * 64 + c_idx
                    val = simulate_score(env, action)
                    
                    if val > best_val:
                        best_val = val
                        best_action = action
                        
        if best_action == -1: return 0, -9999
        return best_action, best_val

    def simulate_score(env, action):
        # 시뮬레이션 백업
        backup_board = list(env.board)
        backup_hand = list(env.agent_hand)
        backup_filled = env.filled
        
        slot = action // 64
        c_idx = action % 64
        
        if slot >= len(env.agent_hand) or env.board[c_idx] != -1:
            return -9999
            
        # 가상 착수
        tid = env.agent_hand.pop(slot)
        env.board[c_idx] = tid
        env.filled += 1
        
        # 점수 계산 (순수 득점 차이)
        fs, ss, diff = env._compute_scores()
        
        # 복구
        env.board = backup_board
        env.agent_hand = backup_hand
        env.filled = backup_filled
        
        return diff

    # ------------------------------------------------------------------
    # 3. 메인 루프
    # ------------------------------------------------------------------
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            parts = line.strip().split()
            if not parts: continue
            cmd = parts[0]

            if cmd == "READY":
                print("OK")
                sys.stdout.flush()
                
                if model is None:
                    import torch as t
                    import torch.nn as nn
                    import torch.nn.functional as F
                    import numpy as n
                    from connexion_env import ConnexionEnv 
                    
                    torch = t
                    np = n
                    torch.set_num_threads(1)
                    
                    # ───────────────────────── 모델 구조 (학습 코드와 동일) ─────────────────────────
                    class ResBlock(nn.Module):
                        def __init__(self, channels):
                            super().__init__()
                            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                            self.bn1 = nn.BatchNorm2d(channels)
                            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                            self.bn2 = nn.BatchNorm2d(channels)
                        def forward(self, x):
                            return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

                    class ActorCritic(nn.Module):
                        def __init__(self, obs_dim, act_dim):
                            super(ActorCritic, self).__init__()
                            self.conv_in = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
                            self.bn_in = nn.BatchNorm2d(128)
                            self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])
                            
                            # Hand Input: 160 (My + Opp)
                            self.fc_hand = nn.Sequential(nn.Linear(160, 128), nn.ReLU())
                            
                            self.actor_conv = nn.Conv2d(128, 32, 1)
                            self.actor_bn = nn.BatchNorm2d(32)
                            self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim)
                            
                            self.critic_conv = nn.Conv2d(128, 8, 1)
                            self.critic_bn = nn.BatchNorm2d(8)
                            self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256)
                            self.critic_fc2 = nn.Linear(256, 1)

                        def forward_shared(self, obs):
                            b = obs.size(0)
                            board = obs[:, :1024].view(b, 64, 16).permute(0, 2, 1).contiguous().view(b, 16, 8, 8)
                            hand = obs[:, 1024:]
                            x = self.res_blocks(F.relu(self.bn_in(self.conv_in(board))))
                            h = self.fc_hand(hand)
                            return x, h

                    device = torch.device("cpu")
                    # Obs Dim: 1184 (Board 1024 + MyHand 80 + OppHand 80)
                    model = ActorCritic(1184, 320).to(device)
                    
                    model_path = "best_model.pt"
                    if not os.path.exists(model_path): model_path = "final_model.pt"
                    
                    if os.path.exists(model_path):
                        try: model.load_state_dict(torch.load(model_path, map_location=device))
                        except: pass
                    model.eval()
                    env_sim = ConnexionEnv()

                    with torch.no_grad():
                        dummy_obs = torch.zeros(1, 1184).to(device)
                        _ = model.forward_shared(dummy_obs)

            elif cmd == "INIT":
                my_hand = [parse_tile(t) for t in parts[1:6]]
                opp_hand = [parse_tile(t) for t in parts[6:11]]
                board_state = [-1] * 64
                
                if env_sim:
                    env_sim.reset()
                    env_sim.agent_hand = my_hand[:]
                    env_sim.opp_hand = opp_hand[:]
                    env_sim.board = board_state[:]

            elif cmd == "TIME":
                # Obs 생성 (1184 차원)
                obs = np.zeros(1184, dtype=np.float32)
                # Board
                for i in range(64):
                    if board_state[i] != -1: obs[i*16 + board_state[i]] = 1.0
                # My Hand
                for i in range(5):
                    if i < len(my_hand): obs[1024 + i*16 + my_hand[i]] = 1.0
                # Opp Hand
                for i in range(5):
                    if i < len(opp_hand): obs[1104 + i*16 + opp_hand[i]] = 1.0
                
                if env_sim:
                    env_sim.board = board_state[:]
                    env_sim.agent_hand = my_hand[:]
                    env_sim.opp_hand = opp_hand[:]
                    env_sim.filled = sum(1 for x in board_state if x != -1)
                    mask = env_sim.get_smart_action_mask()
                    
                    # 🔥 [수정됨] PPO + 공격형 Greedy
                    action = get_hybrid_action(model, env_sim, obs, mask, device, torch, np)
                else:
                    action = 0

                slot, c_idx = action // 64, action % 64
                
                if board_state[c_idx] != -1:
                    for i in range(64):
                        if board_state[i] == -1: c_idx = i; slot = 0; break
                if slot >= len(my_hand): slot = 0
                
                print(f"PUT {valid_cells[c_idx]} {get_tile_str(my_hand[slot])}")
                sys.stdout.flush()
                
                board_state[c_idx] = my_hand.pop(slot)

            elif cmd == "GET":
                new_t = parse_tile(parts[1])
                if new_t is not None: my_hand.append(new_t)

            elif cmd == "OPP":
                c_idx = cell_to_idx.get(parts[1])
                t1_id = parse_tile(parts[2])
                t2_id = parse_tile(parts[3])
                
                if c_idx is not None:
                    board_state[c_idx] = t1_id
                    if t1_id in opp_hand: opp_hand.remove(t1_id)
                    elif opp_hand: opp_hand.pop(0)
                    if t2_id is not None: opp_hand.append(t2_id)

            elif cmd == "FINISH":
                break

        except Exception:
            break

if __name__ == "__main__":
    main()