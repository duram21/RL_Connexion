#!/usr/bin/env python3
import sys
import os
import random
import copy

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
    my_hand = []      # 내 손패
    opp_hand = []     # 상대 손패 (완벽 추적)
    board_state = [-1] * 64
    env_sim = None

    # ------------------------------------------------------------------
    # 2. Minimax 로직 (공격 + 수비)
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
    # 수정된 하이브리드 액션 결정 함수 (PPO + Minimax 융합)
    # ------------------------------------------------------------------
    def get_minimax_action(model, env, obs, mask, device, torch, np):
        # 1. PPO에게 물어보기 ("어디가 좋아 보여?")
        obs_t = torch.tensor(obs).float().unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            x, h = model.forward_shared(obs_t)
            b = x.size(0)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(b, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            logits = logits.masked_fill(~mask_t, -1e9)
            
            # 확률이 가장 높은 수 (PPO의 원픽)
            ppo_action = torch.argmax(logits, dim=1).item()

        # 2. Minimax 계산 ("그 수가 안전한지, 더 좋은 수는 없는지 계산해볼게")
        best_minimax_action, best_net_score = find_best_minimax_move(env)
        
        # 3. PPO가 추천한 수의 '실제 점수' 계산
        # (PPO 추천 수가 Minimax 기준 몇 점인지 확인)
        
        # 가상 착수 및 점수 계산을 위해 잠시 로직 수행
        slot = ppo_action // 64
        c_idx = ppo_action % 64
        
        # 만약 PPO가 둔 수가 빈칸이 아니거나 이상하면 -> 바로 Minimax 따라감
        if slot >= len(env.agent_hand) or env.board[c_idx] != -1:
            return best_minimax_action

        # PPO 수의 가치(점수) 계산
        # (find_best_minimax_move 안의 로직을 재사용하거나 약식 계산)
        # 여기서는 단순화를 위해 simulate_score 재호출 대신 개념적으로 설명:
        
        # 간단 비교: PPO가 추천한 수도 Minimax 탐색 과정에서 점수가 계산되었을 것임.
        # 하지만 코드 구조상 따로 저장 안 했으니, 여기서 PPO 수만 따로 검증.
        
        # 백업
        bak_board = list(env.board)
        bak_filled = env.filled
        bak_hand = list(env.agent_hand)
        
        # PPO 착수
        tid = env.agent_hand.pop(slot)
        env.board[c_idx] = tid
        env.filled += 1
        
        fs, ss, diff = env._compute_scores()
        
        # 상대방의 최선의 반격 예측 (Minimax와 동일한 기준)
        max_opp_gain = 0
        empty_cells = [i for i, x in enumerate(env.board) if x == -1]
        
        if env.opp_hand:
            for opp_slot in range(len(env.opp_hand)):
                ot = env.opp_hand[opp_slot]
                for oc in empty_cells:
                    # 상대 착수 시뮬
                    if env.board[oc] == -1: # 체크
                        val = env._local_score(oc, ot, is_first=False)
                        if val > max_opp_gain: max_opp_gain = val
        
        ppo_net_score = diff - (max_opp_gain * 1.5)
        
        # 원복
        env.board = bak_board
        env.agent_hand = bak_hand
        env.filled = bak_filled

        # 4. 최종 결정 (밸런스 조절)
        # PPO의 수가 Minimax 최선수보다 3점 이내로 차이나면 PPO를 존중 (큰 그림)
        # 3점 이상 손해라면 "PPO야, 이건 실수다" 하고 Minimax를 따름
        
        if ppo_net_score >= best_net_score - 3.0:
            return ppo_action
        else:
            return best_minimax_action
        

    def find_best_minimax_move(env):
        """
        Max(내 점수 - Max(상대 점수)) 를 구하는 함수
        """
        best_net_score = -9999
        best_action = 0
        
        # [최적화] 빈칸 인덱스 미리 확보
        empty_cells = [i for i, x in enumerate(env.board) if x == -1]
        
        # 1. 내가 둘 수 있는 모든 수 탐색
        for slot in range(len(env.agent_hand)):
            my_tid = env.agent_hand[slot]
            
            for c_idx in empty_cells:
                # -- 가상 착수 (나) --
                env.board[c_idx] = my_tid
                env.filled += 1
                
                # 내 득점 계산
                fs, ss, diff = env._compute_scores() 
                # diff = (내 점수 - 상대 점수) -> 현재 상황 점수
                
                # 2. 상대방의 최선의 수 예측 (Counter-Attack)
                # 상대는 자신의 점수(ss)를 높이거나, 내 점수(fs)를 깎으려 할 것임
                # 여기서는 상대도 Greedy하게 자기 점수를 최대화한다고 가정
                
                max_opp_gain = 0
                
                # 상대 손패가 비어있으면(후반) 공격 불가
                if env.opp_hand:
                    # 상대가 둘 수 있는 빈칸 (내가 둔 곳 제외)
                    opp_empty_cells = [i for i in empty_cells if i != c_idx]
                    
                    for opp_slot in range(len(env.opp_hand)):
                        opp_tid = env.opp_hand[opp_slot]
                        for opp_c in opp_empty_cells:
                            # -- 가상 착수 (상대) --
                            env.board[opp_c] = opp_tid
                            
                            # 상대 입장에서 점수 계산
                            # _compute_scores는 항상 (First, Second, Diff) 반환
                            # 상대는 Second Player이므로 Second 점수를 올리는 게 목표
                            # (간략화를 위해 로컬 점수가 아닌 전체 점수 재계산 - 정확도UP, 속도DOWN)
                            # 속도가 걱정되면 여기서 _local_score 로 대체 가능
                            
                            # 여기서는 정확성을 위해 전체 계산 (Python 속도상 빡셀 수 있음 -> 최적화 필요)
                            # 일단 로컬 스코어(Greedy)로 근사치 계산 (속도 확보)
                            opp_gain = env._local_score(opp_c, opp_tid, is_first=False) # 상대는 후공
                            
                            if opp_gain > max_opp_gain:
                                max_opp_gain = opp_gain
                            
                            # 원복 (상대)
                            env.board[opp_c] = -1

                # 3. 평가: (내 점수 상승분) - (상대방의 예상 최대 득점 * 가중치)
                # 상대방 견제 가중치 1.5 (수비 중요!)
                # 주의: diff는 누적 점수이므로, 변화량(Delta)을 봐야 정확하지만,
                # 여기선 단순하게 (현재 내 우세 점수 - 상대가 다음 턴에 낼 점수)로 계산
                
                net_score = diff - (max_opp_gain * 1.5)
                
                if net_score > best_net_score:
                    best_net_score = net_score
                    best_action = slot * 64 + c_idx
                
                # -- 원복 (나) --
                env.board[c_idx] = -1
                env.filled -= 1
                
        # 만약 둘 곳이 없으면 0 리턴
        if best_net_score == -9999: return 0, -9999
        return best_action, best_net_score

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
                    
                    # ResNet-5 구조
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
                            self.fc_hand = nn.Sequential(nn.Linear(80, 128), nn.ReLU())
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
                    model = ActorCritic(1104, 320).to(device)
                    
                    model_path = "best_model.pt"
                    if not os.path.exists(model_path): model_path = "final_model.pt"
                    if os.path.exists(model_path):
                        try: model.load_state_dict(torch.load(model_path, map_location=device))
                        except: pass
                    model.eval()
                    env_sim = ConnexionEnv()

                    # 워밍업
                    with torch.no_grad():
                        dummy_obs = torch.zeros(1, 1104).to(device)
                        _ = model.forward_shared(dummy_obs)

            elif cmd == "INIT":
                # INIT A1...A5 B1...B5
                # 내 패와 상대 패를 모두 파싱
                my_tiles_str = parts[1:6]
                opp_tiles_str = parts[6:11]
                
                my_hand = [parse_tile(t) for t in my_tiles_str]
                opp_hand = [parse_tile(t) for t in opp_tiles_str] # ★ 상대 패 저장
                
                board_state = [-1] * 64
                if env_sim:
                    env_sim.reset()
                    env_sim.agent_hand = my_hand[:]
                    env_sim.opp_hand = opp_hand[:] # ★ Env에도 상대 패 주입
                    env_sim.board = board_state[:]

            elif cmd == "TIME":
                obs = np.zeros(1104, dtype=np.float32)
                for i in range(64):
                    if board_state[i] != -1: obs[i*16 + board_state[i]] = 1.0
                for i in range(5):
                    if i < len(my_hand): obs[1024 + i*16 + my_hand[i]] = 1.0
                
                # 행동 결정 (Minimax)
                if env_sim:
                    env_sim.board = board_state[:]
                    env_sim.agent_hand = my_hand[:]
                    env_sim.opp_hand = opp_hand[:] # 최신 상태 동기화
                    env_sim.filled = sum(1 for x in board_state if x != -1)
                    mask = env_sim.get_smart_action_mask()
                    
                    action = get_minimax_action(model, env_sim, obs, mask, device, torch, np)
                else:
                    action = 0

                slot, c_idx = action // 64, action % 64
                
                # 안전장치
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
                # OPP p T1 T2 t
                c_str, t1_str, t2_str = parts[1], parts[2], parts[3]
                
                c_idx = cell_to_idx.get(c_str)
                t1_id = parse_tile(t1_str)
                
                if c_idx is not None:
                    board_state[c_idx] = t1_id
                    
                    # ★ 상대 패 업데이트 로직
                    # 1. 상대가 낸 타일(T1)을 상대 손패에서 제거
                    if t1_id in opp_hand:
                        opp_hand.remove(t1_id)
                    else:
                        # (혹시 동기화가 깨졌을 경우를 대비해 아무거나 제거)
                        if opp_hand: opp_hand.pop(0)
                    
                    # 2. 상대가 뽑은 타일(T2)을 상대 손패에 추가
                    t2_id = parse_tile(t2_str)
                    if t2_id is not None:
                        opp_hand.append(t2_id)

            elif cmd == "FINISH":
                break

        except Exception:
            break

if __name__ == "__main__":
    main()