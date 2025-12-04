#!/usr/bin/env python3
import sys
import os
import random

# lazy-load 용
torch = None
np = None
ConnexionEnv = None
ActorCritic = None


def main():
    global torch, np, ConnexionEnv, ActorCritic

    # ------------------------------------------------------------------
    # 1. 기본 설정 및 공용 유틸
    # ------------------------------------------------------------------
    colors = ['R', 'G', 'B', 'Y']
    symbols = ['1', '2', '3', '4']

    # 유효한 셀 리스트 (순서는 ConnexionEnv의 ALL_CELLS와 동일하다고 가정)
    valid_cells = []
    forbidden = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}
    for c in ['a', 'b', 'c', 'd', 'e', 'f']:
        for r in ['1', '2', '3', '4', '5', '6']:
            for s in ['-', '+']:
                if (c + r + s) not in forbidden:
                    valid_cells.append(c + r + s)

    cell_to_idx = {s: i for i, s in enumerate(valid_cells)}

    def parse_tile(s: str):
        """문자열 'R1' → tid (0~15), 'X0'이면 None"""
        if s == "X0":
            return None
        try:
            return colors.index(s[0]) * 4 + symbols.index(s[1])
        except Exception:
            return None

    def tile_to_str(tid: int) -> str:
        return colors[tid // 4] + symbols[tid % 4]

    # 상태 변수들 (프로토콜에 따라 갱신)
    my_hand = []      # 내 손패 (list[int])
    opp_hand = []     # 상대 손패 (list[int]) – OPP 명령으로 추적
    board_state = [-1] * 64  # 64칸 보드

    # 시뮬레이션용 env
    env_sim = None
    model = None
    device = None

    # PPO/휴리스틱 하이브리드의 가중치 관련 상수
    OPP_WEIGHT = 1.5          # 상대가 다음 턴에 얻을 수 있는 로컬 득점 가중치
    PPO_TOLERANCE = 3.0       # PPO 수가 최선 휴리스틱보다 이 정도 이하로만 나쁘면 PPO 존중

    # ------------------------------------------------------------------
    # 2. PPO 네트워크 정의 (학습할 때 썼던 것과 동일해야 함)
    # ------------------------------------------------------------------
    def build_model():
        import torch as t
        import torch.nn as nn
        import torch.nn.functional as F_local

        class ResBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                return F_local.relu(
                    x + self.bn2(self.conv2(F_local.relu(self.bn1(self.conv1(x)))))
                )

        class AC(nn.Module):
            def __init__(self, obs_dim: int, act_dim: int):
                super().__init__()
                # 보드 인코더
                self.conv_in = nn.Conv2d(16, 128, 3, 1, 1, bias=False)
                self.bn_in = nn.BatchNorm2d(128)
                self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])
                # 내 손패 (5 × 16 = 80)
                self.fc_hand = nn.Sequential(nn.Linear(80, 128), nn.ReLU())
                # Actor head
                self.actor_conv = nn.Conv2d(128, 32, 1)
                self.actor_bn = nn.BatchNorm2d(32)
                self.actor_fc = nn.Linear(32 * 8 * 8 + 128, act_dim)
                # Critic head (여기서는 사용 안하지만 구조 상 존재)
                self.critic_conv = nn.Conv2d(128, 8, 1)
                self.critic_bn = nn.BatchNorm2d(8)
                self.critic_fc1 = nn.Linear(8 * 8 * 8 + 128, 256)
                self.critic_fc2 = nn.Linear(256, 1)

            def forward_shared(self, obs):
                """
                obs: (B, 1104) = (board 1024 + hand 80)
                board_flat: (B, 64*16)
                """
                B = obs.size(0)
                board_flat = obs[:, :1024]
                hand_flat = obs[:, 1024:]

                # (B, 64, 16) → (B, 16, 64) → (B, 16, 8, 8)
                board = board_flat.view(B, 64, 16).permute(0, 2, 1).contiguous()
                board = board.view(B, 16, 8, 8)

                x = self.res_blocks(t.relu(self.bn_in(self.conv_in(board))))
                h = self.fc_hand(hand_flat)
                return x, h

        return AC

    # ------------------------------------------------------------------
    # 3. 관측 벡터 구성 (board_state + my_hand → 1104차원)
    # ------------------------------------------------------------------
    def build_obs(board, hand):
        """
        board: 리스트 길이 64, 값은 tid 또는 -1
        hand:  내 손패 tid 리스트
        """
        obs = np.zeros(1104, dtype=np.float32)

        # 1) 보드 (64 × 16 = 1024)
        for i in range(64):
            tid = board[i]
            if tid != -1:
                obs[i * 16 + tid] = 1.0

        # 2) 손패 (5 × 16 = 80) – 보이는 순서대로 슬롯
        for i in range(min(5, len(hand))):
            tid = hand[i]
            obs[1024 + i * 16 + tid] = 1.0

        return obs

    # ------------------------------------------------------------------
    # 4. 하이브리드 액션 선택 (PPO + 1-ply 휴리스틱)
    # ------------------------------------------------------------------
    def eval_action_heuristic(env, action_idx: int) -> float:
        """
        env 상태에서 선공(우리)이 action_idx에 뒀을 때:
          - 즉시 diff = first_score - second_score
          - 상대의 “최대 로컬 득점” (색 기준)을 대략적으로 빼서 net_score 계산
        """
        slot = action_idx // 64
        c_idx = action_idx % 64

        # 완전 잘못된 수이면 큰 음수
        if (
            slot < 0
            or slot >= len(env.agent_hand)
            or c_idx < 0
            or c_idx >= env.num_cells
            or env.board[c_idx] != -1
        ):
            return -1e9

        # 상태 백업
        bak_board = list(env.board)
        bak_agent_hand = list(env.agent_hand)
        bak_opp_hand = list(env.opp_hand)
        bak_filled = env.filled

        # 우리 수 가상 적용
        tid = env.agent_hand.pop(slot)
        env.board[c_idx] = tid
        env.filled += 1

        # 현재 전체 점수 계산 (선공/후공 차)
        fs, ss, diff = env._compute_scores()

        # 상대의 최대 로컬 득점(색 기준) 계산
        max_opp_gain = 0
        if env.opp_hand:
            empty_cells = [i for i, v in enumerate(env.board) if v == -1]
            for opp_slot, opp_tid in enumerate(env.opp_hand):
                for oc in empty_cells:
                    # 상대가 (opp_slot, oc)에 둘 때의 색 기준 로컬 점수
                    gain = env._local_score(oc, opp_tid, is_first=False)
                    if gain > max_opp_gain:
                        max_opp_gain = gain

        net_score = diff - (OPP_WEIGHT * max_opp_gain)

        # 상태 복구
        env.board = bak_board
        env.agent_hand = bak_agent_hand
        env.opp_hand = bak_opp_hand
        env.filled = bak_filled

        return float(net_score)

    def choose_hybrid_action(model, env, obs_np, mask_np):
        """
        1) PPO로부터 logits/argmax 얻기
        2) heuristic으로 best_action 찾기
        3) 두 수를 비교해 최종 action 결정
        """
        # 유효한 액션 인덱스
        valid_indices = [i for i, v in enumerate(mask_np) if v]
        if not valid_indices:
            return 0  # 아무 것도 못 두면 그냥 0 리턴 (어차피 게임 끝 상황일 가능성 높음)

        # ----- 1) PPO: 가장 확률 높은 수 -----
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_np).bool().unsqueeze(0).to(device)

        with torch.no_grad():
            x, h = model.forward_shared(obs_t)
            B = x.size(0)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(B, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)  # (1, 320)
            logits = logits.masked_fill(~mask_t, -1e9)

            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # PPO가 추천하는 최상 수
        ppo_action = int(np.argmax(probs))
        if not mask_np[ppo_action]:
            # 혹시나 마스크 밖이면, valid 중에서 확률 최대인 걸로
            valid_probs = [(a, probs[a]) for a in valid_indices]
            ppo_action = max(valid_probs, key=lambda x: x[1])[0]

        # ----- 2) 휴리스틱: 모든 valid 수에 대해 net_score 계산 -----
        best_heur_action = None
        best_heur_score = -1e9

        for a in valid_indices:
            score = eval_action_heuristic(env, a)
            if score > best_heur_score:
                best_heur_score = score
                best_heur_action = a

        # ----- 3) PPO 수의 net_score 계산 -----
        ppo_score = eval_action_heuristic(env, ppo_action)

        # ----- 4) 최종 결정 -----
        # PPO 수가 휴리스틱 최선수에 비해 PPO_TOLERANCE 만큼만 덜 좋아도 허용
        if ppo_score >= best_heur_score - PPO_TOLERANCE:
            return ppo_action
        else:
            return best_heur_action if best_heur_action is not None else ppo_action

    # ------------------------------------------------------------------
    # 5. 메인 프로토콜 처리 루프
    # ------------------------------------------------------------------
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue

        cmd = parts[0]

        # ---------------- READY ----------------
        if cmd == "READY":
            print("OK")
            sys.stdout.flush()

            if torch is None:
                import torch as t
                import numpy as n
                from connexion_env import ConnexionEnv as CE  # 네 프로젝트에 맞게 모듈명 확인

                torch = t
                np = n
                ConnexionEnv = CE
                torch.set_num_threads(1)

                # 모델 빌드
                AC = build_model()
                device_local = torch.device("cpu")

                m = AC(1104, 320).to(device_local)

                # 파라미터 로드
                model_path = "best_model.pt"
                if not os.path.exists(model_path):
                    model_path = "final_model.pt"
                if os.path.exists(model_path):
                    try:
                        state = torch.load(model_path, map_location=device_local)
                        m.load_state_dict(state)
                    except Exception:
                        pass
                m.eval()

                # env_sim 생성
                env = ConnexionEnv()
                # 워밍업 (BN 안정)
                with torch.no_grad():
                    dummy = torch.zeros(1, 1104).to(device_local)
                    # forward_shared만 써도 BN 통과
                    _ = m.forward_shared(dummy)

                model = m
                device = device_local
                env_sim = env

        # ---------------- INIT ----------------
        elif cmd == "INIT":
            # INIT A1...A5 B1...B5
            my_tiles_str = parts[1:6]
            opp_tiles_str = parts[6:11]

            my_hand = [parse_tile(t) for t in my_tiles_str if parse_tile(t) is not None]
            opp_hand = [parse_tile(t) for t in opp_tiles_str if parse_tile(t) is not None]

            board_state = [-1] * 64

            if env_sim is not None:
                env_sim.reset()
                env_sim.board = list(board_state)
                env_sim.agent_hand = my_hand[:]
                env_sim.opp_hand = opp_hand[:]
                env_sim.filled = 0
                env_sim.done = False
                env_sim.prev_diff = 0.0

        # ---------------- TIME (우리 차례) ----------------
        elif cmd == "TIME":
            if env_sim is None or model is None:
                # 안전장치
                # 가능한 첫 칸에 첫 타일 둔다
                c_idx = next((i for i, v in enumerate(board_state) if v == -1), 0)
                if not my_hand:
                    my_hand.append(0)
                print(f"PUT {valid_cells[c_idx]} {tile_to_str(my_hand[0])}")
                sys.stdout.flush()
                continue

            # env_sim에 현재 상태 싱크
            env_sim.board = list(board_state)
            env_sim.agent_hand = my_hand[:]
            env_sim.opp_hand = opp_hand[:]
            env_sim.filled = sum(1 for x in board_state if x != -1)
            env_sim.done = False  # 진행 중

            # 관측 벡터 구성 (board + my_hand)
            obs = build_obs(board_state, my_hand)

            # 액션 마스크 (Smart Mask 기반)
            mask = env_sim.get_smart_action_mask()
            # 혹시 전부 False일 경우, valid mask로 fallback
            if not mask.any():
                mask = env_sim.get_valid_action_mask()

            action = choose_hybrid_action(model, env_sim, obs, mask)

            slot = action // 64
            c_idx = action % 64

            # 안전장치: 잘못된 수면 그냥 첫 가능한 곳으로
            if c_idx < 0 or c_idx >= 64 or board_state[c_idx] != -1:
                c_idx = next((i for i, v in enumerate(board_state) if v == -1), 0)
            if slot < 0 or slot >= len(my_hand):
                slot = 0

            tile_id = my_hand[slot]

            print(f"PUT {valid_cells[c_idx]} {tile_to_str(tile_id)}")
            sys.stdout.flush()

            # 내부 상태 업데이트
            board_state[c_idx] = tile_id
            my_hand.pop(slot)

        # ---------------- GET (내가 카드 한 장 뽑음) ----------------
        elif cmd == "GET":
            new_t = parse_tile(parts[1])
            if new_t is not None:
                my_hand.append(new_t)

        # ---------------- OPP (상대가 둔 수 / 뽑은 수) ----------------
        elif cmd == "OPP":
            # OPP CELL T1 T2 t
            if len(parts) >= 4:
                c_str, t1_str, t2_str = parts[1], parts[2], parts[3]
                c_idx = cell_to_idx.get(c_str)
                t1_id = parse_tile(t1_str)
                t2_id = parse_tile(t2_str)

                if c_idx is not None and t1_id is not None:
                    board_state[c_idx] = t1_id

                    # 상대 손패에서 사용한 타일 제거
                    if t1_id in opp_hand:
                        opp_hand.remove(t1_id)
                    else:
                        if opp_hand:
                            opp_hand.pop(0)

                    # 새로 뽑은 타일 추가
                    if t2_id is not None:
                        opp_hand.append(t2_id)

        # ---------------- FINISH ----------------
        elif cmd == "FINISH":
            break

        # 예외는 그냥 종료
        else:
            continue


if __name__ == "__main__":
    main()
