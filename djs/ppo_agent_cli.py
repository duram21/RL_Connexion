# ppo_agent_cli.py
import sys
import torch
import numpy as np

from connexion_env import ConnexionEnv, Tile, Cell, ALL_TILE_TYPES
from train_ppo import PolicyValueNet, masked_categorical


def load_model(model_path: str = "ppo_connexion.pt"):
    """
    PPO 모델 로드 + 더미 env로 obs_dim, action_dim 구하기
    """
    dummy_env = ConnexionEnv()
    obs, _ = dummy_env.reset(seed=0)
    obs_dim = obs.shape[0]
    action_dim = dummy_env.action_size

    model = PolicyValueNet(obs_dim, action_dim)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, dummy_env


def tile_from_string(s: str) -> Tile:
    # "R1" -> Tile(Color.R, Symbol._1) (connexion_env 쪽 구현과 동일)
    return Tile.fromString(s)


def cell_from_string(s: str) -> Cell:
    return Cell.fromString(s)


def main():
    model, env = load_model()

    # env 상태를 testing tool 기준으로 재설정해서 사용할 거라 reset()은 쓰지 않는다.
    # (all_cells, num_cells, max_hand, action_size는 __init__에서 이미 셋업됨)
    env.board = []
    env.hands = [[], []]  # [my_hand, opp_hand]
    env.turn_index = 0
    env.current_player = 0  # 우리는 항상 "player 0 관점"으로 본다.

    is_first = None          # FIRST / SECOND 여부
    my_tiles: list[Tile] = []   # 내 손패 (env.hands[0]와 동기화)
    opp_tiles: list[Tile] = []  # 상대 손패 (env.hands[1]와 동기화 – 사실 obs에는 안 쓰임)
    last_cell: Cell | None = None
    last_tile: Tile | None = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "READY":
            # READY FIRST / READY SECOND
            role = parts[1]
            is_first = (role == "FIRST")
            print("OK")
            sys.stdout.flush()

        elif cmd == "INIT":
            # INIT A1 A2 A3 A4 A5 B1 B2 B3 B4 B5
            #
            # testing-tool에서:
            # FIRST 에게: INIT first5 second5
            # SECOND 에게: INIT second5 first5
            #
            # "내 입장"에서는 항상 앞 5개가 내 타일, 뒤 5개가 상대 타일.
            my_tiles = [tile_from_string(s) for s in parts[1:6]]
            opp_tiles = [tile_from_string(s) for s in parts[6:11]]

            env.board = []
            env.hands[0] = my_tiles.copy()
            env.hands[1] = opp_tiles.copy()
            env.turn_index = 0
            env.current_player = 0  # 우리는 항상 player 0로 본다.

            last_cell = None
            last_tile = None

        elif cmd == "TIME":
            # TIME my_time opp_time
            # 여기서 액션(=PUT) 선택
            my_time, opp_time = int(parts[1]), int(parts[2])

            # 현재 관측 만들기
            env.current_player = 0  # 항상 나 기준
            obs = env._get_obs()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                logits, value = model(obs_t)
                valid_mask_np = env.get_valid_action_mask().astype(np.bool_)
                valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0)
                dist = masked_categorical(logits, valid_mask)

                # 학습 때는 sample을 썼지만, 대전에서는 argmax를 써도 되고 sample을 써도 된다.
                # 여기선 sample 유지
                action = dist.sample()
                # action = dist.probs.argmax(dim=-1)  # 그리디로 하고 싶으면 이걸 사용

            a = int(action.item())
            cell_idx = a // env.max_hand
            hand_idx = a % env.max_hand

            # env.hands[0]에서 실제 타일 가져오기
            hand = env.hands[0]
            # get_valid_action_mask가 invalid를 막아줬으니 index는 유효하다고 가정
            tile = hand[hand_idx]
            cell = env.all_cells[cell_idx]

            last_cell = cell
            last_tile = tile

            print(f"PUT {cell} {tile}")
            sys.stdout.flush()

        elif cmd == "GET":
            # GET T (T == X0 이면 안 뽑음)
            T_str = parts[1]

            # 방금 둔 타일을 내 손에서 제거 + 보드에 반영
            if last_tile is not None:
                try:
                    env.hands[0].remove(last_tile)
                except ValueError:
                    # 이론상 나오면 안 되지만 방어 코드
                    pass
                if last_cell is not None:
                    env.board.append((last_cell, last_tile))
                    env.turn_index += 1  # 한 수 진행

            # 새로 뽑은 타일이 있으면 손패에 추가
            if T_str != "X0":
                drawn_tile = tile_from_string(T_str)
                env.hands[0].append(drawn_tile)

            last_cell = None
            last_tile = None

        elif cmd == "OPP":
            # OPP p T1 T2 t
            # p: 위치(cell), T1: 상대가 둔 타일, T2: 상대가 뽑은 타일(X0 가능), t: 사용 시간(ms)
            cell_str, tile_str, draw_str, used_time_str = parts[1:5]
            cell = cell_from_string(cell_str)
            tile = tile_from_string(tile_str)
            env.board.append((cell, tile))
            env.turn_index += 1  # 한 수 진행

            # 상대 손패도 유지하고 싶으면 업데이트
            try:
                env.hands[1].remove(tile)
            except ValueError:
                pass
            if draw_str != "X0":
                env.hands[1].append(tile_from_string(draw_str))

        elif cmd == "FINISH":
            break

        else:
            # 알 수 없는 명령은 무시 (테스트 툴은 딴 거 안 줌)
            pass


if __name__ == "__main__":
    main()
