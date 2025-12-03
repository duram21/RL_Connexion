# agent_rl.py
# 학습된 PPO 정책(ppo_connexion.pt)을 사용해서
# NYPC Connexion 심사 서버와 대결하는 실행용 에이전트 예시 코드입니다.

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 게임 기본 타입들 (샘플 코드 그대로)
# ============================================================

class Column(Enum):
    """게임 보드의 열 번호"""
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"


class Row(Enum):
    """게임 보드의 행 번호"""
    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"
    _5 = "5"
    _6 = "6"


class Sign(Enum):
    """게임 보드의 부호"""
    MINUS = "-"
    PLUS = "+"


class Color(Enum):
    """게임 타일의 색"""
    R = "R"
    G = "G"
    B = "B"
    Y = "Y"


class Symbol(Enum):
    """게임 타일의 문양"""
    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"


@dataclass
class Cell:
    """게임 보드의 칸"""
    col: Column  # 게임 보드의 열
    row: Row     # 게임 보드의 행
    sign: Sign   # 게임 보드의 부호

    def __str__(self):
        return f"{self.col.value}{self.row.value}{self.sign.value}"

    def isValid(self) -> bool:
        """
        금지된 칸(a1-, a4-, c3+, c6+, d1-, d4-, f3+, f6+)이 아니면 True
        """
        return self not in [
            Cell(Column.A, Row._1, Sign.MINUS),
            Cell(Column.A, Row._4, Sign.MINUS),
            Cell(Column.C, Row._3, Sign.PLUS),
            Cell(Column.C, Row._6, Sign.PLUS),
            Cell(Column.D, Row._1, Sign.MINUS),
            Cell(Column.D, Row._4, Sign.MINUS),
            Cell(Column.F, Row._3, Sign.PLUS),
            Cell(Column.F, Row._6, Sign.PLUS),
        ]

    def isAdjacent(self, other: "Cell") -> bool:
        """
        주어진 두 칸이 인접한 칸인지 여부를 반환
        """
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)

        return (
            self.sign == Sign.MINUS
            and other.sign == Sign.PLUS
            and (dr == dc == 0 or (dr == 0 and dc == -1) or (dr == -1 and dc == 0))
        ) or (
            self.sign == Sign.PLUS
            and other.sign == Sign.MINUS
            and (dr == dc == 0 or (dr == 0 and dc == 1) or (dr == 1 and dc == 0))
        )

    @classmethod
    def getAllCells(cls) -> List["Cell"]:
        """
        배치 가능한 모든 칸을 반환
        """
        ret = []
        for col in Column:
            for row in Row:
                for sign in Sign:
                    cell = cls(col, row, sign)
                    if cell.isValid():
                        ret.append(cell)
        return ret

    @classmethod
    def fromString(cls, s: str) -> "Cell":
        assert len(s) == 3
        cell = cls(Column(s[0]), Row(s[1]), Sign(s[2]))
        assert cell.isValid()
        return cell


@dataclass
class Tile:
    """
    게임 타일
    """
    color: Color   # 타일의 색
    symbol: Symbol # 타일의 문양

    def __str__(self):
        return f"{self.color.value}{self.symbol.value}"

    def isSame(self, other: "Tile", isFirst: bool) -> bool:
        """
        점수 계산시 두 타일을 같은 타일로 볼 것인가를 반환

        - isFirst: True -> 문양 기준, False -> 색 기준
        """
        return self.symbol == other.symbol if isFirst else self.color == other.color

    @classmethod
    def fromString(cls, s: str) -> "Tile":
        assert len(s) == 2
        return cls(Color(s[0]), Symbol(s[1]))


class Game:
    """
    게임 상태를 관리하는 클래스

    - myTiles (List[Tile]): 내 타일 목록
    - oppTiles (List[Tile]): 상대 타일 목록
    - isFirst (bool): 선후공 (True: 선공, False: 후공)
    - board (List[Tuple[Cell, Tile]]): 현재 보드에 배치된 칸과 타일 목록
    """

    def __init__(self, myTiles: List[Tile], oppTiles: List[Tile], isFirst: bool):
        self.myTiles = myTiles
        self.oppTiles = oppTiles
        self.isFirst = isFirst
        self.board: List[Tuple[Cell, Tile]] = []

    # ============================== (강화학습 버전) ==============================

    def calculateMove(self, _myTime: int, _oppTime: int) -> Tuple[Cell, Tile]:
        """
        현재 상태를 기반으로 놓아야 할 칸과 타일을 계산함.
        (원래 샘플 코드는 그리디였지만, 여기서는 PPO policy를 사용)
        """
        # 1) Game 상태 -> 관측 벡터로 인코딩
        obs = encode_state_from_game(self)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)  # [1, obs_dim]

        # 2) PPO 정책으로 액션 확률 얻기
        with torch.no_grad():
            prob = policy(obs_t)[0]   # [act_dim]

        prob_np = prob.cpu().numpy()

        # 3) valid action mask 구하기
        valid_mask = get_valid_action_mask_from_game(self)
        if not valid_mask.any():
            # 둘 수 있는 곳이 없는 이상 상황: 그냥 첫 번째 가능한 칸/타일에 둔다 (fallback)
            for t in self.myTiles:
                for c in ALL_CELLS:
                    if c not in [cc for cc, _ in self.board]:
                        return c, t

        # 4) invalid action 확률은 -inf로 막고 argmax
        prob_np[~valid_mask] = -1e9
        best_action = int(prob_np.argmax())

        # 5) action index -> (hand_idx, cell_idx) 디코딩
        hand_idx = best_action // NUM_CELLS
        cell_idx = best_action % NUM_CELLS

        # hand_idx가 현재 손패 범위 안인지 한 번 더 확인
        hand_idx = min(hand_idx, len(self.myTiles) - 1)

        cell = ALL_CELLS[cell_idx]
        tile = self.myTiles[hand_idx]
        return cell, tile

    # ============================ [필수 구현 끝] ============================

    def updateAction(
        self,
        myAction: bool,
        action: Tuple[Cell, Tile],
        get: Optional[Tile],
        _usedTime: Optional[int],
    ):
        """
        자신 혹은 상대의 행동을 기반으로 상태를 업데이트 함
        """
        self.board.append(action)

        _, tile = action
        if myAction:
            self.myTiles.remove(tile)
            if get is not None:
                self.myTiles.append(get)
        else:
            self.oppTiles.remove(tile)
            if get is not None:
                self.oppTiles.append(get)

    @classmethod
    def calculateScore(cls, board: List[Tuple[Cell, Tile]], isFirst: bool) -> int:
        """
        보드와 선후공를 기반으로 점수를 계산함
        """
        if not board:
            return 0

        adj = [[False] * len(board) for _ in range(len(board))]
        for i, (ci, ti) in enumerate(board):
            for j, (cj, tj) in enumerate(board):
                if i == j or (ci.isAdjacent(cj) and ti.isSame(tj, isFirst)):
                    adj[i][j] = True

        for k in range(len(board)):
            for i in range(len(board)):
                for j in range(len(board)):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True

        return sum(sum(row) for row in adj)


# ============================================================
# 2. PPO 정책 네트워크 정의 (inference용)
# ============================================================

# --- 전역 상수/유틸 (env와 맞춰줘야 함) ---

ALL_CELLS: List[Cell] = Cell.getAllCells()
NUM_CELLS: int = len(ALL_CELLS)         # 일반적으로 64

ALL_TILE_TYPES: List[Tile] = [
    Tile(color, symbol) for color in Color for symbol in Symbol
]
NUM_TILE_TYPES: int = len(ALL_TILE_TYPES)  # 16
MAX_HAND: int = 5

# 관측 인코딩:
# - board: cell당 [occupied, color(4), symbol(4)] = 9 → NUM_CELLS * 9
# - hand: MAX_HAND * 16
# - isFirst: 1
OBS_DIM: int = NUM_CELLS * 9 + MAX_HAND * NUM_TILE_TYPES + 1
ACT_DIM: int = NUM_CELLS * MAX_HAND       # hand_idx * cell_idx


class PPOPolicy(nn.Module):
    """
    학습 때 사용한 것과 동일한 구조여야 함.
    여기서는 간단히 2층 MLP + softmax policy만.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_pi = nn.Linear(256, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_pi(x)
        prob = F.softmax(logits, dim=-1)
        return prob


# --- 학습된 모델 로드 ---

MODEL_PATH = "ppo_connexion.pt"  # 학습 코드에서 저장한 파일 이름과 맞추기

policy = PPOPolicy(OBS_DIM, ACT_DIM)
try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"# Loaded PPO model from {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    print(f"# WARNING: could not load model: {e}", file=sys.stderr)
    print("# The agent will behave randomly-ish.", file=sys.stderr)
    # 로드 실패하면 그냥 랜덤처럼 동작 (policy는 랜덤 초기화 상태)


# ============================================================
# 3. Game 상태 -> 관측 & valid action mask
# ============================================================

def encode_state_from_game(game: Game) -> np.ndarray:
    """
    Game 객체 상태를 PPO 학습에서 사용한 observation 벡터로 인코딩.
    - board: 각 cell에 어떤 타일(색/문양)이 올라가 있는지만 보고, 주인은 구분 안 함.
    - hand: 내 손패 5장을 one-hot(16)로 인코딩.
    - isFirst: 선공 여부 (float 0/1)
    """
    # 보드 인코딩
    board_feat = np.zeros((NUM_CELLS, 9), dtype=np.float32)
    # cell string -> tile 매핑
    cell_map: Dict[str, Tile] = {str(c): t for (c, t) in game.board}

    color_index = {"R": 0, "G": 1, "B": 2, "Y": 3}
    symbol_index = {"1": 0, "2": 1, "3": 2, "4": 3}

    for idx, cell in enumerate(ALL_CELLS):
        key = str(cell)
        if key not in cell_map:
            continue
        tile = cell_map[key]
        # occupied
        board_feat[idx, 0] = 1.0
        # color one-hot
        c_idx = color_index[tile.color.value]
        board_feat[idx, 1 + c_idx] = 1.0
        # symbol one-hot
        s_idx = symbol_index[tile.symbol.value]
        board_feat[idx, 5 + s_idx] = 1.0

    board_flat = board_feat.flatten()  # [NUM_CELLS * 9]

    # 손패 인코딩
    hand_feat = np.zeros((MAX_HAND, NUM_TILE_TYPES), dtype=np.float32)
    for i, tile in enumerate(game.myTiles):
        if i >= MAX_HAND:
            break
        # 같은 색/문양이면 같은 타입으로 본다
        t_idx = ALL_TILE_TYPES.index(tile)
        hand_feat[i, t_idx] = 1.0

    hand_flat = hand_feat.flatten()  # [MAX_HAND * 16]

    # isFirst 플래그
    is_first = 1.0 if game.isFirst else 0.0
    meta = np.array([is_first], dtype=np.float32)

    obs = np.concatenate([board_flat, hand_flat, meta], axis=0)
    assert obs.shape[0] == OBS_DIM
    return obs


def get_valid_action_mask_from_game(game: Game) -> np.ndarray:
    """
    현재 Game 상태 기준으로 valid한 action을 표시하는 mask.
    - action = hand_idx * NUM_CELLS + cell_idx
    - hand_idx < len(myTiles)
    - cell이 아직 비어있는 칸이어야 함
    """
    mask = np.zeros(ACT_DIM, dtype=bool)
    occupied = {str(c) for c, _ in game.board}

    for h_idx in range(len(game.myTiles)):
        for c_idx, cell in enumerate(ALL_CELLS):
            if str(cell) in occupied:
                continue
            a = h_idx * NUM_CELLS + c_idx
            if a < ACT_DIM:
                mask[a] = True

    return mask


# ============================================================
# 4. I/O 루프 (샘플 코드와 동일, calculateMove만 RL로 변경)
# ============================================================

def main():
    game: Optional[Game] = None
    isFirst: Optional[bool] = None
    lastMove: Optional[Tuple[Cell, Tile]] = None

    while True:
        try:
            line = input().strip()
            if not line:
                continue

            command, *args = line.split()

            if command == "READY":
                # 게임 시작
                isFirst = args[0] == "FIRST"
                print("OK")
                continue

            if command == "INIT":
                # 준비 단계 시작
                myTiles = [t for t in map(Tile.fromString, args[:5]) if t is not None]
                oppTiles = [t for t in map(Tile.fromString, args[5:]) if t is not None]
                assert isFirst is not None
                game = Game(myTiles, oppTiles, isFirst)
                continue

            if command == "TIME":
                # 배치 단계 시작
                myTime, oppTime = int(args[0]), int(args[1])
                assert game is not None
                lastMove = game.calculateMove(myTime, oppTime)
                cell, tile = lastMove
                print(f"PUT {cell} {tile}", flush=True)
                continue

            if command == "GET":
                # 배치 단계가 끝날 때 뽑아온 타일을 이용해서 상태 업데이트
                get = None if args[0] == "X0" else Tile.fromString(args[0])
                assert game is not None and lastMove is not None
                game.updateAction(True, lastMove, get, None)
                continue

            if command == "OPP":
                # 상대의 행동을 기반으로 상태를 업데이트 함
                cell, tile, get, oppTime = (
                    Cell.fromString(args[0]),
                    Tile.fromString(args[1]),
                    None if args[2] == "X0" else Tile.fromString(args[2]),
                    int(args[3]),
                )
                assert game is not None and tile is not None
                game.updateAction(False, (cell, tile), get, oppTime)
                continue

            if command == "FINISH":
                # 게임 종료
                break

            # 알 수 없는 명령어 처리
            print(f"Invalid command: {command}", file=sys.stderr)
            sys.exit(1)

        except EOFError:
            break


if __name__ == "__main__":
    main()
