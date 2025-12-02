# connexion_env.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum  # ✅ 이거 필요
import random
import numpy as np


# ------------------------------------------------
# 기본 타입들 (샘플 코드에서 가져온 부분)
# ------------------------------------------------

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


@dataclass(frozen=True)
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
        - isFirst: True면 문양, False면 색 기준
        """
        return self.symbol == other.symbol if isFirst else self.color == other.color

    @classmethod
    def fromString(cls, s: str) -> "Tile":
        assert len(s) == 2
        return cls(Color(s[0]), Symbol(s[1]))


class Game:
    """
    점수 계산용 클래스 (샘플 코드의 calculateScore 만 따로 뽑은 버전)
    """
    @classmethod
    def calculateScore(cls, board: List[Tuple[Cell, Tile]], isFirst: bool) -> int:
        """
        보드와 선공 여부를 기반으로 점수를 계산함
        - isFirst: True면 문양 기준, False면 색 기준
        """
        if not board:
            return 0

        n = len(board)
        adj = [[False] * n for _ in range(n)]

        # 인접 행렬 구성
        for i, (ci, ti) in enumerate(board):
            for j, (cj, tj) in enumerate(board):
                if i == j or (ci.isAdjacent(cj) and ti.isSame(tj, isFirst)):
                    adj[i][j] = True

        # Warshall로 연결성 계산
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True

        # 각 타일의 점수는 같은 연결 성분에 속한 타일의 수, 총합이 전체 점수
        return sum(sum(row) for row in adj)


# ------------------------------------------------
# 타일 타입/덱 생성 유틸
# ------------------------------------------------

ALL_TILE_TYPES: List[Tile] = [
    Tile(color, symbol) for color in Color for symbol in Symbol
]

def make_player_bag() -> List[Tile]:
    """
    한 플레이어가 갖는 32장: 각 타일 타입 2장씩.
    """
    bag: List[Tile] = []
    for t in ALL_TILE_TYPES:
        bag.append(t)
        bag.append(t)
    random.shuffle(bag)
    return bag


# ------------------------------------------------
# Env 정의
# ------------------------------------------------

class ConnexionEnv:
    """
    Gym 스타일 Connexion 환경 (single-agent self-play 버전)

    - observation: 현재 플레이어 관점의 상태 벡터 (보드 + 손패 + 메타)
    - action: 정수 in [0, num_cells * max_hand)  -> (cell_idx, hand_idx)
    """

    def __init__(self):
        # 모든 사용 가능한 칸 (고정)
        self.all_cells: List[Cell] = Cell.getAllCells()
        self.num_cells: int = len(self.all_cells)  # 56칸
        self.max_hand: int = 5
        self.action_size: int = self.num_cells * self.max_hand

        # 내부 상태
        self.current_player: int = 0  # 0 또는 1
        self.bags: List[List[Tile]] = [[], []]      # 각 플레이어의 남은 덱(주머니)
        self.hands: List[List[Tile]] = [[], []]     # 각 플레이어의 손패(최대 5장)
        self.board: List[Tuple[Cell, Tile]] = []    # (Cell, Tile)
        self.turn_index: int = 0                    # 0 ~ 63
        self.done: bool = False

    # ------------- 필수 API -------------

    def reset(self, seed: Optional[int] = None):
        """
        게임 한 판 시작.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 두 플레이어의 주머니 생성
        self.bags[0] = make_player_bag()
        self.bags[1] = make_player_bag()

        # 손패 5장씩 뽑기
        self.hands[0] = [self.bags[0].pop() for _ in range(5)]
        self.hands[1] = [self.bags[1].pop() for _ in range(5)]

        self.board = []
        self.turn_index = 0
        self.done = False

        # 선/후공 랜덤
        self.current_player = random.randint(0, 1)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        """
        action: int in [0, action_size)
        반환: obs, reward, done, truncated, info
        (gymnasium 스타일로 truncated=False만 사용)
        """
        assert not self.done, "Episode is done. Call reset()."

        # decode action -> (cell, tile)
        cell_idx = action // self.max_hand
        hand_idx = action % self.max_hand

        # 지금 플레이어
        p = self.current_player
        hand = self.hands[p]

        # 유효성 체크
        valid_mask = self.get_valid_action_mask()
        if not valid_mask[action]:
            # invalid action: 큰 패널티 주고 에피소드 끝내기 (예시)
            reward = -1.0
            self.done = True
            obs = self._get_obs()
            return obs, reward, self.done, False, {"invalid_action": True}

        cell = self.all_cells[cell_idx]
        tile = hand[hand_idx]

        # 실제로 보드 상태 업데이트
        self.board.append((cell, tile))
        # 손패에서 사용한 타일 제거
        hand.pop(hand_idx)

        # 주머니에서 새 타일(있으면) 뽑기
        if self.bags[p]:
            hand.append(self.bags[p].pop())

        # 턴 진행
        self.turn_index += 1

        # 에피소드 끝났는지?
        if (
            self.turn_index >= 64
            or (not self.bags[0] and not self.bags[1] and not self.hands[0] and not self.hands[1])
        ):
            self.done = True

        # 리워드 계산
        if self.done:
            reward = self._final_reward()
        else:
            reward = 0.0  # shaping 안 쓰는 기본 버전

        # 플레이어 교대
        self.current_player = 1 - self.current_player

        obs = self._get_obs()
        info = {}
        return obs, reward, self.done, False, info

    # ------------- 보조 함수들 -------------

    def get_valid_action_mask(self) -> np.ndarray:
        """
        현재 플레이어 기준 valid action mask (shape: [action_size], bool 배열)
        - 이미 놓인 칸
        - 손패가 비어있는 인덱스
        등을 제외
        """
        mask = np.zeros(self.action_size, dtype=bool)
        p = self.current_player

        # 이미 놓인 칸 set
        occupied = {c for (c, _) in self.board}

        # 손패 길이
        hand = self.hands[p]
        hand_len = len(hand)

        for cell_idx, cell in enumerate(self.all_cells):
            if cell in occupied:
                continue
            for hand_idx in range(hand_len):
                a = cell_idx * self.max_hand + hand_idx
                mask[a] = True

        return mask

    def _final_reward(self) -> float:
        """
        에피소드 마지막에 점수계산.
        여기서는 player0=선공, player1=후공이라고 가정하고
        player0 관점에서 (my_score - opp_score)의 승/무/패를 리턴.
        """
        first_score = Game.calculateScore(self.board, True)   # 문양 기준
        second_score = Game.calculateScore(self.board, False) # 색 기준

        my_score = first_score
        opp_score = second_score

        if my_score > opp_score:
            return 1.0
        elif my_score < opp_score:
            return -1.0
        else:
            return 0.0

    # ------------- 상태 인코딩 -------------

    def _get_obs(self) -> np.ndarray:
        """
        현재 플레이어(p)의 관점에서 observation 벡터를 만든다.
        (스켈레톤: owner는 일단 '내 것'으로만 표시)
        """
        p = self.current_player

        # 보드 인코딩: 각 cell마다 [empty, my, opp, color(4), symbol(4)] = 3+4+4=11차원
        board_feat = np.zeros((self.num_cells, 11), dtype=np.float32)

        # cell -> index map
        cell_to_idx: Dict[Cell, int] = {c: i for i, c in enumerate(self.all_cells)}

        for (cell, tile) in self.board:
            idx = cell_to_idx[cell]
            # owner: 현재는 소유자 정보가 없어서 일단 my로만 세팅
            board_feat[idx, 1] = 1.0  # [empty, my, opp] 중 my=1

            # color one-hot (R,G,B,Y)
            color_idx = {"R": 0, "G": 1, "B": 2, "Y": 3}[tile.color.value]
            board_feat[idx, 3 + color_idx] = 1.0

            # symbol one-hot (1,2,3,4)
            sym_idx = {"1": 0, "2": 1, "3": 2, "4": 3}[tile.symbol.value]
            board_feat[idx, 7 + sym_idx] = 1.0

        board_flat = board_feat.flatten()

        # 손패 인코딩: max_hand * 16 (타입 one-hot)
        hand_feat = np.zeros((self.max_hand, 16), dtype=np.float32)
        for i, tile in enumerate(self.hands[p]):
            if i >= self.max_hand:
                break
            t_idx = ALL_TILE_TYPES.index(tile)
            hand_feat[i, t_idx] = 1.0
        hand_flat = hand_feat.flatten()

        # 메타: [current_player(0/1), turn_index/64]
        meta = np.array(
            [
                float(p),
                float(self.turn_index) / 64.0,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([board_flat, hand_flat, meta], axis=0)
        return obs
