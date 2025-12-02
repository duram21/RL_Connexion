# connexion_env.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
import numpy as np

# 여기에는 네가 준 Cell, Tile, Color, Symbol, Sign, Game.calculateScore 를
# 그대로 붙여넣거나, from 다른_파일 import * 로 가져와 쓰면 된다.
# (아래에서는 이미 정의되어 있다고 가정하고 사용할게.)

# ------------------------------------------------
# 타일 타입/덱 생성 유틸
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


@dataclass
class Cell:
    """게임 보드의 칸"""

    col: Column  # 게임 보드의 열
    row: Row  # 게임 보드의 행
    sign: Sign  # 게임 보드의 부호

    def __str__(self):
        return f"{self.col.value}{self.row.value}{self.sign.value}"

    def isValid(self) -> bool:
        """
        해당 칸이 금지된 칸인 a1-, a4-, c3+, c6+, d1-, d4-, f3+, f6+ 중 하나가 아니라면 True를 반환
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

        # 두 칸의 행, 열 번호 차이를 계산
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)

        return (
            self.sign == Sign.MINUS
            and other.sign == Sign.PLUS
            and (dr == dc == 0 or dr == 0 and dc == -1 or dr == -1 and dc == 0)
        ) or (
            self.sign == Sign.PLUS
            and other.sign == Sign.MINUS
            and (dr == dc == 0 or dr == 0 and dc == 1 or dr == 1 and dc == 0)
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

    color: Color  # 타일의 색
    symbol: Symbol  # 타일의 문양

    def __str__(self):
        return f"{self.color.value}{self.symbol.value}"

    def isSame(self, other: "Tile", isFirst: bool) -> bool:
        """
        점수 계산시 두 타일을 같은 타일로 볼 것인가를 반환

        인자 목록
        - other (Tile): 비교할 타일
        - isFirst (bool): 선후공 (True: 선공 / 문양을 사용 , False: 후공 / 색을 사용)
        """
        return self.symbol == other.symbol if isFirst else self.color == other.color

    @classmethod
    def fromString(cls, s: str) -> "Tile":
        assert len(s) == 2
        return cls(Color(s[0]), Symbol(s[1]))



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
            # invalid action: 큰 패널티 주고 에피소드 끝내거나,
            # 아니면 그냥 no-op + 페널티를 줄 수 있다.
            # 여기서는 간단히 에피소드 종료 + 패배로 처리 (예시).
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
        if self.turn_index >= 64 or (not self.bags[0] and not self.bags[1] and
                                     not self.hands[0] and not self.hands[1]):
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
        - 빈 칸이 아닌 곳
        - 손패에 존재하지 않는 인덱스
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
        - player0, player1의 점수 계산 후,
        현재 step에서 수를 둔 플레이어 기준 reward를 줄 수도 있고,
        항상 player0 관점 reward를 반환하고, self-play에서 관점을 바꿔 쓸 수도 있다.
        여기서는 "플레이어 0 관점" reward를 반환하는 예시로 작성.
        """
        # board에는 모든 수가 들어있으므로,
        # 선공/후공을 누가 했는지에 따라 first/second 점수를 나눠야 함.
        # 여기서는 간단히: player0을 "선공", player1을 "후공"으로 놓고 예시를 든다.
        # 실제 프로젝트에선 current_player/턴 정보를 기록해서 더 정확히 처리해도 됨.

        first_score = Game.calculateScore(self.board, True)   # 문양 기준
        second_score = Game.calculateScore(self.board, False) # 색 기준

        # player0 = first, player1 = second 라고 가정
        my_score = first_score
        opp_score = second_score

        # 승/무/패 기준 리워드 예시
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
        여기서는 스켈레톤이므로 간단한 one-hot/숫자 인코딩 예시를 준다.

        TODO: 네가 원하는 방식대로 수정하면 된다.
        """
        p = self.current_player
        opp = 1 - p

        # 보드 인코딩: 각 cell마다 [empty, my, opp, color(4), symbol(4)] = 3+4+4=11차원
        board_feat = np.zeros((self.num_cells, 11), dtype=np.float32)

        # cell -> index map
        cell_to_idx: Dict[Cell, int] = {c: i for i, c in enumerate(self.all_cells)}

        for (cell, tile) in self.board:
            idx = cell_to_idx[cell]
            # owner
            # 여기서는 "player0 tile인지, player1 tile인지"를 따로 저장하려면
            # board에 누가 뒀는지도 함께 저장해야 한다.
            # 현재 board에는 (Cell, Tile)만 있어서, 소유자 정보가 없다.
            # → 실제 구현에서는 board에 (Cell, Tile, owner) 형태로 바꾸는 것이 좋다.
            #
            # 일단 스켈레톤에서는 모두 "내 타일"이라고 가정해서 owner one-hot만 세팅해둘게.
            board_feat[idx, 1] = 1.0  # [empty, my, opp] 중 my=1

            # color one-hot (R,G,B,Y)
            color_idx = {"R":0, "G":1, "B":2, "Y":3}[tile.color.value]
            board_feat[idx, 3 + color_idx] = 1.0

            # symbol one-hot (1,2,3,4)
            sym_idx = {"1":0, "2":1, "3":2, "4":3}[tile.symbol.value]
            board_feat[idx, 7 + sym_idx] = 1.0

        board_flat = board_feat.flatten()  # (num_cells*11,)

        # 손패 인코딩: max_hand * 16 (타입 one-hot)
        hand_feat = np.zeros((self.max_hand, 16), dtype=np.float32)
        for i, tile in enumerate(self.hands[p]):
            if i >= self.max_hand:
                break
            t_idx = ALL_TILE_TYPES.index(tile)  # 느리지만 스켈레톤이니 일단
            hand_feat[i, t_idx] = 1.0
        hand_flat = hand_feat.flatten()

        # 메타: [current_player(0/1), turn_index/64]
        meta = np.array([
            float(p),
            float(self.turn_index) / 64.0,
        ], dtype=np.float32)

        obs = np.concatenate([board_flat, hand_flat, meta], axis=0)
        return obs
