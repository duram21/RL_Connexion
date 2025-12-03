# connexion_env.py
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any


# ───────────────────────── Enums & 기본 구조 ─────────────────────────

class Column(Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"


class Row(Enum):
    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"
    _5 = "5"
    _6 = "6"


class Sign(Enum):
    MINUS = "-"
    PLUS = "+"


class Color(Enum):
    R = "R"
    G = "G"
    B = "B"
    Y = "Y"


class Symbol(Enum):
    _1 = "1"
    _2 = "2"
    _3 = "3"
    _4 = "4"


FORBIDDEN_STR = {
    "a1-", "a4-",
    "c3+", "c6+",
    "d1-", "d4-",
    "f3+", "f6+",
}


@dataclass(frozen=True)
class Cell:
    col: Column
    row: Row
    sign: Sign

    def __str__(self) -> str:
        return f"{self.col.value}{self.row.value}{self.sign.value}"

    def isValid(self) -> bool:
        return str(self) not in FORBIDDEN_STR

    def isAdjacent(self, other: "Cell") -> bool:
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
        ret: List[Cell] = []
        for col in Column:
            for row in Row:
                for sign in Sign:
                    c = cls(col, row, sign)
                    if c.isValid():
                        ret.append(c)
        return ret

    @classmethod
    def fromString(cls, s: str) -> "Cell":
        assert len(s) == 3
        cell = cls(Column(s[0]), Row(s[1]), Sign(s[2]))
        assert cell.isValid()
        return cell


@dataclass(frozen=True)
class Tile:
    color: Color
    symbol: Symbol

    def __str__(self) -> str:
        return f"{self.color.value}{self.symbol.value}"

    @classmethod
    def fromString(cls, s: str) -> "Tile":
        assert len(s) == 2
        return cls(Color(s[0]), Symbol(s[1]))


# ───────────────────────── 전역 보드/타일 유틸 ─────────────────────────

ALL_CELLS: List[Cell] = Cell.getAllCells()
NUM_CELLS: int = len(ALL_CELLS)  # 64
CELL_INDEX: Dict[str, int] = {str(c): i for i, c in enumerate(ALL_CELLS)}

# 인접 리스트 (자기 자신 제외)
ADJACENT: List[List[int]] = [[] for _ in range(NUM_CELLS)]
for i, ci in enumerate(ALL_CELLS):
    for j, cj in enumerate(ALL_CELLS):
        if i != j and ci.isAdjacent(cj):
            ADJACENT[i].append(j)

COLOR_LIST = [Color.R, Color.G, Color.B, Color.Y]
SYMBOL_LIST = [Symbol._1, Symbol._2, Symbol._3, Symbol._4]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_LIST)}
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOL_LIST)}


def tile_to_id(t: Tile) -> int:
    """
    0~15 로 타일 인덱싱
    color_idx * 4 + symbol_idx
    """
    c = COLOR_TO_IDX[t.color]
    s = SYMBOL_TO_IDX[t.symbol]
    return c * 4 + s


def id_to_tile(tid: int) -> Tile:
    c_idx = tid // 4
    s_idx = tid % 4
    return Tile(COLOR_LIST[c_idx], SYMBOL_LIST[s_idx])


# ───────────────────────── Env 본체 ─────────────────────────

class ConnexionEnv:
    """
    간단한 Connexion 환경

    - 에이전트: 항상 선공 (문양 기반 점수)
    - 상대: 샘플 코드 기반의 greedy 봇 (후공; 색상 기반 점수)
      -> 한 수 넣었을 때 '자기 색 그룹'이 얼마나 커지는지 근처 이웃 기준으로 로컬 스코어를 최대화.
    - 액션: 0 ~ 319 = (hand_slot 0~4) * 64 + cell_idx(0~63)
    - 관측(obs):
        - 보드: [64, 16] one-hot (타입별) → flatten → 1024
        - 내 손패: [5, 16] one-hot → flatten → 80
        - 총 obs_dim = 1104
    """

    def __init__(self, seed: int = 0, opponent: str = "sample"):
        self.rng = np.random.RandomState(seed)
        assert opponent in ("sample", "random")
        self.opponent_type = opponent

        self.num_cells = NUM_CELLS
        self.action_size = self.num_cells * 5  # 5칸 손패 * 64칸
        self.obs_dim = self.num_cells * 16 + 5 * 16  # 보드 16차원 one-hot + 손패 16차원 one-hot

        # 게임 상태
        self.board: List[int] = []      # 각 칸에 놓인 타일 id (없으면 -1)
        self.filled: int = 0            # 놓인 타일 수
        self.agent_deck: List[int] = [] # 남은 내 덱
        self.opp_deck: List[int] = []   # 남은 상대 덱
        self.agent_hand: List[int] = [] # 내 손패 (tile_id)
        self.opp_hand: List[int] = []   # 상대 손패 (tile_id)
        self.done: bool = False
        self.last_result: Dict[str, Any] = {}

        self.reset()

    # ───────────────────────── 기본 유틸 ─────────────────────────

    def _make_full_deck(self) -> List[Tile]:
        """16가지 조합 × 4장 = 64장 전체 덱"""
        tiles: List[Tile] = []
        for c in COLOR_LIST:
            for s in SYMBOL_LIST:
                for _ in range(4):
                    tiles.append(Tile(c, s))
        return tiles

    def _get_obs(self) -> np.ndarray:
        """
        보드 + 내 손패를 one-hot으로 flatten 해서 반환
        """
        board_feat = np.zeros((self.num_cells, 16), dtype=np.float32)
        for idx, tid in enumerate(self.board):
            if tid != -1:
                board_feat[idx, tid] = 1.0

        hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(self.agent_hand):
            if slot >= 5:
                break
            hand_feat[slot, tid] = 1.0

        obs = np.concatenate([board_feat.flatten(), hand_feat.flatten()])
        return obs

    # ───────────────────────── Gym-ish API ─────────────────────────

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        full_deck = self._make_full_deck()
        self.rng.shuffle(full_deck)

        # 32장씩 나눠서 각자 덱으로
        agent_tiles = full_deck[:32]
        opp_tiles = full_deck[32:]

        self.agent_deck = [tile_to_id(t) for t in agent_tiles]
        self.opp_deck = [tile_to_id(t) for t in opp_tiles]

        # 처음 손패 5장
        self.agent_hand = self.agent_deck[:5]
        self.agent_deck = self.agent_deck[5:]

        self.opp_hand = self.opp_deck[:5]
        self.opp_deck = self.opp_deck[5:]

        # 보드 비우기
        self.board = [-1] * self.num_cells
        self.filled = 0
        self.done = False
        self.last_result = {}

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def get_valid_action_mask(self) -> np.ndarray:
        """
        현재 내 손패/보드 기준으로 유효한 (hand_slot, cell_idx)만 True
        """
        mask = np.zeros(self.action_size, dtype=bool)
        if self.done:
            return mask

        for slot, tid in enumerate(self.agent_hand):
            if slot >= 5:
                break
            # 해당 손패를 놓을 수 있는 빈 칸들
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    a = slot * self.num_cells + ci
                    mask[a] = True

        return mask

    def step(self, action: int):
        """
        action: [0, 320) 정수
        반환: obs, reward, done, truncated(False), info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # 액션 디코딩
        tile_slot = action // self.num_cells   # 0~4
        cell_idx = action % self.num_cells     # 0~63

        valid_mask = self.get_valid_action_mask()
        if action < 0 or action >= self.action_size or not valid_mask[action]:
            # 잘못된 액션 → 큰 패널티 주고 종료
            self.done = True
            obs = self._get_obs()
            info = {"invalid_action": True}
            return obs, -50.0, True, False, info

        # 1) 에이전트 수 두기
        self._apply_agent_move(tile_slot, cell_idx)

        # 보드가 꽉 찼으면 게임 종료
        if self.filled >= self.num_cells:
            self.done = True
            reward = self._compute_final_reward()
            obs = self._get_obs()
            info = {"final": True, **self.last_result}
            return obs, reward, True, False, info

        # 2) 상대 수 두기
        self._opponent_move()

        # 상대까지 두고 난 뒤 보드가 꽉 찼는지 확인
        if self.filled >= self.num_cells:
            self.done = True
            reward = self._compute_final_reward()
            obs = self._get_obs()
            info = {"final": True, **self.last_result}
            return obs, reward, True, False, info

        # 중간 단계에서는 보상 0
        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    # ───────────────────────── 에이전트/상대 수 적용 ─────────────────────────

    def _apply_agent_move(self, slot: int, cell_idx: int):
        tid = self.agent_hand[slot]
        assert self.board[cell_idx] == -1
        self.board[cell_idx] = tid
        self.filled += 1

        # 손패에서 제거 + 덱에서 한 장 뽑기
        self.agent_hand.pop(slot)
        if self.agent_deck:
            self.agent_hand.append(self.agent_deck.pop())

    def _apply_opp_move(self, slot: int, cell_idx: int):
        tid = self.opp_hand[slot]
        assert self.board[cell_idx] == -1
        self.board[cell_idx] = tid
        self.filled += 1

        self.opp_hand.pop(slot)
        if self.opp_deck:
            self.opp_hand.append(self.opp_deck.pop())

    def _opponent_move(self):
        if self.filled >= self.num_cells:
            return
        if not self.opp_hand:
            return

        if self.opponent_type == "random":
            self._opponent_random()
        else:
            self._opponent_sample_greedy()

    def _opponent_random(self):
        moves: List[Tuple[int, int]] = []
        for slot, tid in enumerate(self.opp_hand):
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    moves.append((slot, ci))
        if not moves:
            return
        slot, ci = moves[self.rng.randint(len(moves))]
        self._apply_opp_move(slot, ci)

    # ───────────────────────── 샘플 코드 기반 Greedy 상대 ─────────────────────────

    def _local_group_score(self, cell_idx: int, tid: int, is_first: bool) -> int:
        """
        샘플 코드의 calculateScore를 그대로 돌리면 너무 느려서,
        "지금 이 칸에 이 타일을 놓았을 때 인접한 같은 속성 타일의 개수"로
        로컬 점수(그룹 크기^2)를 계산하는 간단한 휴리스틱.
        - is_first=True  → 문양 기준 (선공)
        - is_first=False → 색 기준  (후공)
        """
        color_idx = tid // 4
        sym_idx = tid % 4
        same = 1  # 자기 자신

        for nb in ADJACENT[cell_idx]:
            other_tid = self.board[nb]
            if other_tid == -1:
                continue
            if is_first:
                if (other_tid % 4) == sym_idx:
                    same += 1
            else:
                if (other_tid // 4) == color_idx:
                    same += 1

        return same * same

    def _opponent_sample_greedy(self):
        """
        샘플 코드 Game.calculateMove와 비슷한 느낌:
        - 모든 (손패 타일, 빈 칸)에 대해
        - '후공 기준(색 기준) 로컬 그룹 점수'가 최대가 되는 수를 고른다.
        """
        best_val = -1.0
        best_moves: List[Tuple[int, int]] = []

        for slot, tid in enumerate(self.opp_hand):
            for ci in range(self.num_cells):
                if self.board[ci] != -1:
                    continue
                val = self._local_group_score(ci, tid, is_first=False)  # 후공: 색 기준
                if val > best_val:
                    best_val = val
                    best_moves = [(slot, ci)]
                elif val == best_val:
                    best_moves.append((slot, ci))

        if not best_moves:
            return
        slot, ci = best_moves[self.rng.randint(len(best_moves))]
        self._apply_opp_move(slot, ci)

    # ───────────────────────── 최종 점수 계산 ─────────────────────────

    def _compute_score(self, use_symbol: bool) -> int:
        """
        use_symbol=True  → 문양으로 연결된 그룹들(선공 기준)
        use_symbol=False → 색으로 연결된 그룹들(후공 기준)
        각각 그룹 크기의 제곱을 더해서 점수 계산.
        """
        parent = list(range(self.num_cells))
        size = [0] * self.num_cells

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        # 초기: 타일이 놓인 칸만 size=1
        for i, tid in enumerate(self.board):
            if tid != -1:
                size[i] = 1

        # 인접 & 같은 속성인 칸끼리 DSU union
        for i, tid in enumerate(self.board):
            if tid == -1:
                continue
            for j in ADJACENT[i]:
                if j <= i:
                    continue
                tid2 = self.board[j]
                if tid2 == -1:
                    continue
                if use_symbol:
                    if (tid % 4) != (tid2 % 4):
                        continue
                else:
                    if (tid // 4) != (tid2 // 4):
                        continue
                union(i, j)

        # 루트별로 size^2 합
        score = 0
        for i in range(self.num_cells):
            if parent[i] == i and size[i] > 0:
                score += size[i] * size[i]
        return score

    def _compute_final_reward(self) -> float:
        """
        - 에이전트는 항상 선공(문양 기준)
        - 상대는 후공(색 기준)
        보상 = 선공 점수 - 후공 점수
        """
        first_score = self._compute_score(use_symbol=True)
        second_score = self._compute_score(use_symbol=False)
        self.last_result = {
            "first_score": first_score,
            "second_score": second_score,
        }
        return float(first_score - second_score)
