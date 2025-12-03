# connexion_env.py

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

# NYPC 쪽 보드/타일 정의 재사용
from nypc_time_rich_plus_first_guard import (
    Tile,
    Cell,
    Color,
    Symbol,
    ALL_CELLS,
    ADJ,
)

from nypc_opponent import NYPCOpponent


class ConnexionEnv:
    """
    PPO 학습용 Connexion 환경.

    - 상태: 보드(64칸), 우리/상대 손패, 덱, 턴
    - 한 step: 에이전트가 한 수 두고 → NYPC 봇이 한 수 둠
    - 종료: 보드가 가득 찼을 때 (64개 모두 채워짐)
    - 보상: 게임 종료 시 (우리 점수 - 상대 점수) / 10.0
    """

    def __init__(self, seed: int = 0, opponent: str = "nypc"):
        self.rng = np.random.RandomState(seed)
        self.opponent_type = opponent  # "nypc" 또는 "random" (나중 확장용)

        # 보드 구조
        self.cells: List[Cell] = ALL_CELLS[:]  # 64칸

        # action: (cell_idx, hand_idx) = cell_idx * 5 + hand_idx
        self.max_hand_size = 5
        self.action_size = len(self.cells) * self.max_hand_size  # 64 * 5 = 320

        # obs_dim은 reset()에서 첫 obs를 보고 결정
        self.obs_dim: Optional[int] = None

        # 에이전트가 선공이라고 가정
        self.is_first_player = True

        # 내부 상태
        self.board: List[Optional[Tuple[str, Tile]]] = []
        self.agent_hand: List[Tile] = []
        self.opp_hand: List[Tile] = []
        self.agent_deck: List[Tile] = []
        self.opp_deck: List[Tile] = []
        self.bot: Optional[NYPCOpponent] = None

        self.done: bool = False
        self.episode_steps: int = 0

        # 색/문양 인덱스 매핑
        self.color_to_idx = {
            Color.R: 0,
            Color.G: 1,
            Color.B: 2,
            Color.Y: 3,
        }
        self.symbol_to_idx = {
            Symbol._1: 0,
            Symbol._2: 1,
            Symbol._3: 2,
            Symbol._4: 3,
        }

    # ───────────────────────── 덱/손패 초기화 ─────────────────────────

    def _init_deck_and_hands(self):
        # 전체 64장 타일
        full_tiles: List[Tile] = []
        for c in [Color.R, Color.G, Color.B, Color.Y]:
            for s in [Symbol._1, Symbol._2, Symbol._3, Symbol._4]:
                for _ in range(4):
                    full_tiles.append(Tile(c, s))

        self.rng.shuffle(full_tiles)

        # 32장씩 나누고, 앞 5장만 손패로
        self.agent_deck = full_tiles[:32]
        self.opp_deck = full_tiles[32:]

        self.agent_hand = self.agent_deck[:5]
        self.agent_deck = self.agent_deck[5:]

        self.opp_hand = self.opp_deck[:5]
        self.opp_deck = self.opp_deck[5:]

    # ───────────────────────── reset ─────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)

        self._init_deck_and_hands()
        self.board = [None] * len(self.cells)
        self.done = False
        self.episode_steps = 0

        # NYPC 봇 초기화
        if self.opponent_type == "nypc":
            bot_tiles_str = [str(t) for t in self.opp_hand]
            agent_tiles_str = [str(t) for t in self.agent_hand]
            # 에이전트가 선공 → 봇은 후공
            self.bot = NYPCOpponent(
                is_first_for_bot=not self.is_first_player,
                bot_tiles_str=bot_tiles_str,
                agent_tiles_str=agent_tiles_str,
            )
        else:
            self.bot = None

        obs = self._get_obs()
        if self.obs_dim is None:
            self.obs_dim = obs.shape[0]

        info = {}
        return obs, info

    # ───────────────────────── action 인코딩/디코딩 ─────────────────────────

    def _encode_action(self, cell_idx: int, hand_idx: int) -> int:
        return cell_idx * self.max_hand_size + hand_idx

    def _decode_action(self, action: int) -> Tuple[int, int]:
        cell_idx = action // self.max_hand_size
        hand_idx = action % self.max_hand_size
        return cell_idx, hand_idx

    # ───────────────────────── 유틸 ─────────────────────────

    def _draw_from_deck(self, deck: List[Tile]) -> Optional[Tile]:
        if not deck:
            return None
        return deck.pop(0)

    def _cell_index_from_str(self, cell_str: str) -> int:
        for i, c in enumerate(self.cells):
            if str(c) == cell_str:
                return i
        raise ValueError(f"Unknown cell string: {cell_str}")

    def _find_tile_index_in_hand(self, hand: List[Tile], tile_str: str) -> int:
        for i, t in enumerate(hand):
            if str(t) == tile_str:
                return i
        raise ValueError(f"Tile {tile_str} not found in given hand")

    # ───────────────────────── valid action mask ─────────────────────────

    def get_valid_action_mask(self) -> np.ndarray:
        """
        길이 self.action_size (=320)짜리 bool 배열 반환.
        - 보드에 비어 있는 칸
        - 손패에 존재하는 인덱스 (0..len(hand)-1)
        에 대해서만 True.
        """
        mask = np.zeros(self.action_size, dtype=bool)
        hand_size = len(self.agent_hand)
        for cell_idx, slot in enumerate(self.board):
            if slot is not None:
                continue  # 이미 돌 있음
            for hand_idx in range(hand_size):
                a = self._encode_action(cell_idx, hand_idx)
                mask[a] = True
        return mask

    # ───────────────────────── observation ─────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        관찰 벡터 구성:
        - 보드 64칸 * (empty/agent/opp 3 + color 4 + symbol 4) = 704
        - agent_hand 5칸 * 16 one-hot = 80
        - opp_hand 5칸 * 16 one-hot = 80
        - global 3 (남은 덱 크기 2, 진행 정도 1)
        총 867 차원 정도
        """
        features: List[float] = []

        # 1) 보드
        for i, slot in enumerate(self.board):
            if slot is None:
                occ_empty, occ_agent, occ_opp = 1.0, 0.0, 0.0
                color_oh = [0.0, 0.0, 0.0, 0.0]
                sym_oh = [0.0, 0.0, 0.0, 0.0]
            else:
                who, tile = slot
                occ_empty = 0.0
                occ_agent = 1.0 if who == "agent" else 0.0
                occ_opp = 1.0 if who == "opp" else 0.0

                c_idx = self.color_to_idx[tile.color]
                s_idx = self.symbol_to_idx[tile.symbol]

                color_oh = [0.0] * 4
                sym_oh = [0.0] * 4
                color_oh[c_idx] = 1.0
                sym_oh[s_idx] = 1.0

            features.extend([occ_empty, occ_agent, occ_opp])
            features.extend(color_oh)
            features.extend(sym_oh)

        # 2) 우리 손패 (최대 5장, 각 16 one-hot)
        def encode_hand(hand: List[Tile]) -> List[float]:
            out: List[float] = []
            for i in range(self.max_hand_size):
                if i < len(hand):
                    t = hand[i]
                    c_idx = self.color_to_idx[t.color]
                    s_idx = self.symbol_to_idx[t.symbol]
                    tid = c_idx * 4 + s_idx  # 0..15
                    oh = [0.0] * 16
                    oh[tid] = 1.0
                else:
                    oh = [0.0] * 16
                out.extend(oh)
            return out

        features.extend(encode_hand(self.agent_hand))
        features.extend(encode_hand(self.opp_hand))

        # 3) global features
        agent_deck_norm = len(self.agent_deck) / 32.0
        opp_deck_norm = len(self.opp_deck) / 32.0
        progress = self.episode_steps / 32.0  # 최대 32 step (양쪽 64수)

        features.extend([agent_deck_norm, opp_deck_norm, progress])

        return np.array(features, dtype=np.float32)

    # ───────────────────────── 점수 계산 ─────────────────────────

    def _compute_scores(self) -> Tuple[int, int]:
        """
        문제 원문 규칙대로 점수 계산:
        - 에이전트(선공): 같은 문양으로 연결된 그룹들 → (크기)^2 의 합
        - 봇(후공): 같은 색으로 연결된 그룹들 → (크기)^2 의 합
        """
        # index → (who, tile)
        agent_score = 0
        opp_score = 0

        # 1) 에이전트: 문양 기반 그룹
        visited = set()
        for i, slot in enumerate(self.board):
            if slot is None:
                continue
            who, tile = slot
            if who != "agent":
                continue
            if i in visited:
                continue

            # BFS로 같은 문양 + 인접 연결 성분 찾기
            target_sym = self.symbol_to_idx[tile.symbol]
            stack = [i]
            visited.add(i)
            comp_indices = [i]

            while stack:
                v = stack.pop()
                for nb in ADJ[v]:
                    nb_slot = self.board[nb]
                    if nb_slot is None:
                        continue
                    nb_who, nb_tile = nb_slot
                    if nb_who != "agent":
                        continue
                    if self.symbol_to_idx[nb_tile.symbol] != target_sym:
                        continue
                    if nb in visited:
                        continue
                    visited.add(nb)
                    stack.append(nb)
                    comp_indices.append(nb)

            sz = len(comp_indices)
            agent_score += sz * sz

        # 2) 상대: 색 기반 그룹
        visited = set()
        for i, slot in enumerate(self.board):
            if slot is None:
                continue
            who, tile = slot
            if who != "opp":
                continue
            if i in visited:
                continue

            target_col = self.color_to_idx[tile.color]
            stack = [i]
            visited.add(i)
            comp_indices = [i]

            while stack:
                v = stack.pop()
                for nb in ADJ[v]:
                    nb_slot = self.board[nb]
                    if nb_slot is None:
                        continue
                    nb_who, nb_tile = nb_slot
                    if nb_who != "opp":
                        continue
                    if self.color_to_idx[nb_tile.color] != target_col:
                        continue
                    if nb in visited:
                        continue
                    visited.add(nb)
                    stack.append(nb)
                    comp_indices.append(nb)

            sz = len(comp_indices)
            opp_score += sz * sz

        return agent_score, opp_score

    def _is_terminal(self) -> bool:
        # 모든 칸이 채워졌으면 종료
        return all(slot is not None for slot in self.board)

    def _final_reward(self) -> float:
        agent_score, opp_score = self._compute_scores()
        # 스케일을 너무 크지 않게 /10 정도
        return (agent_score - opp_score) / 10.0

    # ───────────────────────── step ─────────────────────────

    def step(self, action: int):
        assert not self.done, "Episode is done. Call reset()."

        # 1) 에이전트 수 적용
        cell_idx, hand_idx = self._decode_action(action)

        # invalid action 체크
        if cell_idx < 0 or cell_idx >= len(self.cells):
            self.done = True
            obs = self._get_obs()
            return obs, -100.0, True, False, {}

        if hand_idx >= len(self.agent_hand):
            # 손패에 없는 인덱스 → 강한 패널티
            self.done = True
            obs = self._get_obs()
            return obs, -100.0, True, False, {}

        if self.board[cell_idx] is not None:
            # 이미 채워진 칸
            self.done = True
            obs = self._get_obs()
            return obs, -100.0, True, False, {}

        cell = self.cells[cell_idx]
        tile = self.agent_hand[hand_idx]

        # 보드에 둠
        self.board[cell_idx] = ("agent", tile)

        # 덱에서 1장 뽑기
        drawn_agent_tile = self._draw_from_deck(self.agent_deck)
        drawn_agent_str = None if drawn_agent_tile is None else str(drawn_agent_tile)

        if drawn_agent_tile is None:
            # 더 뽑을 게 없으면 손패 크기 줄이기
            self.agent_hand.pop(hand_idx)
        else:
            # 같은 위치에 새로 뽑은 타일
            self.agent_hand[hand_idx] = drawn_agent_tile

        # NYPC 봇에게 우리 수 알려주기
        if self.bot is not None:
            self.bot.on_agent_move(
                cell_str=str(cell),
                tile_str=str(tile),
                drawn_tile_str=drawn_agent_str,
                used_time_ms=0,
            )

        # 에이전트 수만 둬도 게임이 끝났는지 체크
        if self._is_terminal():
            self.done = True
            reward = self._final_reward()
            obs = self._get_obs()
            self.episode_steps += 1
            return obs, reward, True, False, {}

        # 2) 봇 차례 (opponent move)
        if self.bot is not None:
            opp_cell_str, opp_tile_str = self.bot.choose_move(
                my_time_ms=800, opp_time_ms=800
            )

            opp_cell_idx = self._cell_index_from_str(opp_cell_str)
            opp_cell = self.cells[opp_cell_idx]
            opp_tile_idx = self._find_tile_index_in_hand(self.opp_hand, opp_tile_str)
            opp_tile = self.opp_hand[opp_tile_idx]

            # 보드에 상대 수 적용
            if self.board[opp_cell_idx] is not None:
                # 이상한 경우지만, 안전하게 패널티 주고 종료
                self.done = True
                obs = self._get_obs()
                return obs, -100.0, True, False, {}

            self.board[opp_cell_idx] = ("opp", opp_tile)

            drawn_opp_tile = self._draw_from_deck(self.opp_deck)
            drawn_opp_str = None if drawn_opp_tile is None else str(drawn_opp_tile)

            if drawn_opp_tile is None:
                self.opp_hand.pop(opp_tile_idx)
            else:
                self.opp_hand[opp_tile_idx] = drawn_opp_tile

            # 봇 내부 상태 동기화
            self.bot.on_bot_move_applied(
                cell_str=opp_cell_str,
                tile_str=opp_tile_str,
                drawn_tile_str=drawn_opp_str,
            )

        # 에피소드 진행 카운트
        self.episode_steps += 1

        # 종료 여부 & 보상
        if self._is_terminal():
            self.done = True
            reward = self._final_reward()
        else:
            reward = 0.0

        obs = self._get_obs()
        info = {}
        return obs, reward, self.done, False, info
