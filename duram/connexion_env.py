import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import torch # Self-play용

class Column(Enum):
    A = "a"; B = "b"; C = "c"; D = "d"; E = "e"; F = "f"
class Row(Enum):
    _1 = "1"; _2 = "2"; _3 = "3"; _4 = "4"; _5 = "5"; _6 = "6"
class Sign(Enum):
    MINUS = "-"; PLUS = "+"
class Color(Enum):
    R = "R"; G = "G"; B = "B"; Y = "Y"
class Symbol(Enum):
    _1 = "1"; _2 = "2"; _3 = "3"; _4 = "4"

FORBIDDEN_STR = {"a1-", "a4-", "c3+", "c6+", "d1-", "d4-", "f3+", "f6+"}

@dataclass(frozen=True)
class Cell:
    col: Column; row: Row; sign: Sign
    def __str__(self) -> str: return f"{self.col.value}{self.row.value}{self.sign.value}"
    def isValid(self) -> bool: return str(self) not in FORBIDDEN_STR
    def isAdjacent(self, other: "Cell") -> bool:
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)
        if self.sign == Sign.MINUS and other.sign == Sign.PLUS:
            return (dr == dc == 0 or (dr == 0 and dc == -1) or (dr == -1 and dc == 0))
        if self.sign == Sign.PLUS and other.sign == Sign.MINUS:
            return (dr == dc == 0 or (dr == 0 and dc == 1) or (dr == 1 and dc == 0))
        return False

@dataclass(frozen=True)
class Tile:
    color: Color; symbol: Symbol
    def __str__(self) -> str: return f"{self.color.value}{self.symbol.value}"

ALL_CELLS: List[Cell] = []
for col in Column:
    for row in Row:
        for sign in Sign:
            c = Cell(col, row, sign)
            if c.isValid(): ALL_CELLS.append(c)

NUM_CELLS: int = len(ALL_CELLS)
CELL_INDEX: Dict[str, int] = {str(c): i for i, c in enumerate(ALL_CELLS)}
ADJACENT: List[List[int]] = [[] for _ in range(NUM_CELLS)]
for i, ci in enumerate(ALL_CELLS):
    for j, cj in enumerate(ALL_CELLS):
        if i != j and ci.isAdjacent(cj):
            ADJACENT[i].append(j)

COLOR_LIST = [Color.R, Color.G, Color.B, Color.Y]
SYMBOL_LIST = [Symbol._1, Symbol._2, Symbol._3, Symbol._4]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_LIST)}
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOL_LIST)}

def tile_to_id(t: Tile) -> int: return COLOR_TO_IDX[t.color] * 4 + SYMBOL_TO_IDX[t.symbol]
def id_to_tile(tid: int) -> Tile: return Tile(COLOR_LIST[tid // 4], SYMBOL_LIST[tid % 4])

class ConnexionEnv:
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.num_cells = NUM_CELLS
        self.action_size = self.num_cells * 5
        self.obs_dim = self.num_cells * 16 + 5 * 16
        self.reward_scale = 40.0 
        self.win_bonus = 1.0
        self.greedy_prob = 0.0 
        self.reset()

    def set_greedy_prob(self, prob: float):
        self.greedy_prob = np.clip(prob, 0.0, 1.0)

    def reset(self, seed: Optional[int] = None):
        if seed is not None: self.rng.seed(seed)
        full_deck_tiles = [Tile(c, s) for c in COLOR_LIST for s in SYMBOL_LIST for _ in range(4)]
        self.rng.shuffle(full_deck_tiles)
        full_deck = [tile_to_id(t) for t in full_deck_tiles]
        self.agent_deck = full_deck[:27]
        self.opp_deck = full_deck[27:54]
        self.agent_hand = full_deck[54:59]
        self.opp_hand = full_deck[59:64]
        self.board = [-1] * self.num_cells
        self.filled = 0
        self.done = False
        self.prev_diff = 0.0
        return self._get_obs(), {}

    def _get_obs(self, is_opponent=False) -> np.ndarray:
        # 상대방(Self-Play)을 위한 관측 생성 기능 추가
        target_hand = self.opp_hand if is_opponent else self.agent_hand
        
        board_feat = np.zeros((self.num_cells, 16), dtype=np.float32)
        for idx, tid in enumerate(self.board):
            if tid != -1: board_feat[idx, tid] = 1.0
        
        hand_feat = np.zeros((5, 16), dtype=np.float32)
        for slot, tid in enumerate(target_hand):
             if slot < len(target_hand): hand_feat[slot, tid] = 1.0
        
        return np.concatenate([board_feat.flatten(), hand_feat.flatten()])

    def get_valid_action_mask(self, is_opponent=False) -> np.ndarray:
        target_hand = self.opp_hand if is_opponent else self.agent_hand
        mask = np.zeros(self.action_size, dtype=bool)
        if self.done: return mask
        for slot in range(len(target_hand)):
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    mask[slot * self.num_cells + ci] = True
        return mask
    
    # ---------------------------------------------------
    # [NEW] Smart Masking (외부 호출용)
    # ---------------------------------------------------
    def get_smart_action_mask(self) -> np.ndarray:
        # 학습 가속을 위해 '점수가 70% 이상 나는 수'만 True로 반환
        mask = np.zeros(self.action_size, dtype=bool)
        if self.done: return mask
        
        scores, indices = [], []
        for slot, tid in enumerate(self.agent_hand):
            for ci in range(self.num_cells):
                if self.board[ci] == -1:
                    idx = slot * self.num_cells + ci
                    s = self._local_score(ci, tid, is_first=True)
                    scores.append(s); indices.append(idx)
        
        if not scores: return mask
        max_s = max(scores)
        threshold = max_s * 0.7 if max_s > 0 else -1.0
        for s, idx in zip(scores, indices):
            if s >= threshold: mask[idx] = True
        return mask

    def step(self, action: int, opp_model=None, device=None):
        if self.done: raise RuntimeError("Episode done")
        
        # 1. Agent Move
        tile_slot = action // self.num_cells
        cell_idx = action % self.num_cells
        if self.board[cell_idx] != -1: # 예외처리
             return self._get_obs(), -5.0, True, False, {"invalid": True}

        self._apply_move(is_agent=True, slot=tile_slot, cell_idx=cell_idx)
        if self.filled >= self.num_cells: return self._finalize_step()

        # 2. Opponent Move (Self-Play or Greedy)
        if opp_model is not None:
            self._opponent_model_move(opp_model, device)
        else:
            self._opponent_move()
            
        if self.filled >= self.num_cells: return self._finalize_step()

        return self._finalize_step(done_override=False)

    def _finalize_step(self, done_override=None):
        first_s, second_s, curr_diff = self._compute_scores()
        delta = curr_diff - self.prev_diff
        self.prev_diff = curr_diff
        step_reward = np.clip(delta / self.reward_scale, -2.0, 2.0)
        is_done = (self.filled >= self.num_cells) if done_override is None else done_override
        self.done = is_done
        info = {"diff": curr_diff, "first": first_s, "second": second_s}
        if is_done:
            if curr_diff > 0: step_reward += self.win_bonus
            elif curr_diff < 0: step_reward -= self.win_bonus
        return self._get_obs(), step_reward, is_done, False, info

    def _apply_move(self, is_agent: bool, slot: int, cell_idx: int):
        hand = self.agent_hand if is_agent else self.opp_hand
        deck = self.agent_deck if is_agent else self.opp_deck
        tid = hand.pop(slot)
        self.board[cell_idx] = tid
        self.filled += 1
        if deck: hand.append(deck.pop())

    # ---------------------------------------------------
    # [NEW] Self-Play Logic (Hybrid)
    # ---------------------------------------------------
    def _opponent_model_move(self, model, device):
        # 1. 모델 추천 (Intuition)
        obs = self._get_obs(is_opponent=True)
        mask = self.get_valid_action_mask(is_opponent=True)
        
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Actor Forward
            x, h = model.forward_shared(obs_t)
            b = x.size(0)
            pol = torch.nn.functional.relu(model.actor_bn(model.actor_conv(x))).view(b, -1)
            pol = torch.cat([pol, h], dim=1)
            logits = model.actor_fc(pol)
            logits = logits.masked_fill(~mask_t, -1e9)
            ppo_idx = torch.argmax(logits, dim=1).item()

        # 2. Greedy 계산 (Calculation) - 후공(False) 기준 점수
        # Self-Play 상대는 P2이므로 '후공 점수'를 최대화해야 함
        greedy_idx = -1
        best_val = -9999
        
        # PPO 추천 수의 점수
        ppo_val = self._simulate_opp_score(ppo_idx)
        
        # Greedy 탐색 (간략화: 상위 10개만 봐도 됨, 여기선 전체)
        for slot in range(len(self.opp_hand)):
            for c_idx in range(self.num_cells):
                if self.board[c_idx] == -1:
                    idx = slot * self.num_cells + c_idx
                    # 로컬 점수(간략)가 아니라 실제 시뮬레이션 점수 사용
                    val = self._simulate_opp_score(idx)
                    if val > best_val:
                        best_val = val
                        greedy_idx = idx
        
        # 3. Hybrid 결정 (후공 입장이므로 점수가 높을수록 좋음)
        # PPO가 Greedy보다 5점 이상 손해면 Greedy 선택
        if greedy_idx != -1 and best_val > ppo_val + 5.0:
            final_action = greedy_idx
        else:
            final_action = ppo_idx
            
        # 4. 착수
        slot = final_action // self.num_cells
        c_idx = final_action % self.num_cells
        if self.board[c_idx] != -1: # 에러 방지
             # Fallback
             self._opponent_random()
        else:
             self._apply_move(is_agent=False, slot=slot, cell_idx=c_idx)

    def _simulate_opp_score(self, action_idx):
        # 상대방(P2) 입장에서 이 수를 두면 점수 차(P2 - P1)가 얼마나 되는지 계산
        slot = action_idx // self.num_cells
        c_idx = action_idx % self.num_cells
        
        # Backup
        bak_hand = list(self.opp_hand)
        
        # Apply
        tid = self.opp_hand.pop(slot)
        self.board[c_idx] = tid
        
        # Calc (P2 - P1)
        s1, s2, diff = self._compute_scores() # diff = P1 - P2
        my_gain = s2 - s1 # P2 입장 이득
        
        # Restore
        self.board[c_idx] = -1
        self.opp_hand = bak_hand
        
        return my_gain

    def _opponent_move(self):
        if self.rng.rand() < self.greedy_prob: self._opponent_sample_greedy()
        else: self._opponent_random()

    def _opponent_random(self):
        valid = []
        for s in range(len(self.opp_hand)):
            for c in range(self.num_cells):
                if self.board[c] == -1: valid.append((s, c))
        if valid:
            s, c = valid[self.rng.randint(len(valid))]
            self._apply_move(False, s, c)

    def _opponent_sample_greedy(self):
        best_val = -1; best_moves = []
        for s, tid in enumerate(self.opp_hand):
            for c in range(self.num_cells):
                if self.board[c] == -1:
                    val = self._local_score(c, tid, is_first=False)
                    if val > best_val: best_val = val; best_moves = [(s, c)]
                    elif val == best_val: best_moves.append((s, c))
        if best_moves:
            s, c = best_moves[self.rng.randint(len(best_moves))]
            self._apply_move(False, s, c)

    def _local_score(self, cell_idx, tid, is_first):
        score = 1
        c_idx, s_idx = tid // 4, tid % 4
        for nb in ADJACENT[cell_idx]:
            nb_tid = self.board[nb]
            if nb_tid == -1: continue
            match = (nb_tid % 4 == s_idx) if is_first else (nb_tid // 4 == c_idx)
            if match: score += 1
        return score * score

    def _compute_scores(self):
        return (self._calc_uf(True), self._calc_uf(False), self._calc_uf(True) - self._calc_uf(False))

    def _calc_uf(self, use_symbol: bool):
        parent = list(range(self.num_cells)); size = [0] * self.num_cells
        for i, tid in enumerate(self.board):
            if tid != -1: size[i] = 1
        def find(x):
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra; size[ra] += size[rb]
        for i in range(self.num_cells):
            tid = self.board[i]
            if tid == -1: continue
            for nb in ADJACENT[i]:
                if nb > i: continue
                ntid = self.board[nb]
                if ntid == -1: continue
                match = (tid % 4 == ntid % 4) if use_symbol else (tid // 4 == ntid // 4)
                if match: union(i, nb)
        score = 0
        for i in range(self.num_cells):
            if parent[i] == i and size[i] > 0: score += size[i] ** 2
        return score