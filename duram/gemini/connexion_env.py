import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

# ==============================================================================
# 1. Connexion Game Logic (Sample Code Based)
#    - 제공해주신 샘플 코드의 클래스들을 재사용하되, 시뮬레이션 용도로 일부 수정
# ==============================================================================

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

@dataclass(unsafe_hash=True) # Dictionary Key로 쓰기 위해 hashing 허용
class Cell:
    col: Column; row: Row; sign: Sign
    
    def __str__(self): return f"{self.col.value}{self.row.value}{self.sign.value}"
    
    def isValid(self) -> bool:
        # (샘플 코드 로직 동일)
        invalid_list = [
            ("a", "1", "-"), ("a", "4", "-"), ("c", "3", "+"), ("c", "6", "+"),
            ("d", "1", "-"), ("d", "4", "-"), ("f", "3", "+"), ("f", "6", "+")
        ]
        return (self.col.value, self.row.value, self.sign.value) not in invalid_list

    def isAdjacent(self, other: "Cell") -> bool:
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)
        if self.sign == Sign.MINUS and other.sign == Sign.PLUS:
            return (dr == dc == 0) or (dr == 0 and dc == -1) or (dr == -1 and dc == 0)
        if self.sign == Sign.PLUS and other.sign == Sign.MINUS:
            return (dr == dc == 0) or (dr == 0 and dc == 1) or (dr == 1 and dc == 0)
        return False

    @classmethod
    def getAllCells(cls) -> List["Cell"]:
        ret = []
        for col in Column:
            for row in Row:
                for sign in Sign:
                    c = cls(col, row, sign)
                    if c.isValid(): ret.append(c)
        return ret # Total 64 cells

@dataclass
class Tile:
    color: Color; symbol: Symbol
    def __str__(self): return f"{self.color.value}{self.symbol.value}"
    def isSame(self, other: "Tile", isFirst: bool) -> bool:
        return self.symbol == other.symbol if isFirst else self.color == other.color

# 전역적으로 사용할 Cell 매핑 (0~63 인덱스 <-> Cell 객체)
ALL_CELLS = Cell.getAllCells()
CELL_TO_IDX = {cell: i for i, cell in enumerate(ALL_CELLS)}

# ==============================================================================
# 2. Gymnasium Environment for Connexion
# ==============================================================================

class ConnexionEnv(gym.Env):
    def __init__(self):
        super(ConnexionEnv, self).__init__()
        
        # Action Space: 5(내 손패 슬롯) * 64(보드 위치) = 320가지
        self.action_space = gym.spaces.Discrete(5 * 64)
        
        # Observation Space: (Neural Net 입력 크기)
        # Board 64개: [IsOccupied(1), R,G,B,Y, 1,2,3,4 (8)] -> 9 features per cell
        # Hand 5개: [IsPresent(1), R,G,B,Y, 1,2,3,4 (8)] -> 9 features per slot
        # Total: 64*9 + 5*9 = 576 + 45 = 621
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(621,), dtype=np.float32)

        self.cells = ALL_CELLS
        self.deck = []
        self.my_hand = []
        self.opp_hand = []
        self.board_state = {} # Dict[Cell, Tile]
        self.is_first = True  # Agent is First Player (Symbol)
        self.turn_count = 0
        self.max_turns = 64   # 32 turns each * 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 타일 덱 생성 및 셔플
        self.deck = []
        for c in Color:
            for s in Symbol:
                for _ in range(4): # 각 4개씩
                    self.deck.append(Tile(c, s))
        random.shuffle(self.deck)
        
        # 2. 초기 손패 분배 (각 5개)
        self.my_hand = [self.deck.pop() for _ in range(5)]
        self.opp_hand = [self.deck.pop() for _ in range(5)]
        
        self.board_state = {}
        self.turn_count = 0
        
        # Agent가 선공(First)인지 후공인지 랜덤 결정 (학습 다양성)
        # 이번 구현에서는 Agent가 항상 'isFirst'라고 가정하고 
        # Opponent가 'Second' 역할을 하도록 고정하거나, 파라미터로 받음.
        # 편의상 Agent = First(문양), Opponent = Second(색)으로 고정
        self.agent_role_first = True 

        return self._get_observation(), {"mask": self._get_action_mask()}

    def step(self, action_idx):
        # 1. Action Decoding
        hand_idx = action_idx // 64
        board_idx = action_idx % 64
        
        target_cell = self.cells[board_idx]
        
        # 2. 유효성 검사 (이미 마스킹을 했겠지만, 환경 차원에서 더블 체크)
        if hand_idx >= len(self.my_hand) or target_cell in self.board_state:
            # 치명적 오류 (PPO Masking이 실패했을 때)
            return self._get_observation(), -100, True, False, {"mask": self._get_action_mask()}

        # 3. Agent 착수 (PUT)
        tile_to_place = self.my_hand.pop(hand_idx)
        self.board_state[target_cell] = tile_to_place
        
        # 4. Agent 타일 보충 (GET) - 마지막 5턴 제외
        if len(self.deck) > 0:
            self.my_hand.append(self.deck.pop())
            
        self.turn_count += 1
        
        # 5. 게임 종료 체크
        if self.turn_count >= self.max_turns:
            reward = self._calculate_final_reward()
            return self._get_observation(), reward, True, False, {"mask": self._get_action_mask()}

        # 6. Opponent(상대방) 턴 진행 (Greedy Bot simulation)
        self._opponent_step()
        self.turn_count += 1
        
        done = self.turn_count >= self.max_turns
        
        # 7. 보상 계산
        # Dense Reward: (내 현재 점수 - 상대 현재 점수)의 변화량 사용 가능
        # Sparse Reward: 게임 끝났을 때 승패
        # 여기서는 매 턴 점수 차이를 보상으로 줍니다.
        current_score_diff = self._calculate_score_diff()
        reward = current_score_diff * 0.1 # 스케일링
        
        if done:
            reward += self._calculate_final_reward()

        return self._get_observation(), reward, done, False, {"mask": self._get_action_mask()}

    def _opponent_step(self):
        # 상대방은 Greedy하게 둡니다 (샘플 코드 로직 활용)
        best_score = -1
        best_move = None # (hand_idx, cell)
        
        # 상대방의 가상 보드 구성
        current_board_list = list(self.board_state.items())
        
        # 모든 가능한 수 탐색
        for i, tile in enumerate(self.opp_hand):
            for cell in self.cells:
                if cell not in self.board_state:
                    # 가상으로 둬보기
                    temp_board = current_board_list + [(cell, tile)]
                    # 상대방은 Second(색깔) 기준 점수 계산
                    score = self._calc_score_logic(temp_board, isFirst=not self.agent_role_first)
                    
                    if score > best_score:
                        best_score = score
                        best_move = (i, cell)
        
        if best_move:
            h_idx, target_c = best_move
            played_tile = self.opp_hand.pop(h_idx)
            self.board_state[target_c] = played_tile
            
            if len(self.deck) > 0:
                self.opp_hand.append(self.deck.pop())
        else:
            # 둘 곳이 없는 경우 (거의 없지만) 패스
            pass

    def _get_observation(self):
        # 보드 상태 인코딩 (64 * 9)
        board_vec = []
        for cell in self.cells:
            if cell in self.board_state:
                tile = self.board_state[cell]
                # [Occupied(1), R, G, B, Y, 1, 2, 3, 4]
                feat = [1.0]
                feat += [1.0 if tile.color == c else 0.0 for c in Color]
                feat += [1.0 if tile.symbol == s else 0.0 for s in Symbol]
            else:
                feat = [0.0] * 9
            board_vec.extend(feat)
            
        # 내 손패 인코딩 (5 * 9)
        hand_vec = []
        for i in range(5):
            if i < len(self.my_hand):
                tile = self.my_hand[i]
                feat = [1.0]
                feat += [1.0 if tile.color == c else 0.0 for c in Color]
                feat += [1.0 if tile.symbol == s else 0.0 for s in Symbol]
            else:
                feat = [0.0] * 9
            hand_vec.extend(feat)
            
        return np.array(board_vec + hand_vec, dtype=np.float32)

    def _get_action_mask(self):
        # 1: Valid, 0: Invalid
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        
        occupied_indices = {CELL_TO_IDX[c] for c in self.board_state.keys()}
        
        for h_idx in range(len(self.my_hand)): # 내 손패가 있는 슬롯만
            for c_idx in range(64):
                if c_idx not in occupied_indices: # 빈 칸만
                    action = h_idx * 64 + c_idx
                    mask[action] = 1.0
        return mask

    def _calc_score_logic(self, board_list, isFirst):
        # 샘플 코드의 calculateScore 로직 (Warshall Algo)
        if not board_list: return 0
        n = len(board_list)
        adj = [[False]*n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                c1, t1 = board_list[i]
                c2, t2 = board_list[j]
                if i == j: adj[i][j] = True
                elif c1.isAdjacent(c2) and t1.isSame(t2, isFirst):
                    adj[i][j] = True
                    
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True
                        
        # 점수 계산: (그룹 크기)^2 합
        visited = [False] * n
        total_score = 0
        for i in range(n):
            if not visited[i]:
                group_size = 0
                for j in range(n):
                    if adj[i][j]:
                        visited[j] = True
                        group_size += 1
                total_score += group_size * group_size
        return total_score

    def _calculate_score_diff(self):
        board_list = list(self.board_state.items())
        my_score = self._calc_score_logic(board_list, self.agent_role_first)
        opp_score = self._calc_score_logic(board_list, not self.agent_role_first)
        return my_score - opp_score

    def _calculate_final_reward(self):
        diff = self._calculate_score_diff()
        if diff > 0: return 10.0 # 승리 보너스
        elif diff < 0: return -10.0 # 패배 페널티
        return 0.0


# ==============================================================================
# 3. PPO with Action Masking
# ==============================================================================

class MaskablePPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MaskablePPO, self).__init__()
        self.data = []
        
        # 입력 차원: 621 -> Hidden 512 -> Hidden 256
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.fc_pi = nn.Linear(256, action_dim) # Policy Head
        self.fc_v = nn.Linear(256, 1)           # Value Head
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)

    def pi(self, x, mask, softmax_dim=0):
        # x: State, mask: Action Mask
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        logits = self.fc_pi(x)
        
        # [핵심] Masking: 불가능한 행동의 logit을 -무한대로 설정
        # mask가 0인 곳을 -1e9로 채움
        logits = logits.masked_fill(mask == 0, -1e9)
        
        prob = F.softmax(logits, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, mask_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done, mask = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            mask_lst.append(mask)
            
        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst))
        r = torch.tensor(np.array(r_lst))
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst))
        mask = torch.tensor(np.array(mask_lst), dtype=torch.bool) # Mask도 텐서로
        
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, mask
        
    def train_net(self):
        if len(self.data) == 0: return
        
        s, a, r, s_prime, done_mask, prob_a, mask = self.make_batch()
        
        # Hyperparameters
        gamma = 0.98
        lmbda = 0.95
        eps_clip = 0.1
        K_epoch = 3

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # Re-calculate probs with current network
            # 여기서 mask를 꼭 다시 넣어줘야 함
            pi_probs = self.pi(s, mask, softmax_dim=1) 
            pi_a = pi_probs.gather(1, a)
            
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

# ==============================================================================
# 4. Main Training Loop
# ==============================================================================

def main():
    env = ConnexionEnv()
    state_dim = env.observation_space.shape[0] # 621
    action_dim = env.action_space.n # 320
    
    model = MaskablePPO(state_dim, action_dim)
    
    score = 0.0
    print_interval = 10
    T_horizon = 64 # 한 게임 길이만큼 모아서 업데이트 (Episode 단위 업데이트)

    print("Training Start...")
    
    for n_epi in range(10000):
        s, info = env.reset()
        mask = info["mask"]
        done = False
        
        while not done:
            for t in range(T_horizon):
                # Numpy -> Torch
                s_tensor = torch.from_numpy(s).float()
                mask_tensor = torch.from_numpy(mask).bool() # 0/1 -> Bool
                
                prob = model.pi(s_tensor, mask_tensor)
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, done, truncated, info_prime = env.step(a)
                mask_prime = info_prime["mask"]
                
                # 저장 (State, Action, Reward, NextState, Prob, Done, Mask)
                model.put_data((s, a, r, s_prime, prob[a].item(), done, mask))
                
                s = s_prime
                mask = mask_prime
                score += r
                
                if done:
                    break
            
            # T_horizon 마다 혹은 에피소드 끝날 때 학습
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

if __name__ == '__main__':
    main()