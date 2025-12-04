# nypc_time_rich_blocked_inner.py
# Python 3.9+ : DSU undo + Beam + Opp-topK quiescence + Blocked term + Inner32 bias + TimeGuard
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import sys, os, struct, time
from enum import Enum

# ───────────────────── 베이스 파라미터 ─────────────────────
LAMBDA_BASE = 0.95   # 즉시 상대 득점 패널티 계수(lambda)
MU_BASE     = 0.60   # 퀴에센스(상대 최악 라인) 패널티 계수(mu)
NU_DEF      = 0.40   # 2-스텝 상대 재응수 가중(NU)

# 기본 K (시간 충분할 때 동적으로 더 키움)
K_MY_DEF    = 14     # 내 후보 빔 폭
K_OPP_DEF   = 10     # 상대 후보 빔 폭
K_RE_DEF    = 8      # 내 재응수 빔 폭
K_OPP2_DEF  = 8      # 상대 재재응수 빔 폭

# 퀴에센스 임계치
THRESH_Q1   = 8
THRESH_Q2   = 10
GAMMA_Q     = 0.70   # 내 재응수 차감 계수

# 차단 이득 가중치(내가 지금 둬서 상대 한 방을 줄이는 만큼 보상)
ETA_BLOCK   = 0.80

# 중앙/차수/시너지 보너스
ALPHA_SYNERGY = 1.00
BETA_LIBERTY  = 0.50
CENTER_W_EARLY= 0.32
CENTER_W_MID  = 0.22
CENTER_W_LATE = 0.05
DEG_W         = 0.08

# 셀 바이어스(파일/상수)
USE_CELL_BIAS        = True
CELL_BIAS_DEFAULT    = [0.0]*64

# 안쪽 32칸(중앙 4×4×±) 선호 가중치
INNER_W       = 0.25   # 안쪽이면 +INNER_W, 바깥이면 0
# 외곽 기피까지 주고 싶으면 아래 OFF를 True로 바꾸고 OUTER_W < 0로
OUTER_PENALTY = True
OUTER_W       = -0.1

# 시간 가드(여유에 따라 동적 폭 + 계산 예산)
TIME_BUDGET_MAX_MS   = 1500
TIME_BUDGET_MIN_MS   = 80
TIME_BUDGET_RATIO    = 0.75  # 내 남은 시간의 75%만 사용

# ───────────────────── 보드/타일 ─────────────────────
class Column(Enum): A="a"; B="b"; C="c"; D="d"; E="e"; F="f"
class Row(Enum):    _1="1"; _2="2"; _3="3"; _4="4"; _5="5"; _6="6"
class Sign(Enum):   MINUS="-"; PLUS="+"
class Color(Enum):  R="R"; G="G"; B="B"; Y="Y"
class Symbol(Enum): _1="1"; _2="2"; _3="3"; _4="4"

FORBIDDEN = {"a1-","a4-","c3+","c6+","d1-","d4-","f3+","f6+"}

@dataclass(frozen=True)
class Cell:
    col: Column; row: Row; sign: Sign
    def __str__(self)->str: return f"{self.col.value}{self.row.value}{self.sign.value}"
    def is_valid(self)->bool: return str(self) not in FORBIDDEN
    def is_adjacent(self, other:"Cell")->bool:
        dr = ord(other.row.value) - ord(self.row.value)
        dc = ord(other.col.value) - ord(self.col.value)
        if self.sign==Sign.MINUS and other.sign==Sign.PLUS:
            return (dr==0 and dc==0) or (dr==0 and dc==-1) or (dr==-1 and dc==0)
        if self.sign==Sign.PLUS and other.sign==Sign.MINUS:
            return (dr==0 and dc==0) or (dr==0 and dc== 1) or (dr== 1 and dc==0)
        return False
    @staticmethod
    def all_cells()->List["Cell"]:
        out=[]
        for c in Column:
            for r in Row:
                for s in Sign:
                    cell=Cell(c,r,s)
                    if cell.is_valid(): out.append(cell)
        return out
    @staticmethod
    def from_string(s:str)->"Cell":
        assert len(s)==3
        cell = Cell(Column(s[0]), Row(s[1]), Sign(s[2]))
        assert cell.is_valid()
        return cell

@dataclass(frozen=True)
class Tile:
    color: Color; symbol: Symbol
    def __str__(self)->str: return f"{self.color.value}{self.symbol.value}"
    @staticmethod
    def from_string(s:str)->"Tile":
        assert len(s)==2
        return Tile(Color(s[0]), Symbol(s[1]))

ALL_CELLS: List[Cell] = Cell.all_cells()
V = len(ALL_CELLS)  # 64
KEY2IDX: Dict[str,int] = {str(ALL_CELLS[i]): i for i in range(V)}

# 인접/중앙성/차수
ADJ: List[Tuple[int,...]] = [tuple() for _ in range(V)]
_tmp=[[] for _ in range(V)]
for i,a in enumerate(ALL_CELLS):
    for j,b in enumerate(ALL_CELLS):
        if i!=j and a.is_adjacent(b): _tmp[i].append(j)
ADJ=[tuple(x) for x in _tmp]
DEG=[len(ADJ[i]) for i in range(V)]
def center_score(i:int)->float:
    key=str(ALL_CELLS[i]); c=ord(key[0])-ord('a'); r=int(key[1])-1
    # 맨해튼 기반(가운데 최대)
    return - (abs(c-2.5)+abs(r-2.5))
CENTER=[center_score(i) for i in range(V)]

# ── 안쪽 32칸 마스크(칼럼 b..e, 로우 2..5) ──
def is_inner32(i:int)->bool:
    key=str(ALL_CELLS[i])
    col = ord(key[0]) - ord('a')   # 0..5
    row = int(key[1]) - 1          # 0..5
    # 중앙 4×4 영역: 컬럼 1..4, 로우 1..4 (b..e, 2..5)
    return (1 <= col <= 4) and (1 <= row <= 4)
INNER32 = [1.0 if is_inner32(i) else 0.0 for i in range(V)]
OUTER32 = [0.0 if INNER32[i] else 1.0 for i in range(V)]

# 타일 ID 변환
COLOR2IDX = {Color.R:0, Color.G:1, Color.B:2, Color.Y:3}
SYMB2IDX  = {Symbol._1:0, Symbol._2:1, Symbol._3:2, Symbol._4:3}
def tile_id(t:Tile)->int: return COLOR2IDX[t.color]*4 + SYMB2IDX[t.symbol]
def color_of(tid:int)->int:  return tid//4
def symbol_of(tid:int)->int: return tid%4

# ───────────────────── DSU + Undo ─────────────────────
class DSU:
    __slots__=("p","sz","sum_sq","n","hist")
    def __init__(self,n:int):
        self.n=n
        self.p=list(range(n))
        self.sz=[0]*n
        self.sum_sq=0
        self.hist=[]
    def find(self,x:int)->int:
        while self.p[x]!=x: x=self.p[x]
        return x
    def add_single(self,x:int):
        if self.sz[x]==0:
            self.hist.append(('a',x,self.sum_sq))
            self.p[x]=x; self.sz[x]=1; self.sum_sq+=1
    def unite(self,a:int,b:int):
        a=self.find(a); b=self.find(b)
        if a==b: return
        if self.sz[a]<self.sz[b]: a,b=b,a
        A=self.sz[a]; B=self.sz[b]; prev=self.sum_sq
        self.sum_sq += (A+B)*(A+B) - A*A - B*B
        self.p[b]=a; self.sz[a]=A+B
        self.hist.append(('u', b, a, A, B, prev))
    def snapshot(self)->int: return len(self.hist)
    def rollback(self, snap:int):
        while len(self.hist)>snap:
            rec=self.hist.pop()
            if rec[0]=='a':
                _,x,prev=rec; self.sz[x]=0; self.sum_sq=prev
            else:
                _,b_root,a_root,A,B,prev=rec
                self.p[b_root]=b_root; self.sz[a_root]=A; self.sum_sq=prev

def make_uf8()->List[DSU]: return [DSU(V) for _ in range(8)]

def delta_attr(occ:List[int], dsu:DSU, v:int, is_same_pred)->int:
    """
    v에 새 돌을 추가했을 때 (해당 속성 DSU 기준) 증가하는 점수 = 새 컴포넌트(이웃 동일 속성과 병합)의
    (합친 크기^2 - 기존 각 성분^2)의 합. (Undo-DSU의 sum_sq를 직접 쓰지 않기 위해 지역 계산)
    """
    seen=set(); sum_sz=0; sum_sq=0
    for nb in ADJ[v]:
        tid=occ[nb]
        if tid==-1: continue
        if not is_same_pred(tid): continue
        r=dsu.find(nb)
        if r in seen: continue
        seen.add(r)
        s=dsu.sz[r]
        sum_sz+=s; sum_sq+=s*s
    S=1+sum_sz
    return S*S - sum_sq

# ───────────────────── Submission-2014.bin 로딩 ─────────────────────
# Submission-2014.bin: [double λ][double bias*64]
def load_data_bin():
    lam=LAMBDA_BASE; bias=CELL_BIAS_DEFAULT[:]
    path="Submission-2014.bin"
    if not os.path.exists(path): return lam,bias
    try:
        with open(path,"rb") as f: buf=f.read()
        if len(buf) >= 8 + 64*8:
            lam = struct.unpack_from("<d", buf, 0)[0]
            bias = list(struct.unpack_from("<64d", buf, 8))
    except Exception:
        pass
    return lam,bias

# ───────────────────── 유틸 ─────────────────────
class TimeGuard:
    def __init__(self,budget_ms:int):
        self.start=time.perf_counter()
        self.budget=budget_ms/1000.0
    def over(self)->bool:
        return (time.perf_counter()-self.start)>self.budget
    def elapsed_ms(self)->int:
        return int((time.perf_counter()-self.start)*1000)

def multiset_minus_one(ids: List[int], x: int) -> List[int]:
    removed=False; out=[]
    for v in ids:
        if not removed and v==x:
            removed=True; continue
        out.append(v)
    return out

# ───────────────────── 엔진 ─────────────────────
class Game:
    def __init__(self, myTiles:List[Tile], oppTiles:List[Tile], isFirst:bool):
        self.isFirst=isFirst
        self.myTiles=list(myTiles)
        self.oppTiles=list(oppTiles)

        # 보드 점유(타일ID), -1 = 빈칸
        self.occ=[-1]*V

        # 색 4 + 문양 4 = 8개의 DSU (undo 가능)
        self.uf8=make_uf8()

        # 롤백용 히스토리
        self.occ_hist=[]

        # Submission-2014.bin 로드(옵션)
        lam,bias=load_data_bin()
        self.lambda_base=lam
        self.cellBias = bias if (USE_CELL_BIAS and len(bias)==64) else CELL_BIAS_DEFAULT[:]

    # 프런티어(인접 빈칸 + 중앙 일부)
    def frontier_empties(self)->List[int]:
        occ=self.occ; empties=[]; has_any=False
        for i in range(V):
            if occ[i]!=-1:
                has_any=True
                for nb in ADJ[i]:
                    if occ[nb]==-1: empties.append(nb)
        if not has_any:
            idxs=[i for i in range(V) if occ[i]==-1]
            idxs.sort(key=lambda x: (CENTER[x]+0.2*DEG[x]), reverse=True)
            return idxs[:24]
        if not empties:
            return [i for i in range(V) if occ[i]==-1]
        extras = sorted([i for i in range(V) if occ[i]==-1],
                        key=lambda x: CENTER[x]+0.2*DEG[x], reverse=True)[:8]
        s=set(empties); s.update(extras)
        return list(s)

    # 적용/스냅/롤백
    def apply(self, v:int, tid:int):
        self.occ_hist.append(v); self.occ[v]=tid
        ci=color_of(tid); si=4+symbol_of(tid)
        self.uf8[ci].add_single(v)
        for nb in ADJ[v]:
            t=self.occ[nb]
            if t!=-1 and color_of(t)==ci: self.uf8[ci].unite(v, nb)
        self.uf8[si].add_single(v)
        for nb in ADJ[v]:
            t=self.occ[nb]
            if t!=-1 and (4+symbol_of(t))==si: self.uf8[si].unite(v, nb)

    def snapshot(self):
        return (len(self.occ_hist),
                self.uf8[0].snapshot(), self.uf8[1].snapshot(), self.uf8[2].snapshot(), self.uf8[3].snapshot(),
                self.uf8[4].snapshot(), self.uf8[5].snapshot(), self.uf8[6].snapshot(), self.uf8[7].snapshot())

    def rollback(self, snap):
        occ_len=snap[0]
        while len(self.occ_hist)>occ_len:
            v=self.occ_hist.pop()
            self.occ[v]=-1
        for i in range(8):
            self.uf8[i].rollback(snap[1+i])

    # Δ(내/상대) — 현재 보드에서 (v,tid)를 둘 때 각 관점 즉시 득점
    def eval_delta_both(self, v:int, tid:int)->Tuple[int,int]:
        me_first=self.isFirst
        if me_first:
            d_me  = delta_attr(self.occ, self.uf8[4+symbol_of(tid)], v, lambda nb_tid,tid=tid: symbol_of(nb_tid)==symbol_of(tid))
            d_opp = delta_attr(self.occ, self.uf8[color_of(tid)],       v, lambda nb_tid,tid=tid: color_of(nb_tid)==color_of(tid))
        else:
            d_me  = delta_attr(self.occ, self.uf8[color_of(tid)],       v, lambda nb_tid,tid=tid: color_of(nb_tid)==color_of(tid))
            d_opp = delta_attr(self.occ, self.uf8[4+symbol_of(tid)],    v, lambda nb_tid,tid=tid: symbol_of(nb_tid)==symbol_of(tid))
        return d_me, d_opp

    # 손패 시너지 & 자유도
    def synergy_and_liberty_bonus(self, v:int, tid:int)->float:
        me_first=self.isFirst
        if me_first:
            sym = symbol_of(tid)
            same_cnt = sum(1 for t in self.myTiles if (symbol_of(tile_id(t))==sym))
        else:
            col = color_of(tid)
            same_cnt = sum(1 for t in self.myTiles if (color_of(tile_id(t))==col))
        liberties = sum(1 for nb in ADJ[v] if self.occ[nb]==-1)
        return ALPHA_SYNERGY*same_cnt + BETA_LIBERTY*liberties

    # 동적 파라미터(턴/시간 기반)
    def dynamic_params(self, myTime:int):
        placed = sum(1 for x in self.occ if x!=-1)
        # 초/중/후반
        if placed < 12:
            lam = min(self.lambda_base, 0.85)
            mu  = 0.45
            center_w = CENTER_W_EARLY
            k_my,k_opp,k_re,k_opp2 = K_MY_DEF+2, K_OPP_DEF+2, K_RE_DEF+1, K_OPP2_DEF+1
        elif placed < 36:
            lam = self.lambda_base
            mu  = MU_BASE
            center_w = CENTER_W_MID
            k_my,k_opp,k_re,k_opp2 = K_MY_DEF+1, K_OPP_DEF+1, K_RE_DEF, K_OPP2_DEF
        else:
            lam = min(1.0, self.lambda_base+0.05)
            mu  = 0.60
            center_w = CENTER_W_LATE
            k_my,k_opp,k_re,k_opp2 = K_MY_DEF, K_OPP_DEF+1, K_RE_DEF, K_OPP2_DEF

        # 시간 널널: 폭 키움
        if myTime >= 5000:
            k_my   += 4; k_opp  += 3; k_re   += 2; k_opp2 += 2
        elif myTime >= 2500:
            k_my   += 2; k_opp  += 1

        # 시간 부족: 축소
        if myTime < 400:
            k_my,k_opp,k_re,k_opp2 = max(8,k_my-4), max(6,k_opp-3), max(3,k_re-2), max(4,k_opp2-2)
        elif myTime < 1500:
            k_my,k_opp,k_re,k_opp2 = max(10,k_my-2), max(8,k_opp-1), max(4,k_re), max(5,k_opp2-1)

        return lam, mu, center_w, k_my, k_opp, k_re, k_opp2

    # 상대 top-K 응수 리스트 (현재 상태에서의 1차값)
    def opp_top_moves_now(self, opp_ids:List[int], lam:float, k_opp:int)->List[Tuple[float,int,int]]:
        empties=[i for i in range(V) if self.occ[i]==-1]
        if not empties or not opp_ids: return []
        opp_is_first = (not self.isFirst)
        prelim=[]
        for u in empties:
            for otid in opp_ids:
                d_opp = delta_attr(
                    self.occ, self.uf8[4+symbol_of(otid)] if opp_is_first else self.uf8[color_of(otid)],
                    u,
                    (lambda nb_tid, otid=otid: (symbol_of(nb_tid)==symbol_of(otid)) if opp_is_first else (color_of(nb_tid)==color_of(otid)))
                )
                if d_opp<=0: continue
                d_me_from_opp = delta_attr(
                    self.occ, self.uf8[color_of(otid)] if opp_is_first else self.uf8[4+symbol_of(otid)],
                    u,
                    (lambda nb_tid, otid=otid: (color_of(nb_tid)==color_of(otid)) if opp_is_first else (symbol_of(nb_tid)==symbol_of(otid)))
                )
                # 중앙/안쪽 보정(상대에게는 가산 → 우리가 더 경계)
                center_term = CENTER[u]
                inner_term  = INNER32[u]*INNER_W + (OUTER32[u]*OUTER_W if OUTER_PENALTY else 0.0)
                val = d_opp - lam*d_me_from_opp + 1e-6*(center_term) + inner_term*0.1
                prelim.append((val,u,otid))
        prelim.sort(reverse=True, key=lambda x:x[0])
        return prelim[:min(k_opp,len(prelim))]

    # 상대 '다음 한 수 최대 득점(스칼라)'
    def opp_best_scalar_now(self, opp_ids:List[int], lam:float, k_opp:int) -> float:
        lst = self.opp_top_moves_now(opp_ids, lam, k_opp)
        return lst[0][0] if lst else 0.0

    # 내 재응수: 최선 1수와 그 가치(현재 상태)
    def my_best_reply_move_now(self, my_ids:List[int], lam:float, k_my:int)->Tuple[float, Optional[Tuple[int,int]]]:
        empties=[i for i in range(V) if self.occ[i]==-1]
        if not empties or not my_ids: return 0.0, None
        me_first=self.isFirst
        prelim=[]
        for v in empties:
            for tid in my_ids:
                d_me = delta_attr(
                    self.occ, self.uf8[4+symbol_of(tid)] if me_first else self.uf8[color_of(tid)],
                    v,
                    (lambda nb_tid, tid=tid: (symbol_of(nb_tid)==symbol_of(tid)) if me_first else (color_of(nb_tid)==color_of(tid)))
                )
                if d_me>0: prelim.append((d_me,v,tid))
        if not prelim: return 0.0, None
        prelim.sort(reverse=True, key=lambda x:x[0])
        prelim=prelim[:min(k_my,len(prelim))]

        best=-1e9; best_act=None
        for _,v,tid in prelim:
            d_me = delta_attr(
                self.occ, self.uf8[4+symbol_of(tid)] if me_first else self.uf8[color_of(tid)],
                v,
                (lambda nb_tid, tid=tid: (symbol_of(nb_tid)==symbol_of(tid)) if me_first else (color_of(nb_tid)==color_of(tid)))
            )
            d_opp = delta_attr(
                self.occ, self.uf8[color_of(tid)] if me_first else self.uf8[4+symbol_of(tid)],
                v,
                (lambda nb_tid, tid=tid: (color_of(nb_tid)==color_of(tid)) if me_first else (symbol_of(nb_tid)==symbol_of(tid)))
            )
            center_term = CENTER[v]
            inner_term  = INNER32[v]*INNER_W + (OUTER32[v]*OUTER_W if OUTER_PENALTY else 0.0)
            val = d_me - lam*d_opp + 1e-6*center_term + inner_term*0.1
            if val>best:
                best=val; best_act=(v,tid)
        if best<0: best=0.0
        return best,best_act

    # 메인 선택
    def calculateMove(self, myTime:int, oppTime:int)->Tuple[Cell, Tile]:
        budget = min(int(myTime*TIME_BUDGET_RATIO), TIME_BUDGET_MAX_MS)
        budget = max(TIME_BUDGET_MIN_MS, budget)
        tg = TimeGuard(budget)

        lam, mu, center_w, k_my, k_opp, k_re, k_opp2 = self.dynamic_params(myTime)
        cellBias=self.cellBias

        # 후보 생성(프런티어 우선)
        empties=self.frontier_empties()
        if not empties: empties=[i for i in range(V) if self.occ[i]==-1]
        my_ids_all=[tile_id(t) for t in self.myTiles]
        cands=[(v,tid) for v in empties for tid in my_ids_all]
        assert cands, "No legal moves"

        # 1차: 빠른 스코어로 상위 k_my 빔
        scored=[]
        for v,tid in cands:
            d_me,d_opp = self.eval_delta_both(v, tid)
            extra = self.synergy_and_liberty_bonus(v, tid)
            inner_term = INNER32[v]*INNER_W + (OUTER32[v]*OUTER_W if OUTER_PENALTY else 0.0)
            base = d_me - lam*d_opp + extra + (cellBias[v] if USE_CELL_BIAS else 0.0)
            base += center_w*CENTER[v] + DEG_W*DEG[v] + inner_term
            scored.append((base,v,tid,d_me,d_opp))
        scored.sort(reverse=True, key=lambda x:x[0])
        scored=scored[:min(k_my,len(scored))]

        best_val=-1e100; best_move=None
        opp_ids_all=[tile_id(t) for t in self.oppTiles]

        # ★ 현재 보드에서 상대의 '다음 한 수' 최대 득점(스칼라)
        opp_best_before_scalar = self.opp_best_scalar_now(opp_ids_all, lam, k_opp)

        for base,v,tid,d_me,d_opp in scored:
            if tg.over(): break
            snap=self.snapshot()
            self.apply(v, tid)

            # ★ 내가 둔 뒤 상태에서 상대의 '다음 한 수' 최대 득점(스칼라)
            opp_best_after_scalar = self.opp_best_scalar_now(opp_ids_all, lam, k_opp)
            blocked = max(0.0, opp_best_before_scalar - opp_best_after_scalar)

            # 상대 top-K 응수들 모아서 각각 평가 → 최악 라인으로 패널티
            worst_eff_opp = 0.0
            opp_list = self.opp_top_moves_now(opp_ids_all, lam, k_opp)
            if opp_list:
                for first_val,u,otid in opp_list:
                    if tg.over(): break
                    snap2=self.snapshot()
                    self.apply(u, otid)

                    my_ids_after = multiset_minus_one(my_ids_all, tid)
                    my_reply_val, my_reply_act = self.my_best_reply_move_now(my_ids_after, lam, k_re)

                    eff_opp = max(0.0, first_val - GAMMA_Q*my_reply_val)

                    # 2-step: 위험하면 상대 재재응수(탑1)까지 반영
                    if eff_opp >= THRESH_Q2 and my_reply_act is not None and not tg.over():
                        vr,tr = my_reply_act
                        snap3=self.snapshot()
                        self.apply(vr, tr)

                        opp_ids_after = multiset_minus_one(opp_ids_all, otid)
                        opp2_list = self.opp_top_moves_now(opp_ids_after, lam, k_opp2)
                        if opp2_list:
                            eff_opp += NU_DEF * opp2_list[0][0]  # top1만
                        self.rollback(snap3)

                    if eff_opp > worst_eff_opp:
                        worst_eff_opp = eff_opp

                    self.rollback(snap2)

            # 최종 가치: 즉시 이득 - 퀴에센스 패널티 + 차단 이득 + 정적 보너스
            inner_term = INNER32[v]*INNER_W + (OUTER32[v]*OUTER_W if OUTER_PENALTY else 0.0)
            val = (d_me - lam*d_opp) \
                  - mu*worst_eff_opp \
                  + ETA_BLOCK*blocked \
                  + (cellBias[v] if USE_CELL_BIAS else 0.0) \
                  + center_w*CENTER[v] + DEG_W*DEG[v] + inner_term + 1e-6*(CENTER[v]+DEG[v])

            if val>best_val:
                best_val=val; best_move=(v,tid)

            self.rollback(snap)

        if best_move is None:
            v,tid=scored[0][1], scored[0][2]
        else:
            v,tid=best_move

        cell=ALL_CELLS[v]
        t=Tile(list(Color)[color_of(tid)], list(Symbol)[symbol_of(tid)])
        return cell,t

    # 인터랙션 업데이트(심판 커맨드 반영)
    def updateAction(self, myAction:bool, action:Tuple[Cell, Tile], get:Optional[Tile], _usedTime:Optional[int]):
        cell,tile=action
        v=KEY2IDX[str(cell)]
        tid=tile_id(tile)
        self.apply(v, tid)
        if myAction:
            for i,x in enumerate(self.myTiles):
                if x==tile: self.myTiles.pop(i); break
            if get is not None: self.myTiles.append(get)
        else:
            for i,x in enumerate(self.oppTiles):
                if x==tile: self.oppTiles.pop(i); break
            if get is not None: self.oppTiles.append(get)

# ───────────────────── I/O 루프 ─────────────────────
def main():
    game: Optional[Game]=None
    isFirst: Optional[bool]=None
    lastMove: Optional[Tuple[Cell, Tile]]=None

    while True:
        try:
            line=sys.stdin.readline()
            if not line: break
            line=line.strip()
            if not line: continue
            parts=line.split()
            cmd,args=parts[0],parts[1:]

            if cmd=="READY":
                isFirst=(args[0]=="FIRST")
                print("OK", flush=True); continue

            if cmd=="INIT":
                myTiles  = [Tile.from_string(s) for s in args[:5]]
                oppTiles = [Tile.from_string(s) for s in args[5:10]]
                assert isFirst is not None
                game=Game(myTiles, oppTiles, isFirst); continue

            if cmd=="TIME":
                myTime,oppTime=int(args[0]),int(args[1])
                assert game is not None
                lastMove=game.calculateMove(myTime, oppTime)
                c,t=lastMove
                print(f"PUT {c} {t}", flush=True); continue

            if cmd=="GET":
                assert game is not None and lastMove is not None
                get = None if args[0]=="X0" else Tile.from_string(args[0])
                game.updateAction(True, lastMove, get, None); continue

            if cmd=="OPP":
                assert game is not None
                cell=Cell.from_string(args[0]); tile=Tile.from_string(args[1])
                get=None if args[2]=="X0" else Tile.from_string(args[2])
                oppTime=int(args[3])
                game.updateAction(False, (cell,tile), get, oppTime); continue

            if cmd=="FINISH":
                break

            print(f"Invalid command: {cmd}", file=sys.stderr); sys.exit(1)

        except EOFError:
            break
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__=="__main__":
    main()
