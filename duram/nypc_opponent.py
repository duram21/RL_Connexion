# nypc_opponent.py

from typing import List, Optional

from nypc_time_rich_plus_first_guard import Game as NYPCGame, Tile, Cell


class NYPCOpponent:
    """
    NYPC 강화 봇을 '상대 정책(policy)'으로 감싼 래퍼.

    - 우리 PPO 에이전트 입장에서는 이 클래스가 '상대 플레이어'
    - 내부에서 nypc Game 상태를 계속 업데이트한다.
    """

    def __init__(
        self,
        is_first_for_bot: bool,
        bot_tiles_str: List[str],
        agent_tiles_str: List[str],
    ):
        """
        is_first_for_bot : 봇이 선공이면 True, 후공이면 False
        bot_tiles_str    : 봇 손패 (예: ["R1", "G2", ...])
        agent_tiles_str  : 우리 손패 (봇 기준으로는 oppTiles)
        """
        bot_tiles = [Tile.from_string(s) for s in bot_tiles_str]
        agent_tiles = [Tile.from_string(s) for s in agent_tiles_str]

        self.game = NYPCGame(
            myTiles=bot_tiles,
            oppTiles=agent_tiles,
            isFirst=is_first_for_bot,
        )

    def on_agent_move(
        self,
        cell_str: str,
        tile_str: str,
        drawn_tile_str: Optional[str],
        used_time_ms: int = 0,
    ):
        """
        우리 에이전트가 (cell_str, tile_str)을 둔 뒤,
        봇 입장에서는 '상대(OPP)가 둔 수' 이므로 myAction=False로 updateAction.

        drawn_tile_str : 에이전트가 새로 뽑은 타일 (없으면 None)
        """
        cell = Cell.from_string(cell_str)
        tile = Tile.from_string(tile_str)
        get = None if drawn_tile_str is None else Tile.from_string(drawn_tile_str)

        self.game.updateAction(
            myAction=False,          # 봇 입장에서는 상대가 둔 수
            action=(cell, tile),
            get=get,
            _usedTime=used_time_ms,
        )

    def choose_move(
        self,
        my_time_ms: int = 1000,
        opp_time_ms: int = 1000,
    ) -> tuple[str, str]:
        """
        봇이 현재 상태에서 둘 수를 선택.
        TIME 명령에 대응.

        반환: (cell_str, tile_str)
        """
        cell, tile = self.game.calculateMove(my_time_ms, opp_time_ms)
        return str(cell), str(tile)

    def on_bot_move_applied(
        self,
        cell_str: str,
        tile_str: str,
        drawn_tile_str: Optional[str],
    ):
        """
        우리가 환경 보드에 봇의 수를 반영한 뒤,
        봇 내부 Game 상태도 동기화.

        봇 입장에서는 내가 둔 수 → myAction=True
        """
        cell = Cell.from_string(cell_str)
        tile = Tile.from_string(tile_str)
        get = None if drawn_tile_str is None else Tile.from_string(drawn_tile_str)

        self.game.updateAction(
            myAction=True,
            action=(cell, tile),
            get=get,
            _usedTime=None,
        )
