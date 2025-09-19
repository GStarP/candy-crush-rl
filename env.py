import gymnasium as gym
from consts import Direction, ItemType, Map, TeamColor, Terrain
import numpy as np
from utils import log_exec_time


class SugarFightEnv(gym.Env):
    def __init__(self, server_exe_path: str):
        super().__init__()

        self.server_exe_path = server_exe_path
        self.observation_space = observation_space_manager.describe()
        self.action_space = action_space_manager.describe()


class ObservationSpaceManager:
    """
    观察空间
    分为多个层，每个层关注不同的信息
    每个层的尺寸为地图“格子”级别的二维数组
    """

    LAYERS = [
        "地图层-普通",
        "地图层-加速",
        "地图层-减速",
        "地图层-障碍",
        "地图层-可摧毁障碍",
        "道具层-炸弹背包",
        "道具层-药水",
        "道具层-飞鞋",
        "炸弹层",
        "占领层",
        "玩家层-队友1",
        "玩家层-队友2",
        "玩家层-敌人1",
        "玩家层-敌人2",
    ]
    TERRAIN_LAYER_BEGIN_INDEX = LAYERS.index("地图层-普通")
    ITEM_LAYER_BEGIN_INDEX = LAYERS.index("道具层-炸弹背包")
    BOMB_LAYER_BEGIN_INDEX = LAYERS.index("炸弹层")
    OCCUPY_LAYER_BEGIN_INDEX = LAYERS.index("占领层")
    PLAYER_LAYER_BEGIN_INDEX = LAYERS.index("玩家层-队友1")

    TERRAIN_INDEX_MAP = {
        Terrain.PLAIN: 0,
        Terrain.SPEED_UP: 1,
        Terrain.SPEED_DOWN: 2,
        Terrain.OBSTACLE: 3,
        Terrain.DESTROYABLE_OBSTACLE: 4,
    }

    ITEM_INDEX_MAP = {
        ItemType.BOMB: 0,
        ItemType.POTION: 1,
        ItemType.SHOE: 2,
    }

    def describe(self):
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.LAYERS), Map.HEIGHT, Map.WIDTH),
            dtype=np.float32,
        )

    @log_exec_time
    def from_game_state(self, game_state: dict):
        observation_space = np.zeros(
            (len(self.LAYERS), Map.HEIGHT, Map.WIDTH), dtype=np.float32
        )

        self_team = game_state["my_player"]["team"]

        # 遍历 map => 确定 所有地图层 和 占领层
        for y in range(Map.HEIGHT):
            for x in range(Map.WIDTH):
                map_item = game_state["map"][y][x]
                observation_space[
                    self.TERRAIN_LAYER_BEGIN_INDEX
                    + self.TERRAIN_INDEX_MAP[map_item["terrain"]],
                    y,
                    x,
                ] = 1.0
                # 己方占领 1.0，敌方占领 -1.0，未被占领 0.0
                observation_space[self.OCCUPY_LAYER_BEGIN_INDEX, y, x] = (
                    1.0
                    if map_item["ownership"] == self_team
                    else (0.0 if map_item["ownership"] == TeamColor.NOOP else -1.0)
                )

        # 遍历 map_items => 确定 所有道具层
        for item in game_state["map_items"]:
            observation_space[
                self.ITEM_LAYER_BEGIN_INDEX + self.ITEM_INDEX_MAP[item["type"]],
                item["position"]["y"],
                item["position"]["x"],
            ] = 1.0

        # 遍历 bombs => 确定 炸弹层
        self._handle_bombs(game_state, observation_space)

        # 根据 my_player 和 other_players => 确定 所有玩家层
        self._handle_players(game_state, observation_space, self_team)

        return observation_space

    @log_exec_time
    def _handle_bombs(self, game_state, observation_space):
        # 构建 y_x => bomb 的 Map
        position_bomb_dict = {}
        for bomb in game_state["bombs"]:
            position_bomb_dict[f"{bomb['position']['y']}_{bomb['position']['x']}"] = (
                bomb
            )

        current_tick = game_state["current_tick"]

        def _compute_explode_value(explode_at: int) -> float:
            """
            归一化爆炸值
            爆炸时间为 2s 即 20 ticks
            """
            v = (explode_at - current_tick) / 20.0
            assert v > 0 and v <= 1
            return v

        def _cascade_trigger_bomb(bomb: dict, expect_explode_at: int):
            """
            级联触发炸弹
            1. 将当前炸弹爆炸范围内的所有格赋值为 _normalize_explode_value(explode_at - current_tick)
            1.1 如果某些格的值更小（不包括初始值 0），则说明其更早就会被炸，无需再赋值
            2. 如果爆炸范围内有其它炸弹，且其爆炸时间更久，则要立即触发该炸弹
            """
            bomb_x, bomb_y = bomb["position"]["x"], bomb["position"]["y"]

            # 如果炸弹所在位置爆炸值不为 0 且更小，则说明已被触发过
            explode_value_at_bomb_pos = observation_space[
                self.BOMB_LAYER_BEGIN_INDEX, bomb_y, bomb_x
            ]
            if (
                explode_value_at_bomb_pos != 0
                and explode_value_at_bomb_pos
                <= _compute_explode_value(expect_explode_at)
            ):
                return

            explode_value = _compute_explode_value(expect_explode_at)

            # 遍历爆炸范围内的所有格子，进行爆炸值的更新
            for dy in range(-bomb["range"], bomb["range"] + 1):
                for dx in range(-bomb["range"], bomb["range"] + 1):
                    # 十字形爆炸
                    if (abs(dx) + abs(dy)) > bomb["range"]:
                        continue

                    explode_x, explode_y = bomb_x + dx, bomb_y + dy
                    if 0 <= explode_x < Map.WIDTH and 0 <= explode_y < Map.HEIGHT:
                        cur_explode_value = observation_space[
                            self.BOMB_LAYER_BEGIN_INDEX, explode_y, explode_x
                        ]
                        if cur_explode_value == 0 or cur_explode_value > explode_value:
                            observation_space[
                                self.BOMB_LAYER_BEGIN_INDEX, explode_y, explode_x
                            ] = explode_value
                        # 如果当前爆炸的位置有炸弹，且其爆炸事件更晚，则提前将其触发
                        postion_key = f"{explode_y}_{explode_x}"
                        if postion_key in position_bomb_dict:
                            target_bomb = position_bomb_dict[postion_key]
                            if target_bomb["explode_at"] > expect_explode_at:
                                _cascade_trigger_bomb(target_bomb, expect_explode_at)

        # 遍历触发所有炸弹，如果某个炸弹被之前的炸弹级联触发，这里的触发应该会被跳过
        for bomb in game_state["bombs"]:
            _cascade_trigger_bomb(bomb, bomb["explode_at"])

    @log_exec_time
    def _handle_players(self, game_state, observation_space, self_team):
        all_players = [None] * 4
        # 确保己方角色处于前两位，且队内按 id 升序
        all_players[1 if game_state["my_player"]["id"] >= 3 else 0] = game_state[
            "my_player"
        ]
        for player in game_state["other_players"]:
            if player["team"] == self_team:
                all_players[0 if all_players[0] is None else 1] = player
            else:
                all_players[2 if player["id"] >= 3 else 3] = player
        # 与玩家重叠的格子，按照重叠部分大小赋值
        # 完全重叠为 1.0，完全不重叠为 0.0
        for i, player in enumerate(all_players):
            player_x_max = player["position"]["x"] + (Map.BLOCK_SIZE // 2)
            player_y_max = player["position"]["y"] + (Map.BLOCK_SIZE // 2)
            x_collapse_grids = _compute_collapse_grid(player["position"]["x"])
            y_collapse_grids = _compute_collapse_grid(player["position"]["y"])
            for grid_x in x_collapse_grids:
                collapse_x = player_x_max - (grid_x * Map.BLOCK_SIZE)
                if collapse_x > Map.BLOCK_SIZE:
                    collapse_x = (2 * Map.BLOCK_SIZE) - collapse_x
                for grid_y in y_collapse_grids:
                    collapse_y = player_y_max - (grid_y * Map.BLOCK_SIZE)
                    if collapse_y > Map.BLOCK_SIZE:
                        collapse_y = (2 * Map.BLOCK_SIZE) - collapse_y

                    collapse_area = collapse_x * collapse_y
                    observation_space[
                        self.PLAYER_LAYER_BEGIN_INDEX + i, grid_y, grid_x
                    ] = collapse_area / (Map.BLOCK_SIZE**2)


observation_space_manager = ObservationSpaceManager()


class ActionSpaceManager:
    # 取 方向 + 是否放置炸弹 的 笛卡尔积 作为离散动作空间
    ACTIONS = [
        {"direction": Direction.NOOP, "is_place_bomb": False},
        {"direction": Direction.UP, "is_place_bomb": False},
        {"direction": Direction.DOWN, "is_place_bomb": False},
        {"direction": Direction.LEFT, "is_place_bomb": False},
        {"direction": Direction.RIGHT, "is_place_bomb": False},
        {"direction": Direction.NOOP, "is_place_bomb": True},
        {"direction": Direction.UP, "is_place_bomb": True},
        {"direction": Direction.DOWN, "is_place_bomb": True},
        {"direction": Direction.LEFT, "is_place_bomb": True},
        {"direction": Direction.RIGHT, "is_place_bomb": True},
    ]

    def describe(self):
        action_discrete_size = len(self.ACTIONS)
        return gym.spaces.MultiDiscrete([action_discrete_size, action_discrete_size])

    def decode_action(self, action_value: int):
        return self.ACTIONS[action_value]

    def encode_action(self, action: dict):
        for i, a in self.ACTIONS:
            if (
                a["direction"] == action["direction"]
                and a["is_place_bomb"] == action["is_place_bomb"]
            ):
                return i
        raise ValueError(f"encode_action_err: action={action}")


action_space_manager = ActionSpaceManager()


def _is_in_grid_center(x_or_y: int) -> bool:
    """
    判断玩家是否在格子的中心（在 纵向/横向 是否只与一个格子重叠）
    """
    return (x_or_y - (Map.BLOCK_SIZE // 2)) % Map.BLOCK_SIZE == 0


def _compute_collapse_grid(x_or_y: int) -> list[int]:
    """
    计算与玩家重叠的格子
    例1：传入 175，此时 x 只与 (x-25)//50=3 即第四格重叠
    例2：传入 160，此时 x 与 (x-25)//50=2 和 (x+25)//50=3 重叠
    例3：传入 200，此时 x 与 (x-25)//50=3 和 (x+25)//50=4 重叠
    """
    if _is_in_grid_center(x_or_y):
        return [(x_or_y - (Map.BLOCK_SIZE // 2)) // Map.BLOCK_SIZE]
    else:
        return [
            (x_or_y - (Map.BLOCK_SIZE // 2)) // Map.BLOCK_SIZE,
            (x_or_y + (Map.BLOCK_SIZE // 2)) // Map.BLOCK_SIZE,
        ]
