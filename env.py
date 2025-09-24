import logging
import gymnasium as gym
from consts import Direction, ItemType, Map, PlayerState, TeamColor, Terrain
import numpy as np
from utils import log_exec_time
from typing import Any
from game_manager import GameManager


class SugarFightEnv(gym.Env):
    def __init__(self, server_exe_path: str):
        super().__init__()

        self.server_exe_path = server_exe_path
        self.observation_space = observation_space_manager.describe()
        self.action_space = action_space_manager.describe()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        if self._game_manager is not None:
            try:
                self._game_manager.__exit__()
            except Exception:
                logging.exception("game_manager_exit_err")
            finally:
                self._game_manager = None

        self._game_manager = GameManager(self.server_exe_path, id=0)
        self._game_manager.__enter__()

        first_game_state, tick = self._game_manager.start_game()
        assert tick == 1

        obs = observation_space_manager.from_game_state(first_game_state)
        info = {
            "game_state": first_game_state,
        }

        self._prev_game_state = None
        self._cur_game_state = first_game_state
        self._cur_tick = tick

        return obs, info

    def step(self, action):
        commands = [action_space_manager.encode_command(v) for v in action]
        self._game_manager.dispatch_commands(commands, self._cur_tick)

        next_game_state, tick, game_result = self._game_manager.wait_for_game_state(
            self._cur_tick + 1
        )
        # ! 时序不一致直接抛出异常，前期先以排错为主
        assert tick == self._cur_tick + 1

        next_obs = observation_space_manager.from_game_state(next_game_state)
        reward, reward_detail = reward_manager.compute_reward(
            self._cur_game_state, next_game_state, game_result
        )

        # done = game_result is not None
        # TODO 先用 tick 自行判断是否结束
        done = tick >= GameManager.GAME_DURATION

        info = {
            "commands": commands,
            "next_game_state": next_game_state,
            "reward_detail": reward_detail,
        }

        self._prev_game_state = self._cur_game_state
        self._cur_game_state = next_game_state
        self._cur_tick = tick

        return next_obs, reward, done, False, info


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
        Terrain.NOOP: 3,
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
        self._handle_players(game_state, observation_space)

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

        def on_explode(grid_x, grid_y, tick):
            observation_space[self.BOMB_LAYER_BEGIN_INDEX, grid_y, grid_x] = (
                _compute_explode_value(tick)
            )

        _dfs_bomb(game_state, on_explode)

    @log_exec_time
    def _handle_players(self, game_state, observation_space):
        all_players = _get_sorted_all_players(game_state)
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

    def render(self, observation_space):
        result = ""

        value_fmt = "{:.2f}"
        cell_width = 7

        layer_num, H, W = observation_space.shape
        row_idx_width = max(2, len(str(H - 1)))

        for i in range(layer_num):
            result += f"=== {self.LAYERS[i]} ===\n"
            for y in range(H):
                row_img = " ".join(
                    value_fmt.format(float(observation_space[i, y, x])).rjust(
                        cell_width
                    )
                    for x in range(W)
                )
                result += f"{str(y).rjust(row_idx_width)}: {row_img}\n"

            result += "\n"

        return result


observation_space_manager = ObservationSpaceManager()


class ActionSpaceManager:
    # 取 方向 + 是否放置炸弹 的 笛卡尔积 作为离散动作空间
    COMMANDS = [
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
        action_discrete_size = len(self.COMMANDS)
        # 将两个角色的动作组合成为 多头动作，这样既能学习到两个角色的协作，又能一次推理输出两个角色的动作
        return gym.spaces.MultiDiscrete([action_discrete_size, action_discrete_size])

    def encode_command(self, action: int):
        return self.COMMANDS[action]

    def decode_command(self, command: dict):
        for i, c in enumerate(self.COMMANDS):
            if (
                c["direction"] == command["direction"]
                and c["is_place_bomb"] == command["is_place_bomb"]
            ):
                return i
        raise ValueError(f"encode_action_err: action={command}")


action_space_manager = ActionSpaceManager()


class RewardManager:
    W_OCCUPY = 15 / Map.WIDTH / Map.HEIGHT
    W_STUN = -1
    W_DESTROY_OBSTACLE = 12 / Map.WIDTH / Map.HEIGHT
    W_WIN = 4
    W_LOSE = -4

    @log_exec_time
    def compute_reward(
        self, prev_game_state, cur_game_state, game_result: bool | None = None
    ):
        if prev_game_state is None:
            return 0

        # 结果奖励
        result_reward = 0
        if game_result is not None:
            result_reward = self.W_WIN if game_result else self.W_LOSE

        self_team = cur_game_state["my_player"]["team"]

        # 占领奖励
        prev_occupied_grids = _count_occupied_grids(prev_game_state, self_team)
        cur_occupied_grids = _count_occupied_grids(cur_game_state, self_team)
        occupy_reward = (cur_occupied_grids - prev_occupied_grids) * self.W_OCCUPY

        # 眩晕惩罚
        stun_reward = 0
        prev_all_players = _get_sorted_all_players(prev_game_state)
        cur_all_players = _get_sorted_all_players(cur_game_state)
        for i in range(2):
            if (
                prev_all_players[i]["status"] == PlayerState.GOOD
                and cur_all_players[i]["status"] == PlayerState.BAD
            ):
                stun_reward += self.W_STUN

        # 破坏障碍物奖励
        destroy_obstacle_reward = self._compute_destroy_obstacle_reward(
            prev_game_state, cur_game_state, self_team
        )

        total_reward = (
            result_reward + occupy_reward + stun_reward + destroy_obstacle_reward
        )

        return total_reward, {
            "total_reward": total_reward,
            "result_reward": result_reward,
            "occupy_reward": occupy_reward,
            "stun_reward": stun_reward,
            "destroy_obstacle_reward": destroy_obstacle_reward,
        }

    @log_exec_time
    def _compute_destroy_obstacle_reward(
        self, prev_game_state, cur_game_state, self_team
    ) -> float:
        destroy_obstacle_reward = 0
        # 遍历上一帧的己方的所有炸弹，如果其爆炸范围里有“可破坏障碍物”且这一帧变成平地了，就认为破坏了障碍物
        # TODO 暂不考虑级联爆炸
        for bomb in prev_game_state["bombs"]:
            if bomb["team"] == self_team:
                bomb_x, bomb_y = bomb["position"]["x"], bomb["position"]["y"]
                directions = [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                ]
                for dx, dy in directions:
                    for i in range(1, bomb["range"] + 1):
                        grid_x, grid_y = bomb_x + dx * i, bomb_y + dy * i
                        if not (0 <= grid_x < Map.WIDTH and 0 <= grid_y < Map.HEIGHT):
                            break
                        terrain = prev_game_state["map"][grid_y][grid_x]["terrain"]
                        if terrain == Terrain.OBSTACLE:
                            break
                        if terrain == Terrain.DESTROYABLE_OBSTACLE:
                            if (
                                cur_game_state["map"][grid_y][grid_x]["terrain"]
                                == Terrain.PLAIN
                            ):
                                destroy_obstacle_reward += self.W_DESTROY_OBSTACLE
                            break

        return destroy_obstacle_reward


reward_manager = RewardManager()


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


def _count_occupied_grids(game_state, self_team) -> int:
    """
    统计当前游戏状态下，当前队伍占领的格子数量
    """
    count = 0
    for row in game_state["map"]:
        for grid in row:
            if grid["ownership"] == self_team:
                count += 1
    return count


def _get_sorted_all_players(game_state) -> list[dict]:
    """
    获取当前游戏状态下的所有玩家，先己方后敌方，队内按照 id 升序
    """
    self_team = game_state["my_player"]["team"]
    all_players = [None] * 4

    all_players[(game_state["my_player"]["id"] - 1) // 2] = game_state["my_player"]

    for player in game_state["other_players"]:
        if player["team"] == self_team:
            all_players[(player["id"] - 1) // 2] = player
        else:
            all_players[2 + ((player["id"] - 1) // 2)] = player
    return all_players


@log_exec_time
def _dfs_bomb(game_state, on_explode):
    """
    对所有爆炸可能影响到的位置执行 on_explode(x, y, tick)
    考虑 障碍阻挡 和 级联爆炸；若某格可能在多个时刻被炸，只在最早的 tick 触发一次 on_explode
    """
    position_bomb_dict = {}
    for bomb in game_state["bombs"]:
        position_bomb_dict[(bomb["position"]["x"], bomb["position"]["y"])] = bomb
    # 记录 (x, y) -> 最早爆炸时间
    affected_grid_tick_dict = {}

    def _record_affect(grid_pos: tuple[int, int], tick: int):
        if (
            grid_pos not in affected_grid_tick_dict
            or affected_grid_tick_dict[grid_pos] > tick
        ):
            affected_grid_tick_dict[grid_pos] = tick

    def _propagate_from_bomb(bomb: dict, tick: int):
        bomb_x, bomb_y = bomb["position"]["x"], bomb["position"]["y"]
        # 若炸弹在更早的时间已经爆炸，则无需再次处理
        if (
            bomb_x,
            bomb_y,
        ) in affected_grid_tick_dict and tick >= affected_grid_tick_dict[
            (bomb_x, bomb_y)
        ]:
            return
        # 炸弹所在格被炸
        _record_affect((bomb_x, bomb_y), tick)
        # 十字传播
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]
        for dx, dy in directions:
            for i in range(1, bomb["range"] + 1):
                grid_x, grid_y = bomb_x + dx * i, bomb_y + dy * i
                # 超出地图范围
                if not (0 <= grid_x < Map.WIDTH and 0 <= grid_y < Map.HEIGHT):
                    break
                terrain = game_state["map"][grid_y][grid_x]["terrain"]
                # 遇到障碍，不被炸，且被阻挡
                if terrain == Terrain.OBSTACLE:
                    break
                # 遇到可摧毁障碍，被炸，但被阻挡
                if terrain == Terrain.DESTROYABLE_OBSTACLE:
                    _record_affect((grid_x, grid_y), tick)
                    break
                # 其它情况，被炸，且不被阻挡，并有可能级联爆炸
                _record_affect((grid_x, grid_y), tick)
                if (grid_x, grid_y) in position_bomb_dict:
                    _propagate_from_bomb(position_bomb_dict[(grid_x, grid_y)], tick)

    for bomb in game_state["bombs"]:
        _propagate_from_bomb(bomb, bomb["explode_at"])

    for grid_pos, tick in affected_grid_tick_dict.items():
        on_explode(*grid_pos, tick)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./outputs/env.log", encoding="utf-8", level=logging.INFO
    )
    import json

    with open("./outputs/game_states_example.json", "r", encoding="utf-8") as f:
        game_states = json.load(f)
        for game_state in game_states:
            observation_space = observation_space_manager.from_game_state(game_state)
            logging.info(observation_space_manager.render(observation_space))

        reward, reward_detail = reward_manager.compute_reward(
            game_states[0], game_states[1]
        )
        logging.info({"reward": reward, "reward_detail": reward_detail})
