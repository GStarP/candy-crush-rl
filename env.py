import random
import gymnasium as gym
from utils import configure_logging
from typing import Any
from game_manager import GameManager
import os
from game_utils import ActionSpaceManager, ObservationSpaceManager, RewardManager


observation_space_manager = ObservationSpaceManager()
action_space_manager = ActionSpaceManager()
reward_manager = RewardManager()


class SugarFightEnv(gym.Env):
    def __init__(self, id: int, work_dir: str, debug=False):
        super().__init__()

        self.id = id
        self.work_dir = os.path.join(work_dir, f"game_{self.id}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.debug = debug

        self.observation_space = observation_space_manager.describe()
        self.action_space = action_space_manager.describe()

        self._game_manager = None

        configure_logging(self.work_dir, debug=debug)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed, options=options)
        if self._game_manager is not None:
            self._game_manager.close()
            self._game_manager = None

        self._game_manager = GameManager(
            id=self.id,
            seed=seed or random.randint(0, 2**16 - 1),
            work_dir=self.work_dir,
            debug=self.debug,
        )
        self._game_manager.init()

        first_game_state, tick = self._game_manager.start_game()
        assert tick == 1

        obs = observation_space_manager.from_game_state(first_game_state)
        info = {}

        self._prev_game_state = None
        self._cur_game_state = first_game_state
        self._cur_tick = tick

        return obs, info

    def step(self, action):
        if self._game_manager is None:
            raise RuntimeError("step_err: game_manager_is_none")

        commands = [action_space_manager.encode_command(v) for v in action]
        self._game_manager.dispatch_commands(commands, self._cur_tick)

        next_game_state, last_command_timeout, tick, game_result = (
            self._game_manager.wait_for_game_state(self._cur_tick + 1)
        )
        # ! 时序不一致直接抛出异常，目前仍未触发
        assert tick == self._cur_tick + 1

        next_obs = observation_space_manager.from_game_state(next_game_state)
        reward, reward_detail = reward_manager.compute_reward(
            self._cur_game_state, next_game_state, game_result
        )
        done = game_result is not None

        info = {
            "last_command_timeout": last_command_timeout,
            "commands": commands,
            "next_game_state": next_game_state,
            "reward_detail": reward_detail,
        }

        self._prev_game_state = self._cur_game_state
        self._cur_game_state = next_game_state
        self._cur_tick = tick

        return next_obs, reward, done, False, info

    def close(self):
        if self._game_manager is not None:
            self._game_manager.close()
            self._game_manager = None
