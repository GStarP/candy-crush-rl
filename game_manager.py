import os

# import json
import logging
import time
import multiprocessing
import asyncio
import requests
import uvicorn
import random
from consts import Direction
from utils import configure_logging
from game_utils import inference_game_result
from game_server_api import GameServerAPI


class GameManager:
    TEAM_NAME = "LGD"
    PLAYER1_NAME = "AME"
    PLAYER2_NAME = "Maybe"
    GAME_DURATION = 3 * 60 * 10
    DEFAULT_COMMAND = {"direction": Direction.NOOP, "is_place_bomb": False}

    def __init__(
        self,
        id: int,
        work_dir: str,
        seed: int,
        debug=False,
    ):
        self.id = id
        self.work_dir = work_dir
        self.seed = seed
        self.debug = debug

        self.player_server_port = 8000 + self.id
        self.game_server_api = GameServerAPI()

    def init(self):
        self.shared_game_state_queue = multiprocessing.Queue()
        self.shared_commands_queue = multiprocessing.Queue()

        # TODO 由于现在 server.exe 由外部启动，这里创建配置已无意义，player_server_port 及 seed 也无意义
        self.player_server_port = 8000
        self.seed = 42
        # 创建配置文件
        # config = self._create_config()
        # with open(
        #     os.path.join(self.work_dir, "config.json"), "w", encoding="utf-8"
        # ) as f:
        #     json.dump(config, f, indent=2)

        # 启动玩家服务子进程
        self.player_server_process = multiprocessing.Process(
            target=_run_player_server_in_subprocess,
            args=(
                self.player_server_port,
                self.shared_game_state_queue,
                self.shared_commands_queue,
                self.work_dir,
                self.debug,
            ),
        )
        self.player_server_process.start()
        self._wait_until_player_server_ready(self.player_server_port)

    def close(self):
        # 停止玩家服务
        if self.player_server_process:
            try:
                self.player_server_process.terminate()
                self.player_server_process.join()
            except Exception as e:
                logging.exception(f"terminate_player_server_process_error: {e}")
            finally:
                self.player_server_process = None

        # 清理跨进程资源
        if self.shared_game_state_queue:
            try:
                self.shared_game_state_queue.close()
            except Exception as e:
                logging.exception(f"close_game_state_queue_error: {e}")
        if self.shared_commands_queue:
            try:
                self.shared_commands_queue.close()
            except Exception as e:
                logging.exception(f"close_commands_queue_error: {e}")

    def start_game(self):
        # 向游戏服务器发送一系列信令从而开始游戏
        self.game_server_api.start_new_game()

        game_state, _last_command_timeout, tick, _game_result = (
            self.wait_for_game_state(tick=1, wait_timeout=300)
        )
        return game_state, tick

    def wait_for_game_state(self, tick: int, wait_timeout=1.5):
        # payload = self.shared_game_state_queue.get(timeout=wait_timeout)
        # ! 鉴于游戏有暂停恢复信令，暂不设超时
        payload = self.shared_game_state_queue.get(timeout=None)
        game_state = payload["game_state"]
        last_command_timeout = payload["last_command_timeout"]
        logging.debug(f"consume_game_state: tick={game_state['current_tick']}")

        # TODO 游戏结束时，自行判断是否获胜
        game_result = None
        if game_state["current_tick"] >= self.GAME_DURATION:
            game_result = inference_game_result(game_state)

        return game_state, last_command_timeout, game_state["current_tick"], game_result

    def dispatch_commands(self, commands, tick):
        self.shared_commands_queue.put((tick, commands))
        logging.debug(f"dispatch_commands: tick={tick}, commands={commands}")

    def _create_config(self):
        return {
            "seed": self.seed,
            "teams": [
                {
                    "team_name": self.TEAM_NAME,
                    "players": [
                        {
                            "player_name": self.PLAYER1_NAME,
                            "ip_port": f"127.0.0.1:{self.player_server_port}",
                        },
                        {
                            "player_name": self.PLAYER2_NAME,
                            "ip_port": f"127.0.0.1:{self.player_server_port}",
                        },
                    ],
                },
                {
                    "team_name": "Robot_2",
                    "players": [
                        {"player_name": "Robot_2_1", "ip_port": "127.0.0.1:5001"},
                        {"player_name": "Robot_2_2", "ip_port": "127.0.0.1:5001"},
                    ],
                },
            ],
        }

    def _wait_until_player_server_ready(
        self, player_server_port: int, wait_interval=0.5, wait_timeout=10
    ):
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            try:
                response = requests.head(
                    f"http://127.0.0.1:{player_server_port}/api/v1/ping"
                )
                if response.status_code == 200:
                    logging.info("player_server_ready")
                    return
            except Exception:
                pass

            time.sleep(wait_interval)


def _run_player_server_in_subprocess(
    port: int,
    shared_game_states_queue: multiprocessing.Queue,
    shared_commands_queue: multiprocessing.Queue,
    work_dir: str,
    debug=False,
    wait_timeout=0.08,
):
    from fastapi import FastAPI, Request

    configure_logging(work_dir, debug=debug)

    player_server = FastAPI()
    player_server.state.current_tick = 0

    tick_control_list = [
        {
            "event": asyncio.Event(),
            "commands": None,
            "state_arrived_cnt": 0,
            "paused": False,
        }
        for _ in range(GameManager.GAME_DURATION + 2)
    ]
    import threading as _th

    def _consume_commands_queue():
        while True:
            try:
                tick, commands = shared_commands_queue.get()
                tick_control_list[tick]["commands"] = commands
                tick_control_list[tick]["event"].set()
                logging.debug(f"consume_command: tick={tick}, commands={commands}")
            except Exception:
                pass

    consumer_thread = _th.Thread(target=_consume_commands_queue, daemon=True)
    consumer_thread.start()

    last_command_timeout_dict = {}

    game_server_api = GameServerAPI()

    # 健康检查端点
    @player_server.head("/api/v1/ping")
    async def handle_ping():
        return

    # 游戏交互端点
    @player_server.post("/api/v1/command")
    async def handle_command(request: Request):
        game_state = await request.json()
        current_tick = game_state["current_tick"]
        commands_index = (game_state["my_player"]["id"] - 1) // 2

        if current_tick > player_server.state.current_tick:
            player_server.state.current_tick = current_tick
            shared_game_states_queue.put(
                {
                    "game_state": game_state,
                    "last_command_timeout": last_command_timeout_dict.get(
                        current_tick - 1, False
                    ),
                }
            )
            logging.debug(f"recv_game_state: tick={current_tick}")
        elif current_tick < player_server.state.current_tick:
            return GameManager.DEFAULT_COMMAND

        # ! 如果这一 tick 的两个请求都已接收到，但指令还未接收到，暂停游戏
        tick_control_list[current_tick]["state_arrived_cnt"] += 1
        if (
            tick_control_list[current_tick]["state_arrived_cnt"] == 2
            and tick_control_list[current_tick]["commands"] is None
        ):
            logging.debug(f"pause_game_server: tick={current_tick}")
            game_server_api.pause()
            tick_control_list[current_tick]["paused"] = True

        commands = [GameManager.DEFAULT_COMMAND] * 2
        got_commands = False

        try:
            # await asyncio.wait_for(
            #     tick_control_list[current_tick]["event"].wait(), timeout=wait_timeout
            # )
            # ! 鉴于游戏有暂停恢复信令，暂不设超时
            await asyncio.wait_for(
                tick_control_list[current_tick]["event"].wait(), timeout=None
            )
            commands = tick_control_list[current_tick]["commands"]
            got_commands = True

            # ! 这一 tick 指令就位时，如果游戏被暂停过，则需要恢复游戏
            if tick_control_list[current_tick]["paused"]:
                game_server_api.resume()
                tick_control_list[current_tick]["paused"] = False
                logging.debug(f"resume_game_server: tick={current_tick}")

            logging.debug(
                f"got_command: tick={current_tick}, player_id={game_state['my_player']['id']}, command={commands[commands_index]}"
            )
        except asyncio.TimeoutError:
            pass

        if not got_commands:
            logging.warning(
                f"not_got_commands: tick={current_tick}, player_id={game_state['my_player']['id']}"
            )
            last_command_timeout_dict[current_tick] = True

        return commands[commands_index]

    uvicorn.run(player_server, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    import os

    work_dir = "./temp/test_game_manager"
    os.makedirs(work_dir, exist_ok=True)

    configure_logging(work_dir, debug=True)

    def mock_compute():
        commands = []
        for _ in range(2):
            direction = random.choice(
                [
                    Direction.NOOP,
                    Direction.UP,
                    Direction.DOWN,
                    Direction.LEFT,
                    Direction.RIGHT,
                ]
            )
            is_place_bomb = random.choice([True, False])
            commands.append({"direction": direction, "is_place_bomb": is_place_bomb})
        return commands

    game_manager = GameManager(id=1, work_dir=work_dir, seed=42, debug=True)
    try:
        game_manager.init()
        cur_tick = 1

        game_state, tick = game_manager.start_game()
        assert tick == 1

        for i in range(10):
            commands = mock_compute()
            game_manager.dispatch_commands(commands, cur_tick)
            cur_tick += 1
            game_state, _last_command_timeout, tick, _game_result = (
                game_manager.wait_for_game_state(tick=cur_tick)
            )
            assert tick == cur_tick
    finally:
        game_manager.close()
