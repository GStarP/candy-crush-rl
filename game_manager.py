import os
import json
import logging
import time
from fastapi import FastAPI, Request
import multiprocessing
import subprocess
import asyncio
import uvicorn
import random
from consts import Direction


class GameManager:
    TEAM_NAME = "LGD"
    PLAYER1_NAME = "AME"
    PLAYER2_NAME = "Maybe"
    GAME_DURATION = 3 * 60 * 10
    DEFAULT_COMMAND = {"direction": Direction.NOOP, "is_place_bomb": False}

    def __init__(self, server_exe_path: str, id: int, debug=False):
        self.server_exe_path = os.path.abspath(server_exe_path)
        self.id = id
        self.debug = debug

        self.work_dir = f"./temp/game_{self.id}"
        self.player_server_port = 8000 + self.id

    def __enter__(self):
        self.shared_game_state_queue = multiprocessing.Queue()
        self.shared_commands_queue = multiprocessing.Queue()

        # 创建临时目录
        os.makedirs(self.work_dir, exist_ok=True)
        # 配置日志
        logging.basicConfig(
            filename=os.path.join(self.work_dir, "game_manager.log"),
            encoding="utf-8",
            level=logging.DEBUG if self.debug else logging.INFO,
            format="%(asctime)s.%(msecs)03d %(levelname)s [pid=%(process)d] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # 创建配置文件
        config = self._create_config(player_server_port=self.player_server_port)
        with open(
            os.path.join(self.work_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)
        # 启动游戏服务子进程
        self.game_server_process = subprocess.Popen(
            self.server_exe_path, cwd=self.work_dir
        )
        time.sleep(1)
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
        time.sleep(1)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 停止游戏服务
        if self.game_server_process:
            try:
                self.game_server_process.terminate()
                self.game_server_process.wait()
            except Exception as e:
                logging.exception(f"terminate_game_server_process_error: {e}")
            finally:
                self.game_server_process = None

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
            self.shared_game_state_queue.close()
        if self.shared_commands_queue:
            self.shared_commands_queue.close()

    def start_game(self):
        # TODO 发送游戏开始信令
        game_state, tick, _game_result = self.wait_for_game_state(
            tick=1, wait_timeout=60
        )
        return game_state, tick

    def wait_for_game_state(self, tick: int, wait_timeout=1.5):
        game_state = self.shared_game_state_queue.get(timeout=wait_timeout)
        logging.debug(f"consume_game_state: tick={game_state['current_tick']}")
        # TODO 游戏结束时，获取并返回 game_result
        return game_state, game_state["current_tick"], None

    def dispatch_commands(self, commands, tick):
        self.shared_commands_queue.put((tick, commands))
        logging.debug(f"dispatch_commands: tick={tick}, commands={commands}")

    def _create_config(self, player_server_port: int):
        return {
            "seed": 1,
            "teams": [
                {
                    "team_name": self.TEAM_NAME,
                    "players": [
                        {
                            "player_name": self.PLAYER1_NAME,
                            "ip_port": f"127.0.0.1:{player_server_port}",
                        },
                        {
                            "player_name": self.PLAYER2_NAME,
                            "ip_port": f"127.0.0.1:{player_server_port}",
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


def _run_player_server_in_subprocess(
    port: int,
    shared_game_states_queue: multiprocessing.Queue,
    shared_commands_queue: multiprocessing.Queue,
    work_dir: str,
    debug=False,
    wait_timeout=0.07,
):
    logging.basicConfig(
        filename=os.path.join(work_dir, "game_manager.log"),
        encoding="utf-8",
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s [pid=%(process)d] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    player_server = FastAPI()
    player_server.state.current_tick = 0

    real_interactions = [
        {"game_state": None, "commands": [None, None]}
        for _ in range(GameManager.GAME_DURATION + 2)
    ]

    commands_event_list = [
        {
            "event": asyncio.Event(),
            "commands": None,
        }
        for _ in range(GameManager.GAME_DURATION + 2)
    ]
    import threading as _th

    def _consume_commands_queue():
        while True:
            try:
                tick, commands = shared_commands_queue.get()
                commands_event_list[tick]["commands"] = commands
                commands_event_list[tick]["event"].set()
                logging.debug(f"consume_command: tick={tick}, commands={commands}")
            except Exception:
                pass

    consumer_thread = _th.Thread(target=_consume_commands_queue, daemon=True)
    consumer_thread.start()

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
            shared_game_states_queue.put(game_state)
            logging.debug(f"recv_game_state: tick={current_tick}")
            real_interactions[current_tick]["game_state"] = game_state
        elif current_tick < player_server.state.current_tick:
            real_interactions[current_tick]["commands"][commands_index] = (
                GameManager.DEFAULT_COMMAND
            )
            return GameManager.DEFAULT_COMMAND

        commands = [GameManager.DEFAULT_COMMAND] * 2
        got_commands = False

        try:
            await asyncio.wait_for(
                commands_event_list[current_tick]["event"].wait(), timeout=wait_timeout
            )
            commands = commands_event_list[current_tick]["commands"]
            got_commands = True
            logging.debug(
                f"got_command: tick={current_tick}, player_id={game_state['my_player']['id']}, command={commands[commands_index]}"
            )
        except asyncio.TimeoutError:
            pass

        if not got_commands:
            logging.warning(
                f"not_got_commands: tick={current_tick}, player_id={game_state['my_player']['id']}"
            )

        real_interactions[current_tick]["commands"][commands_index] = commands[
            commands_index
        ]

        return commands[commands_index]

    uvicorn.run(player_server, host="0.0.0.0", port=port)

    with open(
        os.path.join(work_dir, "real_interactions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(real_interactions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

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

    with GameManager(
        server_exe_path="./game/server.exe", id=1, debug=True
    ) as game_manager:
        cur_tick = 1

        game_state, tick = game_manager.start_game()
        assert tick == 1

        for i in range(10):
            commands = mock_compute()
            game_manager.dispatch_commands(commands, cur_tick)
            cur_tick += 1
            game_state, tick, _game_result = game_manager.wait_for_game_state(
                tick=cur_tick
            )
            assert tick == cur_tick
