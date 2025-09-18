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
import threading


class GameWrapper:
    TEAM_NAME = "LGD"
    PLAYER1_NAME = "AME"
    PLAYER2_NAME = "Maybe"
    GAME_DURATION = 3 * 60 * 10

    def __init__(self, server_exe_path: str, id: int):
        self.server_exe_path = os.path.abspath(server_exe_path)
        self.id = id

        self.temp_dir = f"./temp/game_{self.id}"
        self.player_server_port = 8000 + self.id

    def __enter__(self):
        self.game_states = [None] * (self.GAME_DURATION + 1)
        self.compute_times = [None] * (self.GAME_DURATION + 1)

        self.multiprocess_manager = multiprocessing.Manager()
        self.shared_game_state_queue = self.multiprocess_manager.Queue()
        self.shared_commands = self.multiprocess_manager.list(
            [None] * (self.GAME_DURATION + 1)
        )

        # 创建临时目录
        os.makedirs(self.temp_dir, exist_ok=True)
        # 创建配置文件
        config = self._create_config(player_server_port=self.player_server_port)
        with open(
            os.path.join(self.temp_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)
        # 启动游戏服务子进程
        self.game_server_process = subprocess.Popen(
            self.server_exe_path, cwd=self.temp_dir
        )
        time.sleep(1)
        # 启动玩家服务子进程
        self.player_server_process = multiprocessing.Process(
            target=_run_player_server_in_subprocess,
            args=(
                self.player_server_port,
                self.shared_game_state_queue,
                self.shared_commands,
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
        if self.multiprocess_manager:
            self.multiprocess_manager.shutdown()

    def start(self):
        game_ended = False
        while not game_ended:
            try:
                game_state = self.shared_game_state_queue.get()

                current_tick = game_state["current_tick"]
                self.game_states[current_tick] = game_state

                if current_tick >= self.GAME_DURATION:
                    game_ended = True

                # TODO
                def mock_compute():
                    direction = random.choice(["U", "D", "L", "R", "N"])
                    is_place_bomb = random.choice([True, False])
                    command = {"direction": direction, "is_place_bomb": is_place_bomb}

                    # 模拟一个 0.04-0.08s 之间的计算时间
                    mock_compute_time = random.uniform(0.04, 0.06)
                    self.compute_times[current_tick] = mock_compute_time
                    time.sleep(mock_compute_time)

                    self.shared_commands[current_tick] = [command, command]

                thread = threading.Thread(target=mock_compute)
                thread.start()
                try:
                    thread.join(timeout=0.1)
                except Exception:
                    pass

            except Exception:
                logging.exception("handle_game_state")
                break

        # 将整局游戏的情况记录下来
        ticks = []
        for i in range(1, self.GAME_DURATION + 1):
            ticks.append(
                {
                    "tick": i,
                    "game_state": self.game_states[i],
                    "command": self.shared_commands[i],
                    "compute_time": self.compute_times[i],
                }
            )
        overview = {"ticks": ticks}
        with open(
            os.path.join(self.temp_dir, "overview.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(overview, f, indent=2)

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
    shared_game_state_queue: multiprocessing.Queue,
    shared_commands,
    wait_command_timeout=0.07,
    wait_command_interval=0.005,
):
    player_server = FastAPI()
    player_server.state.current_tick = 0

    # 健康检查端点
    @player_server.head("/api/v1/ping")
    async def handle_ping():
        return

    # 游戏交互端点
    @player_server.post("/api/v1/command")
    async def handle_command(request: Request):
        game_state = await request.json()

        current_tick = game_state["current_tick"]
        if current_tick > player_server.state.current_tick:
            player_server.state.current_tick = current_tick
            shared_game_state_queue.put(game_state)
        elif current_tick < player_server.state.current_tick:
            return {"direction": "U", "is_place_bomb": False}

        command = {"direction": "U", "is_place_bomb": False}
        got_command = False
        wait_start_time = time.perf_counter()
        try:
            while (
                current_tick == player_server.state.current_tick
                and time.perf_counter() - wait_start_time < wait_command_timeout
            ):
                if shared_commands[current_tick] is not None:
                    commands = shared_commands[current_tick]
                    relative_player_index = get_player_relative_index(
                        game_state["my_player"]["id"]
                    )
                    command = commands[relative_player_index]
                    got_command = True
                    logging.info(
                        f"got_command: tick={current_tick}, player_index={relative_player_index}, command={command}"
                    )
                    break
                else:
                    await asyncio.sleep(wait_command_interval)
        except Exception:
            logging.exception(f"wait_for_command_error: tick={current_tick}")

        if not got_command:
            logging.warning(f"not_got_command: tick={current_tick}")

        return command

    uvicorn.run(player_server, host="0.0.0.0", port=port)


def get_player_relative_index(player_id: int) -> int:
    """
    1/2 => 0; 3/4 => 1
    """
    return (player_id - 1) // 2


if __name__ == "__main__":
    logging.basicConfig(filename="./outputs/game_wrapper.log", level=logging.INFO)

    with GameWrapper("./game/server.exe", 1) as game_wrapper:
        game_wrapper.start()
