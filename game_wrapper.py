import os
import asyncio
import json
import logging


class GameWrapper:
    TEAM_NAME = "LGD"
    PLAYER1_NAME = "AME"
    PLAYER2_NAME = "萧瑟"

    def __init__(self, server_exe_path: str, id: int):
        self.server_exe_path = os.path.abspath(server_exe_path)
        self.id = id

        self.temp_dir = f"./temp/game_{self.id}"
        self.player_server_port = 8000 + self.id

    async def start(self):
        # 创建临时目录
        os.makedirs(self.temp_dir, exist_ok=True)
        # 创建配置文件
        config = self._create_config(player_server_port=self.player_server_port)
        with open(
            os.path.join(self.temp_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)
        # 启动游戏服务器子进程，直接执行可执行文件
        self.game_server_process = await asyncio.create_subprocess_exec(
            self.server_exe_path, cwd=self.temp_dir
        )
        await asyncio.sleep(1)

    async def stop(self):
        if self.game_server_process:
            try:
                self.game_server_process.terminate()
                await self.game_server_process.wait()
            except Exception as e:
                logging.exception(f"kill_game_server_process_error: {e}")
            finally:
                self.game_server_process = None

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


if __name__ == "__main__":

    async def main():
        game_wrapper = GameWrapper("./game/server.exe", 1)
        await game_wrapper.start()

        await asyncio.sleep(3)
        await game_wrapper.stop()

    asyncio.run(main())
