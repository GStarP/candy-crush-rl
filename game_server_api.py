import logging
import requests
import subprocess
import time
from pathlib import Path


def raise_for_errorCode(ret: dict):
    if ret.get("errorCode", 0) != 0:
        raise RuntimeError(f"raise_for_errorCode: ret={ret}")


game_server_process = None


class GameServerAPI:
    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:3000/v1/game"

    def get_teams(self) -> list[dict]:
        response = requests.get(f"{self.base_url}/get_teams")
        response.raise_for_status()
        ret = response.json()
        raise_for_errorCode(ret)
        return ret["teams"]

    def set_teams(self, teams: list[int]):
        response = requests.post(
            f"{self.base_url}/set_teams",
            json={"redteamid": teams[0], "blueteamid": teams[1]},
        )
        response.raise_for_status()
        ret = response.json()
        raise_for_errorCode(ret)

    def start(self):
        response = requests.get(f"{self.base_url}/start")
        response.raise_for_status()
        ret = response.json()
        raise_for_errorCode(ret)

    def system_command(self, command_type: int):
        response = requests.post(
            f"{self.base_url}/system_command", json={"type": command_type}
        )
        response.raise_for_status()
        ret = response.json()
        raise_for_errorCode(ret)

    def pause(self):
        try:
            self.system_command(0)
        except Exception:
            logging.exception("pause_game_server_error")

    def resume(self):
        try:
            self.system_command(1)
        except Exception:
            logging.exception("resume_game_server_error")

    def stop(self):
        try:
            self.system_command(2)
        except Exception:
            logging.exception("stop_game_server_error")

    def start_new_game(self):
        teams = self.get_teams()
        self.set_teams([team["team_id"] for team in teams])

        self.start()
        time.sleep(1)

        self.resume()

    def relaunch_game_server(self):
        global game_server_process

        # 如果游戏服务器进程存在，优雅退出
        if game_server_process is not None:
            try:
                # 检查进程是否还在运行
                if game_server_process.poll() is None:
                    # 进程仍在运行，尝试终止
                    game_server_process.terminate()
                    time.sleep(1)

                    # 如果还是没有退出，强制杀死
                    if game_server_process.poll() is None:
                        game_server_process.kill()
                        logging.warning("强制终止游戏服务器进程")

                game_server_process.wait()  # 等待进程完全退出
                logging.info("游戏服务器进程已停止")
            except Exception as e:
                logging.exception(f"停止游戏服务器进程时出错: {e}")
            finally:
                game_server_process = None

        # 启动新的游戏服务器进程
        game_dir = Path(__file__).resolve().parent / "game"
        game_server_exe = game_dir / "server.exe"

        try:
            logging.info(f"正在启动游戏服务器: {game_server_exe}")
            game_server_process = subprocess.Popen(
                [str(game_server_exe)],
                cwd=str(game_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE
                if hasattr(subprocess, "CREATE_NEW_CONSOLE")
                else 0,
            )

            # 等待一下确保服务器启动
            time.sleep(3)

            # 检查进程是否成功启动
            if game_server_process.poll() is not None:
                # 进程已经退出，说明启动失败
                stdout, stderr = game_server_process.communicate()
                error_msg = f"游戏服务器启动失败，退出码: {game_server_process.returncode}\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
                logging.error(error_msg)
                game_server_process = None
                raise RuntimeError(error_msg)

            logging.info(f"游戏服务器已成功启动，进程ID: {game_server_process.pid}")

        except Exception as e:
            logging.exception(f"启动游戏服务器时出错: {e}")
            game_server_process = None
            raise
