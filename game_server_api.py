import logging
import requests


def raise_for_errorCode(ret: dict):
    if ret.get("errorCode", 0) != 0:
        raise RuntimeError(f"raise_for_errorCode: ret={ret}")


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
        self.stop()
        teams = self.get_teams()
        self.set_teams([team["team_id"] for team in teams])
        self.start()
        self.resume()
