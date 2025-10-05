import argparse
from fastapi import FastAPI, Request
import uvicorn
from game_utils import ObservationSpaceManager, ActionSpaceManager
import asyncio
from consts import GAME_TOTAL_TICKS
import logging
import numpy as np
from stable_baselines3 import PPO
import torch
from game_manager import GameManager

if __name__ == "__main__":
    # 必须引入，否则模型无法正确加载
    from custom_cnn import SugarGridCNN  # noqa: F401

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run player server with model.")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the server to listen on"
    )
    args = parser.parse_args()

    model_path = args.model_path

    observation_space_manager = ObservationSpaceManager()
    action_space_manager = ActionSpaceManager()

    # 选择设备并加载模型（需要提前 import SugarGridCNN 以便 SB3 还原自定义特征提取器）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"load_model: model_path={model_path}, device={device}")
    model: PPO = PPO.load(model_path, device=device)

    # 预热一次，降低首帧延迟
    dummy_obs = observation_space_manager.describe().sample().astype(np.float32)
    model.predict(dummy_obs, deterministic=True)

    app = FastAPI()

    tick_control_list = [
        {
            "event": asyncio.Event(),
            "commands": [GameManager.DEFAULT_COMMAND, GameManager.DEFAULT_COMMAND],
            "predicted": False,
            "lock": asyncio.Lock(),  # 避免同一 tick 并发两次推理
        }
        for _ in range(GAME_TOTAL_TICKS + 2)
    ]

    @app.head("/api/v1/ping")
    async def handle_ping():
        return

    @app.post("/api/v1/command")
    async def handle_command(request: Request):
        game_state = await request.json()
        current_tick = game_state["current_tick"]
        commands_index = (game_state["my_player"]["id"] - 1) // 2

        async with tick_control_list[current_tick]["lock"]:
            if not tick_control_list[current_tick]["predicted"]:
                try:
                    observation = observation_space_manager.from_game_state(
                        game_state
                    ).astype(np.float32)
                    actions, _ = model.predict(observation, deterministic=True)

                    if current_tick <= 100:
                        logging.info(f"actions: tick={current_tick}, actions={actions}")

                    commands = [
                        action_space_manager.encode_command(int(a)) for a in actions
                    ]
                    tick_control_list[current_tick]["commands"] = commands
                except Exception:
                    logging.exception(f"predict_err: tick={current_tick}")
                finally:
                    tick_control_list[current_tick]["event"].set()
                    tick_control_list[current_tick]["predicted"] = True

        command = GameManager.DEFAULT_COMMAND
        got_commands = False
        try:
            await asyncio.wait_for(
                tick_control_list[current_tick]["event"].wait(), timeout=0.7
            )
            command = tick_control_list[current_tick]["commands"][commands_index]
            got_commands = True
        except Exception:
            logging.exception(
                f"wait_for_command_err: tick={current_tick}, player_id={game_state['my_player']['id']}"
            )

        if not got_commands:
            logging.warning(
                f"not_got_command: tick={current_tick}, player_id={game_state['my_player']['id']}"
            )

        return command

    logging.info(f"player_server_started: port={args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
