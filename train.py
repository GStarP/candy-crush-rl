from env import SugarFightEnv
from stable_baselines3.common.monitor import Monitor
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == "__main__":
    SEED = 42

    work_dir = f"./temp/train_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    os.makedirs(work_dir, exist_ok=True)

    tensorboard_dir = os.path.join(work_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    model_dir = os.path.join(work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    env = SugarFightEnv(
        server_exe_path="./game/server.exe", id=1, work_dir=work_dir, debug=True
    )
    env = Monitor(env)

    # checkpoint 回调：定期保存
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix="ppo_sugarfight_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=tensorboard_dir,
        device="cuda",
    )

    # 预热模型
    model.predict(env.observation_space.sample(), deterministic=False)

    env.reset(seed=SEED)
    model.learn(total_timesteps=5400, progress_bar=True)
    model.save(os.path.join(model_dir, "AAA_ppo_sugarfight_ckpt"))

    print("train ended, model saved")

    env.close()
