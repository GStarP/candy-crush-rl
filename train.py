from env import SugarFightEnv
from stable_baselines3.common.monitor import Monitor
import time
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import logging


class TimeoutMaskCallback(BaseCallback):
    """Mask transitions whose action timed out (info['last_command_timeout']=True).

    Strategy:
    - Advantage for those steps set to 0 so they don't contribute to policy gradient.
    - Return set to value prediction (no value loss update either) -> effectively skipped.
    - Log timeout rate to tensorboard: custom/timeout_rate.
    Notes:
    - Requires SB3 rollout buffer to keep per-step infos (SB3 >= 2.0). If not available, it safely no-ops.
    - Does NOT physically remove steps -> keeps GAE recursion stable.
    """

    def __init__(self):
        super().__init__()
        self._last_timeout_rate = 0.0

    def _on_rollout_end(self) -> bool:  # type: ignore[override]
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None:
            logging.warning("TimeoutMaskCallback: rollout_buffer_missing, skipping")
            return True

        infos = getattr(buf, "infos", None)
        if infos is None:
            logging.warning("TimeoutMaskCallback: infos_missing_on_buffer, skipping")
            return True

        # Some envs (VecEnv) may return list of dict per step; here we assume single dict per step.
        try:
            mask = np.array(
                [info.get("last_command_timeout", False) for info in infos],
                dtype=bool,
            )
        except Exception as e:
            logging.warning(f"TimeoutMaskCallback: building mask failed: {e}")
            return True

        if mask.any():
            # Zero out advantage and value learning signal for timed-out steps.
            buf.advantages[mask] = 0.0
            # Replace returns with value predictions -> no value loss contribution.
            buf.returns[mask] = buf.values[mask]
            # Optional: could also zero rewards if you find value function still biased.
            # buf.rewards[mask] = 0.0
        timeout_rate = float(mask.mean()) if len(mask) > 0 else 0.0
        self._last_timeout_rate = timeout_rate
        # Log for monitoring
        self.logger.record("custom/timeout_rate", timeout_rate)
        return True

    def _on_step(self) -> bool:  # type: ignore[override]
        return True


if __name__ == "__main__":
    SEED = 42
    DEBUG = False
    RESUME_MODEL_PATH = (
        "./temp/train_20250928_190419/models/ppo_sugarfight_ckpt_5400_steps.zip"
    )

    work_dir = f"./temp/train_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    os.makedirs(work_dir, exist_ok=True)

    tensorboard_dir = os.path.join(work_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    model_dir = os.path.join(work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    env = SugarFightEnv(id=1, work_dir=work_dir, debug=DEBUG)
    env = Monitor(env)

    # 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=1800,
        save_path=model_dir,
        name_prefix="ppo_sugarfight_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    timeout_mask_callback = TimeoutMaskCallback()

    # 使用自定义 CNN 特征提取器，兼容自定义的观察空间
    from custom_cnn import SugarGridCNN

    if RESUME_MODEL_PATH:
        # ====== 恢复训练 ======
        # ? 注意 tensorboard 会继续写入之前的工作目录
        print(f"resume_from: {RESUME_MODEL_PATH}")
        model = PPO.load(RESUME_MODEL_PATH, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        # ====== 全新训练 ======
        policy_kwargs = dict(
            features_extractor_class=SugarGridCNN,
        )
        model = PPO(
            policy="CnnPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=SEED,
            tensorboard_log=tensorboard_dir,
            device="cuda",
        )
        reset_num_timesteps = True

    # 预热模型，否则开始几个 tick 容易超时
    model.predict(env.observation_space.sample(), deterministic=False)

    model.learn(
        total_timesteps=9000,
        progress_bar=True,
        callback=[checkpoint_callback],
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(os.path.join(model_dir, "A_ppo_sugarfight_ckpt"))

    print("🦄 train ended, model saved")

    env.close()
