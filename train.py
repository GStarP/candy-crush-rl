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


class MetricsCallback(BaseCallback):
    """Aggregate simple domain metrics and push to tensorboard.

    Metrics:
        - custom/good_bomb_rate: fraction of steps in rollout where place_good_bomb_reward>0
        - custom/destroy_obstacle_sum: sum of destroy_obstacle_reward in rollout
        - custom/stun_event_rate: fraction of steps with stun_reward < 0
    Assumes env info contains reward_detail dict.
    """

    def __init__(self):
        super().__init__()
        self.reset_accumulators()

    def reset_accumulators(self):
        self.good_bomb_events = 0
        self.destroy_obstacle_sum = 0.0
        self.stun_events = 0
        self.total_steps = 0

    def _on_rollout_start(self) -> None:  # type: ignore[override]
        self.reset_accumulators()

    def _on_step(self) -> bool:  # type: ignore[override]
        infos = self.locals.get("infos")
        if not infos:
            return True
        # SB3 with non-vec env: infos is a tuple/list len 1
        if isinstance(infos, (list, tuple)):
            iterable = infos
        else:
            iterable = [infos]
        for info in iterable:
            rd = info.get("reward_detail")
            if rd:
                if rd.get("place_good_bomb_reward", 0) > 0:
                    self.good_bomb_events += 1
                self.destroy_obstacle_sum += rd.get("destroy_obstacle_reward", 0.0)
                if rd.get("stun_reward", 0) < 0:
                    self.stun_events += 1
            self.total_steps += 1
        return True

    def _on_rollout_end(self) -> bool:  # type: ignore[override]
        if self.total_steps > 0:
            good_bomb_rate = self.good_bomb_events / self.total_steps
            stun_rate = self.stun_events / self.total_steps
        else:
            good_bomb_rate = 0.0
            stun_rate = 0.0
        self.logger.record("custom/good_bomb_rate", good_bomb_rate)
        self.logger.record(
            "custom/destroy_obstacle_sum", float(self.destroy_obstacle_sum)
        )
        self.logger.record("custom/stun_rate", stun_rate)
        return True


if __name__ == "__main__":
    SEED = 42
    DEBUG = False
    RESUME_MODEL_PATH = ""
    TOTAL_STEPS = 18000

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

    metrics_callback = MetricsCallback()

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
            # * 超参数
            n_steps=512,
            batch_size=512,
            learning_rate=2.5e-4,
            ent_coef=0.04,
            gamma=0.995,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        reset_num_timesteps = True

    # 预热模型，否则开始几个 tick 容易超时
    model.predict(env.observation_space.sample(), deterministic=False)

    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
        callback=[checkpoint_callback, timeout_mask_callback, metrics_callback],
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(os.path.join(model_dir, "A_ppo_sugarfight_ckpt"))

    print("🦄 train ended, model saved")

    env.close()
