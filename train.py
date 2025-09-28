from env import SugarFightEnv
import time
import os
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN

# from ray.rllib.algorithms.algorithm import Algorithm
import ray
import json
from pathlib import Path


if __name__ == "__main__":
    SEED = 42

    work_dir = f"./temp/train_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    os.makedirs(work_dir, exist_ok=True)

    model_dir = os.path.join(work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    def make_env(env_context: EnvContext):
        return SugarFightEnv(
            server_exe_path="./game/server.exe",
            id=(env_context.worker_index * 10 + env_context.vector_index),
            work_dir=work_dir,
            debug=True,
        )

    ENV_NAME = "SugarFightEnv-RLlib"
    register_env(ENV_NAME, make_env)

    ray.init(local_mode=False)

    config = (
        APPOConfig()
        .environment(env=ENV_NAME)
        .resources(num_gpus=1)
        .framework("torch")
        .learners(num_learners=1, num_gpus_per_learner=1)
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            num_gpus_per_env_runner=0,
            # 每次从 actor 回传的样本数，过小会通信频繁，过大会数据过多
            rollout_fragment_length=50,
            # 使用独立进程启动 actor，更隔离
            remote_worker_envs=True,
            # 允许不完成一局也回传样本
            batch_mode="truncate_episodes",
        )
        .rl_module(
            # ! 为了解决观察空间与内置 CNN 输入尺寸不匹配的问题
            model_config={
                "conv_layout": "NCHW",
                "conv_filters": [
                    [32, [3, 3], 1],
                    [64, [3, 3], 1],
                    [128, [3, 3], 1],
                ],
                "conv_activation": "relu",
                "fcnet_hiddens": [512, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            }
        )
        .training(
            # 每多少样本训练一次（所有并行环境）
            train_batch_size=300,
            train_batch_size_per_learner=300,
            # 超参数
            lr=0.0005,
            gamma=0.995,
        )
    )
    algo = config.build_algo()

    # 加载已有模型文件
    # LAST_MODEL_FILE = ""
    # if os.path.exists(LAST_MODEL_FILE):
    #     algo = Algorithm.from_checkpoint(LAST_MODEL_FILE)
    #     print(f"resume_checkpoint: file={LAST_MODEL_FILE}")

    done = False
    all_model_info = []

    try:
        while not done:
            # 积累到 train_batch_size 个样本后返回
            result = algo.train()

            training_iteration = result.get("training_iteration", -1)
            train_reward = result.get(ENV_RUNNER_RESULTS, {}).get(
                EPISODE_RETURN_MEAN, None
            )
            print(f"train: training_iteration={training_iteration}")

            if train_reward is None:
                continue

            done = True

            checkpoint_path = algo.save_to_path(path=Path(model_dir))

            all_model_info.append(
                {
                    "training_iteration": training_iteration,
                    "train_reward": train_reward,
                    "checkpoint_path": checkpoint_path,
                }
            )

    finally:
        ray.shutdown()

        if len(all_model_info) > 0:
            with open(
                os.path.join(model_dir, "all_model_info.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(all_model_info, f, ensure_ascii=False, indent=4)
