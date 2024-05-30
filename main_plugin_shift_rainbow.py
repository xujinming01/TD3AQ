import os
import random

import fire
import gym
import numpy as np
import torch

import envs  # noqa: E402
from agents.rainbow import DQNAgent
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from stable_baselines3.common.monitor import Monitor
from common.dp_optimal import OPTIMAL



def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ['PYTHONHASHSEED'] = str(seed)

def run(
        seed=1,
        env_name='ems-v11',
        cycle='plugin_1hhddt',
        max_timesteps=200000,
        eval_freq=1653,
        gamma=0.9899283966156008,
        lr=0.0008731330958270822,
        memory_size=200000,
        batch_size=512,
        learning_start=5000,
        target_update_interval=2500,
        device='cuda:0',
        cycle_test=True,
        # cycle_test=False,
):
    print(f'Running {env_name} with cycle {cycle} on {device}')
    dp_solution = OPTIMAL[cycle]
    env = gym.make(env_name, cycle=cycle)
    env = ScaledStateWrapper(env)  # scale state to [-1, 1]
    # env = ScaledParameterisedActionWrapper(env)  # scale action to [-1, 1]

    logs_dir = f"tmp/RAINBOW"
    identity = '2403_b512_eval1_1ep'  # NOTE: change this to identify the run
    # identity = 'optuna'
    log_dir = f'{logs_dir}/{env_name}_{identity}_{seed}'
    os.makedirs(log_dir, exist_ok=True)
    # env = Monitor(env, log_dir)

    seed_torch(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    agent = DQNAgent(
        env=env,
        memory_size=memory_size,
        batch_size=batch_size,
        learning_start=learning_start,
        target_update=target_update_interval,
        gamma=gamma,
        lr=lr,
        device=device,
    )

    if cycle_test:
        from common.evaluation_plugin import cycle_test
        agent.load_model(f'{log_dir}/best_model')
        eval_path = f'evaluations/{env_name}_{identity}/seed{seed}'
        print(f'eval_path: {eval_path}')
        cycle_test(env, agent, eval_path, cycle, dp_solution)
        # agent.load_model(f'{log_dir}/final_model')
        # eval_path = f'evaluations/{env_name}_{identity}/seed{seed}/final'
        # cycle_test(env, agent, eval_path, cycle, dp_solution)
    else:
        best_return = agent.train(max_timesteps, log_dir, dp_solution, eval_freq)
        torch.cuda.empty_cache()

        return best_return, dp_solution

if __name__ == '__main__':
    fire.Fire(run)
