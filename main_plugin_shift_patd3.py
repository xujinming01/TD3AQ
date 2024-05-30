import argparse
import os
import random
import time

import gym
import numpy as np
import torch
from gym.wrappers import RescaleAction

import envs
from agents.TD3 import TD3
from common.evaluation_plugin import cycle_test
from common.wrappers import ScaledStateWrapper
from common.dp_optimal import OPTIMAL



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device_type, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device_type

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device))


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


def evaluate(env, agent):
    # set noise to 0
    policy_noise = agent.policy_noise
    agent.policy_noise = 0.

    # evaluate for 1 episode
    env.reset()
    state = env.reset_zero(soc_init=0.9)
    terminal = False
    episode_steps = 0
    episode_return = 0.
    while not terminal:
        episode_steps += 1
        action = agent.select_action(state)
        state, reward, terminal, _ = env.step(action)
        episode_return += reward

    # restore noise
    agent.policy_noise = policy_noise

    return episode_return, episode_steps


def run(args):
    dp = OPTIMAL[args.cycle]
    title = f"{args.algo.lower()}_{args.title}"

    # initialize the environment
    env = args.env_test if args.cycle_test else args.env_train
    print("running env:", env, "cycle:", args.cycle)
    env = gym.make(env, cycle=args.cycle)
    env = ScaledStateWrapper(env)  # scale states to [-1, 1]
    env = RescaleAction(env, -1, 1)  # scale actions to [-1, 1]

    # set random seed
    seed_torch(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(
        state_dim=s_dim,
        action_dim=a_dim,
        max_action=max_action,
        device=args.device,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    replay_buffer = ReplayBuffer(s_dim, a_dim, args.device, args.memory_size)

    model_dir = f"{args.env_train}_{title}"
    model_name = f"{title}_{args.seed}"
    model_dir = os.path.join("models/", model_dir, model_name)

    if args.cycle_test:
        agent.load_models(f'{model_dir}_best')
        print(f"Model loaded from: models/{model_name}")
        save_path = f'evaluations/{args.env_test}_{title}'
        save_path = os.path.join(save_path, model_name)
        cycle_test(env, agent, save_path, args.cycle, dp)
        import sys
        sys.exit()

    if args.load_model:
        agent.load_models(model_dir)
        print(f"The training is resumed from: models/{model_name}")

    start_time = time.time()

    best_return = -np.inf
    eval_returns = []
    total_timesteps = 0
    while total_timesteps < args.max_timesteps:
        state, terminal = env.reset(), False
        action = agent.select_action(state)
        episode_reward = 0.

        while not terminal:
            total_timesteps += 1

            next_state, reward, terminal, _ = env.step(action)
            next_action = agent.select_action(next_state)

            # store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, terminal)

            action = next_action
            state = next_state
            episode_reward += reward

            # train agent after collecting sufficient data
            if total_timesteps >= args.start_timesteps:
                agent.train(replay_buffer, args.batch_size)

            # evaluate agent periodically
            if total_timesteps % args.eval_freq == 0:
                eval_return, eval_episode_steps = evaluate(env, agent)

                print(f"{total_timesteps} steps, "
                      f"evaluate return: {eval_return:.4f}")

                # save best model during training
                if eval_return > best_return:
                    best_return = eval_return
                    os.makedirs(f"models/{args.env_train}_{title}",
                                exist_ok=True)
                    agent.save_models(f"{model_dir}_best")
                    print(f'performance gap: {eval_return / dp - 1:.2%}')

                eval_returns.append(eval_return)

    print(f'The best return is {best_return:.4f}, the gap to DP is '
          f'{best_return / dp - 1:.2%}')

    # save results
    redir = f"tmp/{args.algo}"
    import sys
    if sys.platform == 'win32':
        redir = redir.replace('\\', '/')
    os.makedirs(redir, exist_ok=True)
    print(f"Saving results to {redir}_{args.env_train}_{title}_{args.seed}")
    res_id = f"Eval_Reward_{args.env_train}_{title}_"
    save_path = os.path.join(redir, res_id + f"{args.seed}" + ".csv")
    np.savetxt(save_path, eval_returns, delimiter=',')

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()

    return best_return, dp, args.seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ------------No need to change------------
    parser.add_argument('--algo', default='PATD3', help='Algorithm to use.', type=str)
    parser.add_argument('--env-train', default='ems-v10', help='Environment for training.', type=str)
    parser.add_argument('--env-test', default='ems-v10', help='Environment for testing.', type=str)
    parser.add_argument('--cycle', default='plugin_1chtc_lt', help='Cycle to use.', type=str)
    parser.add_argument('--load_model', default=False, help='Load model.', type=bool)
    parser.add_argument('--cycle_test', default=False, help='Driving cycle test.', type=bool)
    # -----------------------------------------

    parser.add_argument('--title', default='2403_b256_eval1_1ep', help='Prefix of output files.', type=str)
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument('--max-timesteps', default=200000, help='Maximum number of timesteps.', type=int)
    parser.add_argument('--memory-size', default=200000, help='Replay memory size in transitions.', type=int)
    parser.add_argument('--start-timesteps', default=5000, help='Number of steps required to start learning.', type=int)
    parser.add_argument('--policy-noise', default=0.2, help='Standard deviation of Gaussian noise added to the actions for the exploration purposes.', type=float)
    parser.add_argument('--noise-clip', default=0.5, help='Range to clip target policy noise.', type=float)
    parser.add_argument('--batch-size', default=256, help='Minibatch size.', type=int)
    parser.add_argument('--discount', default=0.9874811195018006, help='Discount factor.', type=float)
    parser.add_argument('--actor-lr', default=0.0018167349642919, help='Actor learning rate.', type=float)
    parser.add_argument('--critic-lr', default=0.0018167349642919, help='Critic learning rate.', type=float)
    parser.add_argument('--policy-freq', default=2, help='Frequency of delayed policy updates.', type=int)
    parser.add_argument('--tau', default=0.0012366766593553957, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--eval-freq', default=1653, help='How often (time steps) to evaluate.', type=int)
    arguments = parser.parse_args()

    start = time.time()
    # device = torch.device("cpu")
    arguments.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arguments.cycle_test = True
    arguments.seed = 1
    run(arguments)

    # results = {}  # for multiple seeds
    # for i in range(2, 6):
    #     arguments.seed = i
    #     best_return, dp_reward, seed = run(arguments)
    #     results[seed] = f'best: {best_return}, ' \
    #                     f'gap to {dp_reward}: {best_return / dp_reward - 1:.2%}'
    # for k, v in results.items():
    #     print(f'seed {k}: {v}')

    torch.cuda.empty_cache()
    print(f"Total time: {((time.time() - start) / 3600):.2f} hours")
