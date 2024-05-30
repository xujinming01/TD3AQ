import argparse
import os
import time

import gym
import numpy as np
import torch

import envs
from agents.utils.noise import OrnsteinUhlenbeckActionNoise
from common.evaluation_plugin import cycle_test
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.utils import pad_action
from common.dp_optimal import OPTIMAL


def evaluate(env, agent, epsilon, episodes=1000):
    returns = []
    episode_steps = []
    agent.epsilon = 0.
    agent.noise = False
    for _ in range(episodes):
        state = env.reset()
        if episodes == 1:
            soc_init = 0.9
        else:
            soc_init = np.random.uniform(0.35, 0.9)
        state = env.reset_zero(soc_init=soc_init)
        # env.evaluate_on()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, all_action_parameters = agent.act(state)
            action = pad_action(act, all_action_parameters)
            state, reward, terminal, _ = env.step(action)
            total_reward += reward

        episode_steps.append(t)
        returns.append(total_reward)

    agent.epsilon = epsilon
    # print("agent.epsilon",agent.epsilon)
    agent.noise = True

    mean_return = np.mean(returns[-episodes:])
    mean_steps = np.mean(episode_steps[-episodes:])

    return mean_return, mean_steps

def run(args):
    dp_reward = OPTIMAL[args.cycle]

    title = f"{args.algo.lower()}_{args.title}"
    # if args.save_freq > 0 and args.save_dir:
    #     save_dir = os.path.join(args.save_dir, args.title + "{}".format(str(args.seed)))
    #     os.makedirs(save_dir, exist_ok=True)
    # assert not (args.save_frames and args.visualise)
    # if args.visualise:
    #     assert args.render_freq > 0
    # if args.save_frames:
    #     assert args.render_freq > 0
    #     vidir = os.path.join(save_dir, "frames")
    #     os.makedirs(vidir, exist_ok=True)

    env = args.env_test if args.cycle_test else args.env_train
    print("running env:", env, "cycle:", args.cycle)
    env = gym.make(env, cycle=args.cycle)
    # init_params_ = [0., 0.]  # TODO: initialize continuous actions?
    # if args.scale_actions:  # NOTE: different from environment
    #     n_discrete = 0
    #     for a in range(env.action_space.spaces.__len__()):
    #         if env.action_space[a].__class__.__name__ == "Discrete":
    #             n_discrete += 1
    #             continue
    #         else:
    #             idx = a - n_discrete
    #             diff = init_params_[idx] - env.action_space.spaces[a].low
    #             scale = env.action_space.spaces[a].high - env.action_space.spaces[a].low
    #             init_params_[idx] = diff * 2. / scale - 1.

    env = ScaledStateWrapper(env)  # scale state to [-1, 1]
    # env = PlatformFlattenedActionWrapper(env)
    if args.scale_actions:
        env = ScaledParameterisedActionWrapper(env)  # scale action to [-1, 1]

    # dir = os.path.join(save_dir, args.title)
    # env = Monitor(env, directory=os.path.join(dir, str(args.seed)), video_callable=False, write_upon_reset=False,
    #               force=True)
    env.seed(args.seed)
    np.random.seed(args.seed)

    # print(env.observation_space)

    if args.algo == 'TD3AQ':
        from agents.td3aq import TD3AQAgent
        agent_class = TD3AQAgent
    elif args.algo == 'PDQN_TD3':
        from agents.pdqn_td3 import PDQNAgent
        agent_class = PDQNAgent
    elif args.algo == 'TD3AQM':
        from agents.td3aq_mini import TD3AQMAgent
        agent_class = TD3AQMAgent
    else:
        raise ValueError("Unknown algorithm")

    agent = agent_class(
        env.observation_space,
        env.action_space,
        batch_size=args.batch_size,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_actor_param=args.learning_rate_actor_param,
        epsilon_steps=args.epsilon_steps,
        epsilon_final=args.epsilon_final,
        gamma=args.gamma,
        policy_freq=args.policy_freq,  # delay policy updates
        tau_actor=args.tau_actor,
        tau_actor_param=args.tau_actor_param,
        clip_grad=args.clip_grad,
        indexed=args.indexed,
        weighted=args.weighted,
        average=args.average,
        random_weighted=args.random_weighted,
        initial_memory_threshold=args.initial_memory_threshold,
        noise=args.noise,
        use_ornstein_noise=args.use_ornstein_noise,
        use_normal_noise=args.use_normal_noise,
        replay_memory_size=args.replay_memory_size,
        inverting_gradients=args.inverting_gradients,
        zero_index_gradients=args.zero_index_gradients,
        device=args.device,
        seed=args.seed)

    # if args.initialise_params:
    #     initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.shape[0]))
    #     # initial_bias = np.zeros(env.action_space.spaces[0].n)
    #     # for a in range(env.action_space.spaces[0].n):
    #     #     initial_bias[a] = init_params_[a]
    #     initial_bias = init_params_
    #     agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)

    model_dir = f"{args.env_train}_{title}"
    model_name = f"{title}_{args.seed}"
    model_dir = os.path.join("models/", model_dir, model_name)
    assert not (args.evaluate_model and args.cycle_test)
    if args.evaluate_model:  # load model and evaluate
        agent.load_models(model_dir)
        print(f"Model loaded from: models/{model_name}")

        # the evaluation environment is same as the training environment
        eval_episodes = 1
        evaluate(env, agent, args.epsilon_final, eval_episodes)
        import sys
        sys.exit()
    if args.cycle_test:
        agent.load_models(f'{model_dir}_best')
        print(f"Model loaded from: models/{model_name}")
        save_path = os.path.join(f'evaluations/{args.env_test}_{title}', f'{title}_{args.seed}')
        cycle_test(env, agent, save_path, args.cycle, dp_reward, episodes=args.eval_episodes)
        import sys
        sys.exit()

    # print(agent)
    if args.load_model:
        agent.load_models(model_dir)
        print(f"The training is resumed from: models/{model_name}")
    max_steps = 10000000  # meaningless for now, loop break by terminal
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0

    best_return = -np.inf
    Eval_Reward = 0.0
    Eval_Return = []
    Reward = []
    episode_steps = []
    episode_steps_100 = []
    total_timesteps = 0
    # for i in range(episodes):
    while total_timesteps < args.max_timesteps:
        state = env.reset()

        state = np.array(state, dtype=np.float32, copy=False)

        disc_a, conti_a = agent.act(state)
        action = pad_action(disc_a, conti_a)

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):
            total_timesteps += 1
            next_state, reward, terminal, _ = env.step(action)

            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_disc_a, next_conti_a = agent.act(next_state)

            next_action = pad_action(next_disc_a, next_conti_a)
            agent.step(state, action, reward, next_state, next_action, terminal)
            action = next_action
            state = next_state

            episode_reward += reward
            if total_timesteps % args.eval_freq == 0:
                while not terminal:  # 如果没结束需要继续推演
                    state = np.array(state, dtype=np.float32, copy=False)
                    disc_a, conti_a = agent.act(state)
                    action = pad_action(disc_a, conti_a)
                    state, reward, terminal, _ = env.step(action)

                Eval_Reward, Test_episode_step = evaluate(env, agent, agent.epsilon, episodes=args.eval_episodes)

                print(f"{total_timesteps} steps, evaluate over {args.eval_episodes} episodes, "
                      f"mean return: {Eval_Reward:.4f}, mean steps: {Test_episode_step:.0f}")
                if Eval_Reward > best_return:
                    best_return = Eval_Reward
                    os.makedirs(f"models/{args.env_train}_{title}",
                                exist_ok=True)
                    agent.save_models(f"{model_dir}_best")
                    print(f'performance gap: '
                          f'{Eval_Reward / dp_reward - 1:.2%}')

                # Eval_Return.append(Eval_Reward / Test_episode_step)
                Eval_Return.append(Eval_Reward)
                # Reward.append(total_reward / (i + 1))
                episode_steps_100.append(Test_episode_step)

            if terminal:
                agent.end_episode()
                break

        returns.append(episode_reward)
        total_reward += episode_reward

        # if i % 100 == 0:
        #     Eval_Reward, Test_episode_step = evaluate(env, agent,agent.epsilon, episodes=100)
        #
        #     print('{0:5s} R:{1:.4f} r100:{2:.4f} steps:{3:.4f}'.format(str(i), total_reward / (i + 1), Eval_Reward, Test_episode_step))
        #     Eval_Return.append(Eval_Reward)
        #     Reward.append(total_reward / (i + 1))
        #     episode_steps_100.append(Test_episode_step)

    print(f'The best return is {best_return:.4f}, the gap to DP is '
          f'{best_return / dp_reward - 1:.2%}')
    print(f'The final return is {Eval_Reward:.4f}, the gap to DP is '
          f'{Eval_Reward / dp_reward - 1:.2%}')

    redir = f"tmp/{args.algo}"
    # data = "0715"
    # redir = os.path.join(dir, data)
    import sys
    if sys.platform == 'win32':
        redir = redir.replace('\\', '/')
    os.makedirs(redir, exist_ok=True)
    print(f"Saving results to {redir}_{args.env_train}_{title}_{args.seed}")
    # title1 = f"Reward_{args.algo}_ems_"
    title2 = f"Eval_Reward_{args.env_train}_{title}_"
    # title3 = f"episode_steps_100_{args.algo}_ems_"

    # np.savetxt(os.path.join(redir, title1 + "{}".format(str(args.seed) + ".csv")), Reward, delimiter=',')
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Eval_Return, delimiter=',')
    # np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), episode_steps_100, delimiter=',')

    if args.save_model:
        os.makedirs(f"models/{args.env_train}_{title}", exist_ok=True)
        agent.save_models(model_dir)

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()

    return best_return, dp_reward, args.seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.9801256558248345, help='Discount factor.', type=float)
    parser.add_argument('--inverting-gradients', default=True, help='Use inverting gradients scheme instead of squashing function.', type=bool)
    parser.add_argument('--initial-memory-threshold', default=5000, help='Number of transitions required to start learning.', type=int)  # may have been running with 500??
    parser.add_argument('--noise', default=True, help='Use noise in action selection.', type=bool)
    parser.add_argument('--use-ornstein-noise', default=False, help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--use_normal_noise', default=True, help='Use normal noise', type=bool)
    parser.add_argument('--replay-memory-size', default=200000, help='Replay memory size in transitions.', type=int)
    parser.add_argument('--epsilon-steps', default=30, help='Number of episodes over which to linearly anneal epsilon.', type=int)
    parser.add_argument('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
    parser.add_argument('--tau-actor', default=0.003239900224954052, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--tau-actor-param', default=0.0011275335915165718, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--learning-rate-actor', default=0.0020700573615610616, help="discrete actor  learning rate.", type=float)
    parser.add_argument('--learning-rate-actor-param', default=0.0021502881652993074, help="parameter actor  learning rate.", type=float)
    parser.add_argument('--scale-actions', default=True, help="Scale actions.", type=bool)
    parser.add_argument('--initialise-params', default=False, help='Initialise action parameters.', type=bool)  # TODO: initialize?
    parser.add_argument('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--indexed', default=False, help='Indexed loss function.', type=bool)
    parser.add_argument('--weighted', default=False, help='Naive weighted loss function.', type=bool)
    parser.add_argument('--average', default=False, help='Average weighted loss function.', type=bool)
    parser.add_argument('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
    parser.add_argument('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
    parser.add_argument('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
    parser.add_argument("--policy_freq", default=2, help="Frequency of delayed policy updates", type=int)
    parser.add_argument('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
    parser.add_argument("--save_model", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default=False)  # Load model and optimizer parameters
    parser.add_argument("--evaluate_model", action="store_true")  # Load model and optimizer parameters then evaluate
    # parser.add_argument('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)  # useless for now, only for after training
    # parser.add_argument('--episodes', default=50000, help='Number of episodes.', type=int)  # useless for now
    # parser.add_argument('--split', default=False, help='Separate action-parameter inputs.', type=bool)
    # parser.add_argument('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
    # parser.add_argument('--save-dir', default="results/model/ems", help='Output directory.', type=str)
    # parser.add_argument('--render-freq', default=0, help='How often to render / save frames of an episode.', type=int)
    # parser.add_argument('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
    # parser.add_argument('--visualise', default=False, help="Render game states. Incompatible with save-frames.", type=bool)

    parser.add_argument('--algo', default="TD3AQM", help="Algorithm to use", type=str)
    # parser.add_argument('--title', default="m2e5_ep30_ig0_Tw_gpu", help="Prefix of output files", type=str)
    parser.add_argument('--title', default="2403_b256_eval1_1ep", help="Prefix of output files", type=str)
    parser.add_argument('--cycle', default='plugin_1chtc_lt', help="Cycle to use", type=str)
    parser.add_argument('--env_train', default="ems-v3", help="Environment for training", type=str)
    parser.add_argument('--env_test', default="ems-v3", help="Environment for testing", type=str)
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument("--eval_freq", default=1653, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=1, type=int, help="How many episodes to use for evaluation during training.")
    parser.add_argument("--max_timesteps", default=200000, type=float)  # Max time steps to run environment for
    parser.add_argument('--batch-size', default=256, help='Minibatch size.', type=int)
    parser.add_argument("--cycle_test", default=False, help="driving cycle test", type=bool)
    # parser.add_argument("--DP2buffer", default=False, help="DP2buffer", type=bool)
    args = parser.parse_args()

    start = time.time()
    torch.set_num_threads(16)
    args.device = torch.device("cpu")
    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO: BEFORE RUNNING THIS SCRIPT, CHECK THE FOLLOWING:
    #   Whether you use the correct environment.
    #   Whether the `args.title` is correct with the parameters.
    #   Whether you want to set `args.cycle_test` to True.

    # args.DP2buffer = True
    args.cycle_test = True
    args.seed = 1
    run(args)

    # results = {}
    # for i in range(2, 6):
    #     args.seed = i
    #     best_return, dp_reward, seed = run(args)
    #     results[seed] = f'best: {best_return}, ' \
    #                     f'gap to {dp_reward}: {best_return / dp_reward - 1:.2%}'
    # for k, v in results.items():
    #     print(f'seed {k}: {v}')
    torch.cuda.empty_cache()
    print(f"Total time: {((time.time() - start) / 3600):.2f} hours")
