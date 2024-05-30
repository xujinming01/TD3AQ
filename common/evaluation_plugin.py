import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.io as sio
from gym.spaces import Box, Discrete, Tuple

from common.utils import pad_action, least_squares
from agents.hppo_noshare import PPO

dT = 1  # [s] NOTE: be consistent with the environment
# SAVE_FIG = True
SAVE_FIG = False
LABEL_RL = 'RL'
COLOR_RL = 'tab:red'
LW = 1  # linewidth


def from_ppo(agent, env, rl_dict, soc_init):
    """Get results from hybrid-ppo.
    Note the method is outdated, find the latest version in `from_cdrl`.
    """
    steps = 0
    rewards = 0.

    state = env.reset()
    state = env.reset_zero(soc_init=soc_init)
    env.evaluate_on()
    terminal = False

    while not terminal:
        steps += 1
        state = np.array(state, dtype=np.float32, copy=False)
        prob_discrete_action, discrete_action, parameter_action = agent.select_action(
            state, is_test=True)
        action = pad_action(discrete_action, parameter_action)
        state, reward, terminal, result = env.step(action)
        for k, v in result.items():
            rl_dict[k].append(v)
        rewards += reward

    return rl_dict, steps, rewards

def from_cdrl(agent, env, rl_dict, soc_init, penalty, episodes):
    """Get results from continuous-discrete RL."""
    returns = []
    episode_steps = []

    for _ in range(episodes):

        agent.epsilon = 0.
        agent.noise = False

        env.reset()
        state = env.reset_zero(get_soc_init(soc_init))
        env.evaluate_on()
        terminal = False
        steps = 0
        rewards = 0.
        while not terminal:
            steps += 1
            state = np.array(state, dtype=np.float32, copy=False)
            discrete, continuous = agent.act(state)
            action = pad_action(discrete, continuous)
            state, reward, terminal, result = env.step(action)
            for k, v in result.items():
                if k in ['p_Tm', 'p_w', 'p_gear', 'p_clutch', 'p_soc', 'p_soc_sum']:
                    penalty[k] += v
                else:
                    rl_dict[k].append(v)
            rewards += reward
        # print(f'Uncompensated fuel consumption: {-rewards:.4f} kg\n')
        returns.append(rewards)
        episode_steps.append(steps)

    mean_return = np.mean(returns)
    mean_steps = np.mean(episode_steps)

    eval_print(episodes, mean_return, mean_steps, penalty)

    return rl_dict, mean_steps, mean_return


def from_dqn(agent, env, results, soc_init, penalty, episodes):
    """Get results from discrete RL, not just for DQN, no need to pad action.
    The penalty and episodes are not implemented now.
    """
    steps = 0
    rewards = 0.

    agent.dqn.reset_noise_zero()
    agent.dqn_target.reset_noise_zero()
    env.reset()
    state = env.reset_zero(soc_init)
    env.evaluate_on()
    terminal = False
    while not terminal:
        steps += 1
        state, reward, terminal, result = env.step(agent.act(state))
        for k, v in result.items():
            if k in ['p_Tm', 'p_w', 'p_gear', 'p_clutch', 'p_soc', 'p_soc_sum']:
                penalty[k] += v
            else:
                results[k].append(v)
        rewards += reward
    # print(f'Uncompensated fuel consumption: {-rewards:.4f} kg\n')

    return results, steps, rewards

def from_td3(agent, env, results, soc_init, penalty, episodes):
    """Get results from continuous RL. (PA-TD3)
    penalty and episodes are not implemented now.
    """
    steps = 0
    rewards = 0.
    
    agent.policy_noise = 0.

    env.reset()
    state = env.reset_zero(soc_init)
    env.evaluate_on()
    terminal = False
    while not terminal:
        steps += 1
        action = agent.select_action(state)
        state, reward, terminal, result = env.step(action)
        for k, v in result.items():
            if k in ['p_Tm', 'p_w', 'p_gear', 'p_clutch', 'p_soc', 'p_soc_sum']:
                penalty[k] += v
            else:
                results[k].append(v)
        rewards += reward
    # print(f'Uncompensated fuel consumption: {-rewards:.4f} kg\n')

    return results, steps, rewards


def rl(agent, env, rl_dict, penalty, episodes):
    """Get results and equivalent fuel consumption from different methods."""
    if type(env.action_space) is Tuple:
        if isinstance(agent, PPO):
            eval_method = from_ppo
        else:
            eval_method = from_cdrl
    elif type(env.action_space) is Discrete:
        eval_method = from_dqn
    elif type(env.action_space) is Box:
        eval_method = from_td3
    else:
        raise ValueError(f'Unknown action space.')

    if episodes == 1:  # single episode, fixed SOC_0
        SOC_0 = 0.9
    else:  # multiple episodes, random SOC_0
        SOC_0 = 'random'

    import time
    start = time.time()
    RESULTS, steps, REWARDS = eval_method(agent, env, rl_dict, SOC_0, penalty, episodes)
    print(f'Elapsed time: {time.time() - start:.2f} s, step time: {(time.time() - start) / (steps * episodes) * 1000:.2f} ms')

    return RESULTS, steps, REWARDS


def compare_plot(fig_name, x_axis, y_rl, y_dp, x_label, y_label, title,
                 save_path):
    """Plot a single plot between RL and DP."""
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(6.4, 3.2)
    ax.plot(x_axis, y_rl, label=LABEL_RL, color=COLOR_RL, linewidth=LW)
    ax.plot(x_axis, y_dp, label='DP', color='black', alpha=1, linewidth=LW)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(title)
    ax.legend()
    ax.grid(color='lightgray', linewidth=LW / 2)
    plt.tight_layout(pad=0.3)
    if SAVE_FIG:
        fig.savefig(f'{save_path}/{fig_name}.png', dpi=100)
        fig.savefig(f'{save_path}/{fig_name}.pdf')
    plt.show()

    return fig


def compare2dp(m_rl, m_dp, save_path):
    """Compare the results with the DP solution."""

    figs = []
    # plt.style.use('seaborn-notebook')
    wall_time = len(m_rl['Tw'][0]) * dT
    x = np.arange(0, wall_time, dT)

    # # with Te_org
    # fig_Te_org, ax = plt.subplots()
    # ax.figure.set_size_inches(6.4, 3.2)
    # ax.plot(x, m_rl['Te'][0], label='Te', color=COLOR_RL, linewidth=LW)
    # ax.plot(x,
    #         m_rl['Te_org'][0],
    #         label='Te_org',
    #         color='tab:purple',
    #         linewidth=LW)
    # ax.plot(x, m_dp['Te'][0], label='DP', color='black', alpha=1, linewidth=LW)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Engine torque [Nm]')
    # ax.set_title('Engine torque')
    # ax.legend()
    # ax.grid(color='lightgray', linewidth=LW / 2)
    # plt.tight_layout(pad=0.3)
    # if SAVE_FIG:
    #     fig_Te_org.savefig(f'{save_path}/Te_org.png', dpi=100)
    #     fig_Te_org.savefig(f'{save_path}/Te_org.pdf')
    # plt.show()
    #
    # without Te_org
    fig_Te, ax = plt.subplots()
    ax.figure.set_size_inches(6.4, 3.2)
    ax.plot(x, m_rl['Te'][0], label='RL', color=COLOR_RL, linewidth=LW)
    ax.plot(x, m_dp['Te'][0], label='DP', color='black', alpha=1, linewidth=LW)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Engine torque [Nm]')
    ax.set_title('Engine torque')
    ax.legend()
    ax.grid(color='lightgray', linewidth=LW / 2)
    plt.tight_layout(pad=0.3)
    if SAVE_FIG:
        fig_Te.savefig(f'{save_path}/Te.png', dpi=100)
        fig_Te.savefig(f'{save_path}/Te.pdf')
    plt.show()

    fig_Tm = compare_plot('Tm', x, m_rl['Tm'][0], m_dp['Tm'][0], 'Time [s]',
                          'Motor torque [Nm]', 'Motor torque', save_path)
    fig_Tb = compare_plot('Tb', x, m_rl['Tb'][0], m_dp['Tb'][0], 'Time [s]',
                          'Brake torque [Nm]', 'Brake torque', save_path)
    fig_soc = compare_plot('soc', x, m_rl['soc'][0], m_dp['soc'][0],
                           'Time [s]', 'SOC [%]', 'SOC', save_path)
    # fig_w = compare_plot('w', x, m_rl['w'][0], m_dp['wm'][0], 'Time [s]',
    #                      'Angular velocity [rad/s]', 'Angular velocity',
    #                      save_path)
    fig_ig = compare_plot('ig', x, m_rl['gear'][0] + 1, m_dp['gear'][0],
                          'Time [s]', 'Gear', 'Gear', save_path)

    fig_clutch, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, m_rl['clutch'][0], label='RL', color=COLOR_RL, linewidth=LW)
    ax[1].plot(x, m_dp['clutch'][0], label='DP', color='black', alpha=1,
               linewidth=LW)
    ax[1].set_xlabel('Time [s]')
    ax[0].set_yticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[0].set_title('Clutch')
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')
    plt.tight_layout(pad=0.3)
    if SAVE_FIG:
        fig_clutch.savefig(f'{save_path}/clutch.png', dpi=100)
        fig_clutch.savefig(f'{save_path}/clutch.pdf')
    plt.show()

    # fig_r, ax = plt.subplots()
    # ax.figure.set_size_inches(18, 6)
    # ax.plot(m_rl['reward'][0], label=LABEL_RL, color=COLOR_RL, linewidth=LW)
    # ax.plot(-m_dp['cost'][0], label='DP', color='black', alpha=1, linewidth=LW)
    # ax.set_xlabel('Timesteps')
    # ax.set_ylabel('Reward')
    # ax.set_title('Reward')
    # ax.legend()
    # ax.grid(color='lightgray', linewidth=LW / 2)
    # plt.tight_layout(pad=0.3)
    # if SAVE_FIG:
    #     fig_r.savefig(f'{save_path}/reward.png', dpi=100)
    #     fig_r.savefig(f'{save_path}/reward.pdf')
    # plt.show()

    # fig_ap, ax = plt.subplots()
    # ax.figure.set_size_inches(18, 6)
    # ax.plot(x, m_dp['ap'][0], label='a', color='tab:blue', linewidth=LW)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Acceleration [m/s^2]')
    # ax.set_title('Acceleration')
    # ax.grid(color='lightgray', linewidth=LW / 2)
    # plt.tight_layout(pad=0.3)
    # if SAVE_FIG:
    #     fig_ap.savefig(f'{save_path}/ap.png', dpi=100)
    #     fig_ap.savefig(f'{save_path}/ap.pdf')
    # plt.show()

    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times']
    # plt.rcParams['font.size'] = 12
    # fig_vp, ax = plt.subplots()
    # ax.figure.set_size_inches(6.4, 2.4)
    # # set xlim to the length of the cycle
    # ax.set_xlim([0, len(m_dp['vp'][0]) * dT])
    # ax.set_ylim([0, max(m_dp['vp'][0]) * 1.1])
    # ax.grid(color='lightgray', linewidth=LW / 2, axis='y', linestyle='--')
    # ax.plot(x, m_dp['vp'][0], label='v', color='#3c46ff', linewidth=LW)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Velocity [m/s]')
    # # ax.set_title('Velocity')
    # plt.tight_layout(pad=0.3)
    # if SAVE_FIG:
    #     fig_vp.savefig(f'{save_path}/fig-vp.png', dpi=100)
    #     fig_vp.savefig(f'{save_path}/fig-vp.pdf')
    # plt.show()

    # fig_Tw, ax = plt.subplots()
    # ax.figure.set_size_inches(18, 6)
    # ax.plot(x, m_rl['Tw'][0], label='Tw', color='tab:blue', linewidth=LW)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Wheel torque [Nm]')
    # ax.set_title('Wheel torque')
    # ax.legend()
    # ax.grid(color='lightgray', linewidth=LW / 2)
    # plt.tight_layout(pad=0.3)
    # if SAVE_FIG:
    #     fig_Tw.savefig(f'{save_path}/Tw.png', dpi=100)
    #     fig_Tw.savefig(f'{save_path}/Tw.pdf')
    # plt.show()

    # figs.extend([
    #     fig_Te_org, fig_Te, fig_Tm, fig_Tb, fig_ig, fig_soc, fig_w, fig_r,
    #     fig_ap, fig_vp, fig_Tw
    # ])
    #
    # return figs


def cycle_test(env, agent, save_path, cyc, r_dp, episodes=1):
    rl_dict = {
        'Tw': [],
        'Te': [],
        'Tm': [],
        'Tb': [],
        'gear': [],
        'gear_diff': [],
        'w': [],
        'soc': [],
        'reward': [],
        'Tm_org': [],
        'Te_org': [],
        'clutch': [],
        'shift': [],
        'currency': [],
    }
    penalty = {
        'p_Tm': 0.,
        'p_w': 0.,
        'p_gear': 0.,
        'p_clutch': 0.,
        'p_soc': 0.,
        'p_soc_sum': 0.,
    }

    results, steps, rewards = rl(agent, env, rl_dict, penalty, episodes)
    print(f'{cyc}: steps={steps}, equivalent CNY cost={rewards:.4f}, '
          f'gap: {rewards / r_dp - 1:.2%}')

    # save_cycle_path = f'{save_path}/{cyc}_{dT}s'
    # os.makedirs(save_cycle_path, exist_ok=True)
    # sio.savemat(f'{save_cycle_path}/_results_{cyc}_{dT}s.mat', results)
    #
    # mat_rl = sio.loadmat(f'{save_cycle_path}/_results_{cyc}_{dT}s.mat')
    # mat_dp = sio.loadmat(f'./results_{cyc}')  # NOTE: DP results
    # soc_final_dp = mat_dp['soc'][0][-1]
    # print(f"SOC final: DP={soc_final_dp:.4f}, "
    #       f"RL={mat_rl['soc'][0][-1]:.4f}")
    # compare2dp(mat_rl, mat_dp, save_cycle_path)

    return steps, rewards


def get_soc_init(soc_init):
    if soc_init == 'random':
        return npr.uniform(0.35, 0.9)
    else:
        return soc_init


def eval_print(episodes, mean_return, mean_steps, penalty):
    print(
        f'Evaluate on {episodes} episodes, mean steps: {mean_steps}, mean return: {mean_return:.4f},\n'
        f'Mean penalty:\n'
        f'  p_Tm: {penalty["p_Tm"] / episodes:.2f}, \n'
        f'  p_w: {penalty["p_w"] / episodes:.2f},\n'
        f'  p_gear: {penalty["p_gear"] / episodes:.2f},\n'
        f'  p_clutch: {penalty["p_clutch"] / episodes:.2f},\n'
        f'  p_soc: {penalty["p_soc"] / episodes:.2f},\n'
        f'  p_soc_sum: {penalty["p_soc_sum"] / episodes:.4f}'
    )


if __name__ == '__main__':
    # NOTE: change the path to the directory where the data is saved
    env_version = 'ems-v2'
    algo_flag = 'td3aqm_m2e5_ep30_ig0_Tw'
    seed = '1'
    cycle = 'chtc_lt_1s'
    data_path = f'../evaluations/{env_version}_{algo_flag}/{algo_flag}_{seed}'

    data_rl = sio.loadmat(f'{data_path}/{cycle}/_results_{cycle}.mat')
    data_dp = sio.loadmat('../results_chtc_lt.mat')
    compare2dp(data_rl, data_dp, data_path)
