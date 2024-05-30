import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.io as sio
from gym.spaces import Box, Discrete, Tuple

from common.utils import pad_action, least_squares

dT = 1  # [s] NOTE: be consistent with the environment
# SAVE_FIG = True
SAVE_FIG = False
LABEL_RL = 'RL'
COLOR_RL = 'tab:red'
LW = 1  # linewidth


def from_cdrl(agent, env, rl_dict, soc_init):
    """Get results from continuous-discrete RL."""
    steps = 0
    rewards = 0.

    agent.epsilon = 0.
    agent.noise = False

    env.reset()
    state = env.reset_zero(soc_init)
    env.evaluate_on()
    terminal = False
    while not terminal:
        steps += 1
        state = np.array(state, dtype=np.float32, copy=False)
        discrete, continuous = agent.act(state)
        action = pad_action(discrete, continuous)
        state, reward, terminal, result = env.step(action)
        for k, v in result.items():
            rl_dict[k].append(v)
        rewards += reward
    print(f'Uncompensated fuel consumption: {-rewards:.4f} kg\n')

    return rl_dict, steps, rewards


def from_dqn(agent, env, results, soc_init):
    """Get results from discrete RL, not just for DQN, no need to pad action."""
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
            results[k].append(v)
        rewards += reward
    print(f'Uncompensated fuel consumption: {-rewards:.4f} kg\n')

    return results, steps, rewards

def from_td3(agent, env, results, soc_init):
    """Get results from continuous RL. (PA-TD3)"""
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
            results[k].append(v)
        rewards += reward
    print(f'Uncompensated fuel consumption: {-rewards:.4f} kg\n')

    return results, steps, rewards


def rl(agent, env, rl_dict, soc_final_dp, method):
    """Get results and equivalent fuel consumption from different methods.
    Note the default SOC_0 is 0.6, soc_max is 0.8, soc_min is 0.4."""
    if method == 'cdrl':
        eval_method = from_cdrl
    elif method == 'dqn':
        eval_method = from_dqn
    elif method == 'td3':
        eval_method = from_td3
    else:
        raise ValueError(f'Unknown method {method}')
    SOC_0 = 0.6
    RESULTS, steps, REWARDS = eval_method(agent, env, rl_dict, SOC_0)
    DELTA_SOC = RESULTS['soc'][-1] - SOC_0

    # collect 6 groups of random initial SOC
    # SOC_0 to soc_max
    soc_0_list = npr.uniform(SOC_0, 0.8, 3)
    # soc_min to SOC_0
    soc_0_list = np.append(soc_0_list, npr.uniform(0.4, SOC_0, 3))
    delta_soc_list = [DELTA_SOC]
    fuel_list = [-REWARDS]  # reward = - fuel
    dup_rl_dict = deepcopy(rl_dict)
    import time
    start = time.time()
    for soc_init in soc_0_list:
        results, _, rewards = eval_method(agent, env, dup_rl_dict, soc_init)
        delta_soc_list.append(results['soc'][-1] - soc_init)
        fuel_list.append(-rewards)
    print(f'Evaluation time: {time.time() - start:.2f} s')
    print(f'Random initial SOC: {soc_0_list}')
    print(f'delta_soc_list: {[round(s, 6) for s in delta_soc_list]}')
    print(f'fuel_list: {[round(f, 6) for f in fuel_list]}')
    # fuel = a + b * delta_soc,
    a, b = least_squares(x=delta_soc_list, y=fuel_list)
    # equivalent fuel consumption
    DELTA_SOC_DP = soc_final_dp - SOC_0
    FUEL_EQ = a + b * DELTA_SOC_DP

    return RESULTS, steps, FUEL_EQ


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
    # # without Te_org
    # fig_Te, ax = plt.subplots()
    # ax.figure.set_size_inches(6.4, 3.2)
    # ax.plot(x, m_rl['Te'][0], label='Te', color=COLOR_RL, linewidth=LW)
    # ax.plot(x, m_dp['Te'][0], label='DP', color='black', alpha=1, linewidth=LW)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Engine torque [Nm]')
    # ax.set_title('Engine torque')
    # ax.legend()
    # ax.grid(color='lightgray', linewidth=LW / 2)
    # plt.tight_layout(pad=0.3)
    # if SAVE_FIG:
    #     fig_Te.savefig(f'{save_path}/Te.png', dpi=100)
    #     fig_Te.savefig(f'{save_path}/Te.pdf')
    # plt.show()

    # fig_Tm = compare_plot('Tm', x, m_rl['Tm'][0], m_dp['Tm'][0], 'Time [s]',
    #                       'Motor torque [Nm]', 'Motor torque', save_path)
    # fig_Tb = compare_plot('Tb', x, m_rl['Tb'][0], m_dp['Tb'][0], 'Time [s]',
    #                       'Brake torque [Nm]', 'Brake torque', save_path)
    fig_soc = compare_plot('soc', x, m_rl['soc'][0], m_dp['soc'][0],
                           'Time [s]', 'SOC [%]', 'SOC', save_path)
    # fig_w = compare_plot('w', x, m_rl['w'][0], m_dp['wm'][0], 'Time [s]',
    #                      'Angular velocity [rad/s]', 'Angular velocity',
    #                      save_path)
    fig_ig = compare_plot('ig', x, m_rl['gear'][0] + 1, m_dp['ig'][0],
                          'Time [s]', 'Gear', 'Gear', save_path)

    # fig_r, ax = plt.subplots()
    # ax.figure.set_size_inches(18, 6)
    # ax.plot(m_rl['reward'][0], label=LABEL_RL, color=COLOR_RL, linewidth=LW)
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


def cycle_test(env, agent, save_path, cyc, r_dp):
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
        'Te_org': []
    }

    mat_dp = sio.loadmat(f'./results_{cyc}')  # NOTE: DP results
    soc_final_dp = mat_dp['soc'][0][-1]
    if type(env.action_space) == Tuple:
        results, steps, rewards = rl(agent, env, rl_dict, soc_final_dp, 'cdrl')
    elif type(env.action_space) == Discrete:
        results, steps, rewards = rl(agent, env, rl_dict, soc_final_dp, 'dqn')
    elif type(env.action_space) == Box:
        results, steps, rewards = rl(agent, env, rl_dict, soc_final_dp, 'td3')
    else:
        raise ValueError('Unknown action space.')
    print(f'{cyc}: steps={steps}, equivalent fuel consume={rewards:.4f}, '
          f'gap: {rewards / (- r_dp) - 1:.2%}')

    save_cycle_path = f'{save_path}/{cyc}_{dT}s'
    os.makedirs(save_cycle_path, exist_ok=True)
    sio.savemat(f'{save_cycle_path}/_results_{cyc}_{dT}s.mat', results)

    mat_rl = sio.loadmat(f'{save_cycle_path}/_results_{cyc}_{dT}s.mat')
    print(f"SOC final: DP={soc_final_dp:.4f}, "
          f"RL={mat_rl['soc'][0][-1]:.4f}")
    compare2dp(mat_rl, mat_dp, save_cycle_path)

    return steps, rewards


if __name__ == '__main__':
    # NOTE: change the path to the directory where the data is saved
    env_version = 'ems-v2'
    algo_flag = 'td3aqm_m2e5_freq3k'
    seed = '2'
    cycle = 'chtc_lt_1s'
    data_path = f'../evaluations/{env_version}_{algo_flag}/{algo_flag}_{seed}'

    data_rl = sio.loadmat(f'{data_path}/{cycle}/_results_{cycle}.mat')
    data_dp = sio.loadmat('../results_chtc_lt.mat')
    compare2dp(data_rl, data_dp, data_path)
