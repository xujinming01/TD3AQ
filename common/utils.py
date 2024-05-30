import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


def pad_action(disc_act: int, conti_act: np.ndarray):
    """Concatenate discrete action and continuous actions."""
    if conti_act.shape == ():
        conti_act = np.array([conti_act])
    return np.concatenate([np.array([disc_act]), conti_act]).tolist()


class ReplayBuffer(object):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim,
                 all_parameter_action_dim, discrete_emb_dim, parameter_emb_dim,
                 max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.parameter_action = np.zeros((max_size, parameter_action_dim))
        self.all_parameter_action = np.zeros(
            (max_size, all_parameter_action_dim))

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim))
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.state_next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cpu")

    def add(self, state, discrete_action, parameter_action,
            all_parameter_action, discrete_emb, parameter_emb, next_state,
            state_next_state, reward, done):
        self.state[self.ptr] = state
        self.discrete_action[self.ptr] = discrete_action
        self.parameter_action[self.ptr] = parameter_action
        self.all_parameter_action[self.ptr] = all_parameter_action
        self.discrete_emb[self.ptr] = discrete_emb
        self.parameter_emb[self.ptr] = parameter_emb
        self.next_state[self.ptr] = next_state
        self.state_next_state[self.ptr] = state_next_state

        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.discrete_action[ind]).to(self.device),
            torch.FloatTensor(self.parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.all_parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.discrete_emb[ind]).to(self.device),
            torch.FloatTensor(self.parameter_emb[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.state_next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        """Update the progress bar"""
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


class ProgressBarManager(object):
    """
    Use "with" block, allowing for correct initialization and destruction
    """

    def __init__(self, total_timesteps):
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        """create a progress bar"""
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """destroy the progress bar"""
        self.pbar.close()

        return False


def least_squares(x, y):
    """linear regression by least squares method, y = a + bx"""
    n = len(x)
    sumx = 0
    sumx2 = 0
    sumy = 0
    sumy2 = 0
    sumxy = 0

    for i in range(0, n, 1):
        sumx += x[i]
        sumy += y[i]
        sumx2 += x[i] ** 2
        sumy2 += y[i] ** 2
        sumxy += x[i] * y[i]
    lxx = sumx2 - sumx ** 2 / n
    lxy = sumxy - sumx * sumy / n
    lyy = sumy2 - sumy ** 2 / n
    appro_b = lxy / lxx
    appro_a = sumy / n - appro_b * sumx / n
    R2 = lxy ** 2 / (lxx * lyy)
    print(f'R2 = {R2:.4f}')
    print(f'y (fuel_eq) = {appro_a:.4f} + {appro_b:.4f} * x (delta SOC)')

    data_range = max(x) - min(x)
    xnew = [min(x) - 0.2 * data_range, max(x) + 0.2 * data_range]
    result = [appro_a + appro_b * i for i in xnew]

    # # plt.style.use('seaborn-poster')
    # plt.rcParams['figure.figsize'] = [6.4, 3.2]
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times']
    # plt.rcParams['font.size'] = 12
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.scatter(x=x, y=y, color="#440154", marker='D')
    # plt.plot(xnew, result, color="#22a884")
    # plt.annotate(f'y = {appro_a:.4f} + {appro_b:.4f}x',
    #              xy=(0.6, 0.3), xycoords='axes fraction')
    # plt.xlabel(''r'$\Delta$ SOC')
    # plt.ylabel('Fuel consumption (kg)')
    # plt.tight_layout(pad=0.3)
    # # plt.savefig('fig_paper/fig-fit.png')
    # # plt.savefig('fig_paper/fig-fit.pdf')
    # plt.show()

    return appro_a, appro_b
