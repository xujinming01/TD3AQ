import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable
import heapq
from agents.agent import Agent
from agents.memory.memory import Memory
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise

import matplotlib.pyplot as plt


NEURON = 64


class QActor(nn.Module):
    """
    The code borrowed from:
    https://github.com/sfujim/TD3
    """

    def __init__(self, state_size, action_size, action_parameter_size):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        inputSize = self.state_size + self.action_parameter_size

        # create layers
        self.layers_1 = nn.ModuleList()
        self.layers_1.append(nn.Linear(inputSize, NEURON))
        self.layers_1.append(nn.Linear(NEURON, NEURON))
        self.layers_1.append(nn.Linear(NEURON, self.action_size))

        self.layers_2 = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        self.layers_2.append(nn.Linear(inputSize, NEURON))
        self.layers_2.append(nn.Linear(NEURON, NEURON))
        self.layers_2.append(nn.Linear(NEURON, self.action_size))

    def forward(self, state, action_parameters):

        x = torch.cat((state, action_parameters), dim=1)
        x_1 = F.relu(self.layers_1[0](x))
        x_1 = F.relu(self.layers_1[1](x_1))
        Q1 = self.layers_1[-1](x_1)

        x_2 = F.relu(self.layers_2[0](x))
        x_2 = F.relu(self.layers_2[1](x_2))
        Q2 = self.layers_1[-1](x_2)
        return Q1 , Q2

    def Q1 (self, state, action_parameters):
        x = torch.cat((state, action_parameters), dim=1)
        x_1 = F.relu(self.layers_1[0](x))
        x_1 = F.relu(self.layers_1[1](x_1))
        Q1 = self.layers_1[-1](x_1)

        return Q1

class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        # self.squashing_function = squashing_function

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        self.layers.append(nn.Linear(inputSize, NEURON))
        self.layers.append(nn.Linear(NEURON, NEURON))

        self.action_parameters_output_layer = nn.Linear(NEURON, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)


        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state

        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        action_params = self.action_parameters_output_layer(x)
        # action_params += self.action_parameters_passthrough_layer(state)  # TODO: no passthrough layer
        # if self.squashing_function:
        #     assert False  # scaling not implemented yet
        #     action_params = action_params.tanh()
        #     action_params = action_params * self.action_param_lim
        action_params = torch.tanh(action_params)
        return action_params


class TD3AQMAgent(Agent):
    """
    Parameterized TD3 DDAQ Agent.

    NOTE: Generally, self.action refers to discrete actions, and self.action_parameters refers to continuous actions.
    """

    NAME = "TD3AQM Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_param_class=ParamActor,
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 noise=False,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 use_normal_noise=False,
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=10,
                 policy_freq=2,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):

        super(TD3AQMAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_size = int(self.action_space.spaces.__len__() - 1)
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.action_parameter_size + 1)])
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high for i in range(1, self.action_parameter_size + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low for i in range(1, self.action_parameter_size + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.noise = noise
        self.use_ornstein_noise = use_ornstein_noise
        self.use_normal_noise = use_normal_noise

        print("discrete actions size:", self.num_actions,
              "continuous actions size:", self.action_parameter_size)
        # print(observation_space.shape,self.observation_space.shape[0])
        self.replay_memory = Memory(replay_memory_size, observation_space.shape, (1+self.action_parameter_size,), next_actions=False)
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size).to(device)
        self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)
        self.cost_his = []

        self.policy_freq = policy_freq
        self.total_it = 0

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "noise: {}\n".format(self.noise) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Normal Noise?: {}\n".format(self.use_normal_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer

        assert initial_weights.shape == passthrough_layer.weight.data.size()  # FIXME: the correct size?
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    # def _ornstein_uhlenbeck_noise(self, all_action_parameters):
    #     """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
    #     return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)

            all_action_parameters = self.actor_param.forward(state)

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = self.np_random.choice(self.num_actions)
                if self.noise and not (self.use_normal_noise or self.use_ornstein_noise):
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                             self.action_parameter_max_numpy))
            else:
                # select maximum action
                Q_a = self.actor.Q1(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)  # 返回最大离散动作的索引

            all_action_parameters = all_action_parameters.cpu().data.numpy()
            if self.noise:
                assert not (self.use_ornstein_noise and self.use_normal_noise), "Only one type of noise can be used at a time."
                if self.use_ornstein_noise:
                    noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001)  # theta=0.01, sigma=0.01)
                    all_action_parameters += noise.sample()
                elif self.use_normal_noise:
                    sizes = self.action_parameter_sizes
                    all_action_parameters = np.clip(all_action_parameters + np.random.normal(0, 0.02, sizes), -1, 1).squeeze()

        return action, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        """NOTE: this function is not working for now."""
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad
# P-DDPG
    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        # print(action)
        # a = np.concatenate(([act], all_action_parameters)).ravel()
        self._step += 1
        action = np.array(action)
        next_action = np.array(next_action)
        self._add_sample(state, action, reward, next_state, next_action, terminal=terminal)

        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1


    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        self.total_it += 1

        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(self.device)

        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            # target policy smoothing
            noise = torch.randn_like(pred_next_action_parameters) * 0
            # noise = noise.clamp(-0.5, 0.5)
            # pred_next_action_parameters = (pred_next_action_parameters + noise).clamp(-1, 1)
            pred_Q_a_1, pred_Q_a_2 = self.actor_target(next_states, pred_next_action_parameters)
            Qprime_1 = torch.max(pred_Q_a_1, 1, keepdim=True)[0].squeeze()
            Qprime_2 = torch.max(pred_Q_a_2, 1, keepdim=True)[0].squeeze()
            Qprime = torch.min(Qprime_1, Qprime_2)
            # Compute the TD error
            # target = rewards + (1 - terminals) * self.gamma * Qprime
            target = rewards + self.gamma * Qprime  # removed terminals
        # Compute current Q-values using policy network
        q_values_1, q_values_2 = self.actor(states, action_parameters)
        y_predicted_1 = q_values_1.gather(1, actions.view(-1, 1)).squeeze()
        y_predicted_2 = q_values_2.gather(1, actions.view(-1, 1)).squeeze()

        y_expected = target
        loss_Q = self.loss_func(y_predicted_1, y_expected) + self.loss_func(y_predicted_2, y_expected)

        self.actor_optimiser.zero_grad()   # 清空上一步残留参数值
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()   # 将参数更新值施加到 net 的 parameters 上

        # ----------------------delayed actor updates ----------------------
        if self.total_it % self.policy_freq == 0:
            with torch.no_grad():
                action_params = self.actor_param(states)

            action_params.requires_grad = True
            assert (self.weighted ^ self.average ^ self.random_weighted) or \
                   not (self.weighted or self.average or self.random_weighted)
            Q = self.actor.Q1(states, action_params)
            Q_val = Q
            # Q_loss = torch.mean(torch.sum(Q_val, 1))   #求和再取均值
            # select maximum Q value
            Q_val_max = torch.max(Q_val, 1, keepdim=True)[0].squeeze()
            Q_loss = torch.mean(torch.sum(Q_val_max))
            x = Q_loss
            x_np = x.data.cpu().numpy()
            self.cost_his.append(x_np)

            self.actor.zero_grad()
            Q_loss.backward()

            from copy import deepcopy
            delta_a = deepcopy(action_params.grad.data)
            # step 2 TODO: what is the purpose of this step?
            action_params = self.actor_param(Variable(states))

            delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

            if self.zero_index_gradients:
                delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

            out = -torch.mul(delta_a, action_params)
            self.actor_param.zero_grad()
            out.backward(torch.ones(out.shape).to(self.device))

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

            self.actor_param_optimiser.step()

            soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
            soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimiser.state_dict(), filename + "_actor_optimiser")

        torch.save(self.actor_param.state_dict(), filename + "_actor_param")
        torch.save(self.actor_param_optimiser.state_dict(), filename + "_actor_param_optimiser")
        print('Models saved successfully')

    def load_models(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimiser.load_state_dict(torch.load(filename + "_actor_optimiser"))

        self.actor_param.load_state_dict(torch.load(filename + "_actor_param"))
        self.actor_param_target = copy.deepcopy(self.actor_param)
        self.actor_param_optimiser.load_state_dict(torch.load(filename + "_actor_param_optimiser"))
        print('Models loaded successfully')

    # def plot_cost(self):
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his, c='y', label='td3aq_actor_loss')
    #     plt.legend(loc='best')
    #     plt.ylabel('loss')
    #     plt.xlabel('Training Steps')
    #     plt.show()

