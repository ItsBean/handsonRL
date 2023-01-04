import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import cliffwalk


class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出
        self.add_count = 0

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # 将数据加入buffer

    def add_2(self, state, action, reward, next_state, done):
        self.add_count += 1
        element = (state, action, reward, next_state, done)
        if element not in self.buffer:
            self.buffer.append(element)
        # self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        # print(state, action, reward, next_state, done,sep='\n')
        # assert 0, 'stop'
        return np.array(state), action, reward, np.array(next_state), done

    def sample2(self, batch_size):

        transitions = []
        for i in range(batch_size):
            transitions.append(random.choice(self.buffer))
        state, action, reward, next_state, done = zip(*transitions)
        # print(state, action, reward, next_state, done,sep='\n')

        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    def size2(self):
        return self.add_count


class Qnet(torch.nn.Module):
    ''' 一层隐层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.float)
        # print(x.ndimension())
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器，记录更新次数

    def take_action(self, state):  # epsilon greedy策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # print(state, torch.tensor(state, dtype=torch.float),torch.tensor(state, dtype=torch.float).size())
            state = torch.tensor(state, dtype=torch.float)
            action = self.q_net(state).argmax().item()
            # print(self.q_net(state),state,action)
        return action

    # 更新一次轨迹:
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值

        # another_max_q = self.q_net(next_states).max(1)[0].view(-1, 1)
        # print(max_next_q_values, another_max_q, sep="\n")
        # assert 0, str(torch.allclose(max_next_q_values, another_max_q))+" :wether they are the same"
        # assert 0,self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD目标
        # print(q_targets, q_values, sep="\n")
        # assert 0

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        # additional_states = torch.tensor([[3., i] for i in range(1, 12)], dtype=torch.float)
        # additional_q_targets = torch.tensor([[0.] * 4 for kk in range(1, 12)], dtype=torch.float)
        # print(self.q_net(additional_states), additional_q_targets, sep="\n")
        # print(F.mse_loss(self.q_net(additional_states), additional_q_targets))
        # assert 0, "wether they are the same"

        # additional_loss = torch.mean(
        #     F.mse_loss(self.q_net(additional_states), additional_q_targets))
        # self.optimizer.zero_grad()
        # additional_loss.backward()
        # self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络

        self.count += 1


def convert_index(index, col=12):
    i = index // col
    j = index % col
    return [i, j]


def action_to_chinese(action):
    if action == 0:
        return "上"
    elif action == 1:
        return "下"
    elif action == 2:
        return "左"
    elif action == 3:
        return "右"
    else:
        return "error"


lr = 2e-3
num_episodes = 500  # 500
hidden_dim = 128 * 2
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# env_name = 'CartPole-v1'
# env = gym.make(env_name)
env_name = 'CliffWalk'
env = cliffwalk.CliffWalkingEnv()
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = 2
action_dim = 4
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
return_list = []
file_rank = 0
# clean all content in q-charts.csv
with open("q-charts.csv", "w") as f:
    f.write("")
q_net_backpro_times = 0

for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = [3, 0]  # env.reset()[0]
            env.current_state = 3 * 12 + 0
            # assert 0,state
            done = False

            while not done:
                # cliffwalk.print_q_value(agent)
                action = agent.take_action(state)
                probability, next_state, reward, done = env.step(action)
                next_state = convert_index(next_state)
                episode_return += reward
                current_game_times = i * int(num_episodes / 10) + i_episode
                print(current_game_times, i_episode, q_net_backpro_times, state, action_to_chinese(action), next_state,
                      reward, done,
                      episode_return, replay_buffer.size(), agent.epsilon, sep=",")

                # replay_buffer.add_2(state, action, reward, next_state, done)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                # if episode_return < -1000:
                #     break
                # print(state, action, reward, next_state, done)
                # if replay_buffer.size() < 20:
                #     agent.epsilon = 0.99 * (20 - replay_buffer.size()) / 20 + 0.01
                # if replay_buffer.size() >= 20:# minimal_size:  # 当buffer数据数量超过一定值后，才进行Q网络训练
                if replay_buffer.size() > minimal_size:  # 当buffer数据数量超过一定值后，才进行Q网络训练
                    # agent.epsilon = 0.01
                    # if episode_return < -500:
                    #     agent.epsilon = ((episode_return + 500) / (episode_return)) * 0.99 + 0.01
                    # print(replay_buffer.buffer)
                    # assert 0, "see the buffer"

                    # b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample2(batch_size)
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}

                    agent.update(transition_dict)
                    q_net_backpro_times += 1
                    # show Q-chart of agent.q_net using plot:
                    q_chart_list = []
                    for index_row in range(env.nrow):
                        for index_col in range(env.ncol):
                            # assert 0,
                            q_chart_list.append(agent.q_net(torch.tensor([index_row, index_col],
                                                                         dtype=torch.float)).tolist())

                    transposed_list = list(map(list, zip(*q_chart_list)))
                    oneD_transposed = np.ravel(transposed_list)
                    # store oneD_transposed to q-charts.csv
                    # with open("q-charts.csv", "a") as f:
                    #     f.write(str(file_rank) + "," + ",".join(map(str, oneD_transposed)) + "\n")
                    # file_rank += 1

                    # print(transposed_list, oned_transposed, sep="\n")
                    # assert 0
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
print(replay_buffer.buffer)
cliffwalk.print_strategy(agent, env)
cliffwalk.print_q_value(agent)
cliffwalk.print_q_values(agent)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
