import gym

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", render_mode="human")
# env = env.unwrapped  # 解封装才能访问状态转移矩阵P
# env.render()  # 环境渲染，通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
i = 0
env.reset()
while i < 100:
    env.render()
    i = int(input("input direction:"))
    if (i == 100):
        break
    # action = input("input " + _.__str__() + " :")  # env.action_space.sample()
    action = i
    if (action == ""): continue
    action = int(action)
    observation, reward, done, info, unknown = env.step(action)  # take a random action
    print(action, observation, reward, info, done, sep=" , ")
    if done and reward == 0:
        env.reset()
    if done and reward == 1:
        print("you win!!!!")
        env.reset()
        # 0: LEFT
        #
        # 1: DOWN
        #
        # 2: RIGHT
        #
        # 3: UP
# for s in env.P:
#     for a in env.P[s]:
#         for s_ in env.P[s][a]:
#             if s_[2] == 1.0: # 获得奖励为1，代表是终点
#                 ends.add(s_[1])
#             if s_[3] == True:
#                 holes.add(s_[1])
# holes = holes - ends
# print("冰洞的索引:", holes)
# print("终点的索引", ends)
#
# for a in env.P[14]:  # 查看终点左边一格的状态转移信息
#     print(env.P[14][a])
