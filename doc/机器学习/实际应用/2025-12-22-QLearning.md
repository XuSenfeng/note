# Qlearning

## 基础变量

+ state状态: 机器人所在的位置
+ state space状态空间: 所有的状态
+ action: 行动, 可以采取的行动
+ action space of a state: 行动空间
+ transition dynamics: 状态转移, 行动导致的状态变化
+ reward function: 奖励函数

笛卡尔积: 两个序列, 分别取一个数值相乘获取到的结果集合

> 这里可以使用S(状态)和A(不同行为的概率)的获取到△(S)或△(R)的笛卡尔积

policy: 策略, 是一个条件概率S->△(A), 在某个位置采取不同行动的概率

> π(a1|s1) = 0, π(a2|s1) = 1在s1的时候一定会a2

trajectory: 轨迹, state-action-reward chain, 一系列的action以及对应获取到的回报

![image-20251222094835800](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512220948892.png)

return: 回报, 沿着这个trajectory获取到的reward相加

折扣率discount: 使得未来额度收益率比当前的收益率小`γ∈[0, 1 )`

discounted return: 对未来t步的影响, 乘以一个折扣因子γ^t^ 

state value: 从一个state开始的各个trajectory求return期待值, 从V出发采取的是π策略, 可以获取到的结果的总和

<img src="https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512220956575.png" alt="image-20251222095603529" style="zoom:50%;" />

action value: 从状态S出发, 采取动作a, 遵循π策略, 未来的所有回报值期望总和

![image-20251222095743307](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512220957357.png)

最优策略: state value可以用于判断一个策略的好坏, 如果有一个π*的V大于其他所有的V, 这个就是最优的策略

最优state value V: 基于最优策略可以获取到的V

最优action value Q: 遵循策略π* 的时候, (s, a)的最大可能价值

> 强化学习指的是智能体在以最大化累计奖励为目标, 与环境交互过程中学习到最优的策略的过程
>
> 交互指的是智能体在不同的状态, 依据当前的策略采取不同的行动, 完成状态的转移, 同时环境向智能体返回即时的奖励
>
> 智能体根据移植的信息更新策略, 学习到最优的策论

![image-20251222101211365](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221012420.png)

![image-20251222101717589](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221017650.png)

某个策略的V是在当前位置采取不同action的概率乘以可以获取到的收益+下一个位置可以获取到的V的值乘以一个权重以后得值

![image-20251222140410670](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221404739.png)

这里获取到的V\*最大的实际就是Q\*最大的值

![image-20251222140555940](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221405984.png)

## Q-learning算法

实际就是不断地找到最大的Q\*(s, a), 实际是学习到一个s\*a的表格, 初始化一个Q表, 用于记录最优的Q的近似值, 不断的进行更新

更新公式: 新的预估值 = 旧的预估值 + 步长 x [ 目标 - 旧目标值 ] 

![image-20251222141048146](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221410201.png)

### 步骤

初始化Q表为全部为0

以一定的概率进行探索, 剩下的概率使用当前的最大值Q, 不断执行获得反馈, 不断执行动作, 获取环境的反馈, 下一个状态S以及实时奖励r

利用公式更新表格, 直到下一个状态到达终止条件, 开始下一个episode, 最终的目标是Q表收敛

```python
def reset_qtable(self):
    """重置Q表"""
    self.qtable=np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def choose_action(self, action_space, state, qtable):
        """依据一个随机数选择当前的决策"""
        explor_exploit_tradeoff = rng.uniform(0, 1)
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample() # 一个随机的行为
        else:
            # 使用当前的Q最大值里面的action
            max_ids = np.where(qtable[state:] == max(qtable[state:]))[0]
        	action=rng.choice(max_ids)
            
action = explorer.choose_action(
	action_space=env.action_space,state=state,
    qtable=learner.qtable
)
all_states.append(state)
all_actions.append(action)
# 获取环境的反馈
new_state,reward,terminated,truncated,info=env.step(action)
done=terminated or truncated
# 更新表
learn.qtable[state,action]= learner.update(state, action, reward, new_state)

def update(self, state, action, reward, new_state):
    """Q(s, a):= Q(s, a) + lr[R(s, a) + gamma * maxQ(s', a') - Q(s, a)]"""
    delta = (
    	reward + self.gamma * np.max(self.qtable[new_state,:]) - 
        self.qtable[state,action]
    )
    q_update=self.qtable[state, action] + self.learn_rate * delta
    return q_update
```

## DQN

在使用Qlearning的时候, 可以记录的值是有限的, 比较适合一些状态以及动作离散, 空间比较少的情况, 在实际的情况里面需要试试的获取当前的状态进行反应

这时候可以使用圣经网络等非线性的函数, 近似的表示Q值

### 预处理

在使用神经网络之前, 对数据进行预处理, 减少实际使用的运算量

+ 消除闪烁: 游戏里面有一些动画, 循环播放动画, 可以使用几帧里面取所有最大值的方式获取到更加稳定的图形
+ 提取亮通道: 不同的通道不影响游戏的结果的时候, 可以使用灰度处理
+ 缩放图像: 
+ 帧堆叠: 一帧不可以获取到运行的情况, 但是使用多的几帧就可以了

### 损失函数

![image-20251222150539760](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221505812.png)

在实际应用的时候会出现不稳定以及发散

+ 对观测序列存在相关性: 连续的时间数据是有害的
+ 对应Q的微小更新会改变策论, 更改数据分布: 多个动作的Q值相近的时候会影响策论
+ action value(Q值)和目标存在相关性: 在追逐的时候, 目标和角色使用相同的更新决策

解决: 使用两个模块

### 两个模块

#### 经验回放

设置一个重放内存区, replay memory, 里面存放的是四元组

这个区域的容量是N, N是一个可以定义的超参数, 每个四元组是: 状态, 动作, 奖励, 下一状态

> 通过从存储区里面随机抽样, 消除序列的相关性, 更有效的利用经验, 同时可以避免遗忘

#### 目标网络

固定target, 先把目标固定住, 让训练网络逼近目标值, 每格C步再从Q网络里面复制参数更新Q网络 

### 代码实现

[cleanrl/cleanrl/dqn_atari.py at master · vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py)

```python
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)
```

![image-20251222152537128](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202512221525203.png)

```python
q_network = QNetwork(envs).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict()) # 初始化参数为一样的
```

```python
epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
if random.random() < epsilon:
    # 进行探索
    actions = np.array([envs.single_action_space.sample()
                        for _ in range(envs.num_envs)])
else:
    # 使用经验值
    q_values = q_network(torch.Tensor(obs).to(device))
    actions = torch.argmax(q_values, dim=1).cpu().numpy()
```

```python
# update target network
if global_step % args.target_network_frequency == 0:
    for target_network_param, q_network_param in zip(target_network.parameters(),
                                                     q_network.parameters()):
        target_network_param.data.copy_(
            args.tau * q_network_param.data + (1.0 - args.tau) *
            target_network_param.data
        )
```

论文里面是直接把新的网络更新到另一个网络里面, 但是代码使用的是加权平均, 使用tau作为分配





