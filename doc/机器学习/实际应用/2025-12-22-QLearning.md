# Qlearning

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