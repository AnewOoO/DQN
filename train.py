import gym
from hyperparameter import *
from agent import *

def train(env, agent, num_episode, eps_init, eps_min,eps_decay, max_t, num_frame=2, constant=0):
    reward_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):

        episode_reward = 0
        done = False
        frame = env.reset()
        frame = preprocess(frame, constant)
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(frame)
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)
        t = 0

        while not done and t < max_t:
            env.render()
            t += 1
            action = agent.action_choose(eps, state)
            frame, reward, done, _ = env.step(action + 1)
            frame = preprocess(frame, constant)
            state_deque.append(frame)
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            agent.memory.append((state, action, reward, next_state, done))

            if t % 5 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()

            state = next_state.copy()
            episode_reward += reward

        reward_log.append(episode_reward)
        average_log.append(np.mean(reward_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episode_reward, average_log[-1]), end='')
        if t % 100 == 0:
            print()

        eps = max(eps * eps_decay, eps_min)

    return reward_log



def preprocess(image, constant):
    image = image[34:194, :, :] # 160, 160, 3   长 宽 通道
    image = np.mean(image, axis=2, keepdims=False) # 160, 160  pytorch里面channel在最前面 取平均值
    image = image[::2, ::2] # 80, 80
    image = image/256 #归一化
    image = image - constant/256# remove background 84 去背景可padding
    return image

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    agent = Agent(NUM_FRAME, 3, BATCH_SIZE, LEARNING_RATE, GAMMA)
    rewards_log = train(env, agent, EPISODE_NUM, EPS_INIT, EPS_MIN, EPS_DECAY, MAX, NUM_FRAME, 90)
    np.save('{}_rewards.npy'.format(ENV_NAME), rewards_log)
    torch.save(agent.Q_local.state_dict(),'{}_weights.pth'.format(ENV_NAME))

