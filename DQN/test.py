import gym
import time

env = gym.make('Breakout-v0')
for i_episode in range(5):
    observation = env.reset()
    total_reward = 0
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # print(info)
        if done:
            print("reward {}".format(total_reward))
            print(info)
            print(done)
            print("Episode {} finished after {} steps".format(i_episode+1, t+1))
            break
        time.sleep(0.05)
env.close()