import gym
from hyperparameter import *

# observation:an RGB image of the screen,which is an array of shape (210, 160, 3)
# action:Discrete(6)['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

env = gym.make(ENV_NAME)

print(env.unwrapped.get_action_meanings())
