# save all hyperparameters
ENV_NAME = 'Pong-v0'    # 训练环境

# Agent parameters
BATCH_SIZE = 128    # 片段数量
LEARNING_RATE = 0.001   # 学习率
TUA = 0.001
GAMMA = 0.99

# Training parameters
EPISODE_NUM = 3000
EPS_INIT = 1
EPS_DECAY = 0.995
EPS_MIN = 0.05
MAX = 1500
NUM_FRAME = 2