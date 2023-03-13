import gym
from gym import spaces
import numpy as np  
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from sparse_agent import SparseAgent
from dense_agent import DenseAgent
from wrappers import BaseWrapper
from utils import custom_eval, train_and_evaluate, plot_results

if __name__ == '__main__':

    GRID_SIZE = 5
    TRAIN_STEPS = 500_000

    dense_env = DummyVecEnv([lambda : DenseAgent(size=GRID_SIZE)])
    dense_env = BaseWrapper(dense_env)
    dense_model = PPO('MultiInputPolicy',dense_env,verbose=0)

    sparse_env = DummyVecEnv([lambda : SparseAgent(size=GRID_SIZE)])
    sparse_env = BaseWrapper(sparse_env)
    sparse_model = PPO('MultiInputPolicy',sparse_env,verbose=0)

    step_counter, dense_results = train_and_evaluate(dense_model, max_steps=TRAIN_STEPS)
    step_counter, sparse_results = train_and_evaluate(sparse_model, max_steps=TRAIN_STEPS)

    results = {"Dense Agent" : dense_results, "Sparse Agent" : sparse_results}
    plot_results(results)

