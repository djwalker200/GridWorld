import gym
from gym import spaces
import numpy as np  
import pygame
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common_evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

class GridWrapper(VecEnvWrapper):
    def __init__(self,venv: VecEnv):
        VecEnvWrapper.__init__(self,venv)
    
    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
    
    def step_async(self,actions: np.ndarray) -> None:
        self.venv.step_async(actions)
    
    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
