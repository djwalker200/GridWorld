import numpy as np  
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class BaseWrapper(VecEnvWrapper):
    def __init__(self,venv: VecEnv):
        VecEnvWrapper.__init__(self,venv)
    
    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs
    
    def step_async(self,actions: np.ndarray) -> None:
        self.venv.step_async(actions)
    
    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
