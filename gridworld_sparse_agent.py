import gym
import numpy as np  
import pygame
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common_evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

CANDY_PROB = 0.1
GOAL_REWARD = 1.0
DANGER_PENALTY = -2.0
TIME_PENALTY = -1.0

DEFAULT_SIZE = 5
WINDOW_SIZE = 512
RENDER_FPS = 4

class GridWorldDenseAgent(gym.Env):
    metadata = {"render_modes" : ["human", "rgb_array"], "render_fps" : RENDER_FPS}
    def __init__(self,render_mode=None,size=DEFAULT_SIZE,window_size=WINDOW_SIZE,candy_prob=CANDY_PROB):

        self.size = size
        self.window_size = window_size
        self.max_steps = size ** 2
        self.candy_prob = candy_prob
        self.current_step = 0
        
        self.observation_space = gym.spaces.Dict(
            {
                "agent" : gym.spaces.Box(0,size-1,shape=(2,),dtype=int)
                "target" : gym.spaces.Box(0,size-1,shape=(2,),dtype=int)
                "danger" : gym.spaces.Box(0,size-1,shape=(2,),dtype=int)
                "candy_map" : gym.spaces.MuiltiBinary([size,size])
            }
        )

        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1,0])
            1: np.array([0,1])
            2: np.array([-1,0])
            3: np.array([0,-1])
        }

        assert render_mode is  None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None 

    def get_obs(self):
        return {
            "agent": self.agent_location,
            "target": self.target_location,
            "danger": self.danger_location,
            "candy_map": self.candy_map,
        }

    def get_info(self):
        return {"distance": np.linalg.norm(self.agent_location - self.target_location,ord=1)}

    def reset(self,seed=None,options=None):

        self.current_step = 0

        self.agent_location = np.random.randint(0,self.size,size=2,dtype=int)

        self.target_location = self.agent_location
        while np.array_equal(self.target_location,self.agent_location):
            self.target_location = np.random.randint(0,self.size,size=2,dtype=int)
            
        self.danger_location = self.agent_location
        while np.array_equal(self.danger_location,self.target_location) \
            or np.array_equal(self.danger_location,self.agent_location):

            self.danger_location = np.random.randint(0,self.size,size=2,dtype=int)
        
        self.candy_map = np.random.choice(a=[True,False],size=(self.size,self.size),
                                        p=[self.candy_prob,1-self.candy_prob])


        self.candy_map[self.agent_location[0],self.agent_location[1]] = 0
        self.candy_map[self.target_location[0],self.target_location[1]] = 0
        self.candy_map[self.danger_location[0],self.danger_location[1]] = 0

        self.num_candies = np.sum(self.candy_map)
        self.candies_eaten = 0

        observation = self.get_obs()
        if self.render_mode == "human":
            self.render_frame()

        return observation

    def step(self,action):

        direction = self.action_to_direction[action]

        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)

        goal_reached = np.array_equal(self.agent_location,self.target_location)
        danger_reached = np.array_equal(self.agent_location,self.danger_location)

        time_limit_reached = self.current_step > self.max_steps

        candy_reached = False
        if self.candy_map[self.agent_location[0],self.agent_location[1]]:
            self.candies_eaten += 1
            self.candy_map[self.agent_location[0],self.agent_location[1]] = 0
            candy_reached = 0

        self.current_step += 1
        terminated = goal_reached or danger_reached or time_limit_reached


        candy_frac = self.candies_eaten / self.num_candies if self.num_candies > 0 else 0

        if terminated:
            reward = GOAL_REWARD if goal_reached else DANGER_PENALTY if danger_reached else TIME_PENALTY 
            reward += candy_frac
        else:
            reward = 0
        
        observation = self.get_obs()

        if self.render_mode == "human":
            self.render_frame()

        infos = self.get_info()
        if terminated:
            infos["goal_reached"] = goal_reached
            infos["danger_reached"] = danger_reached
            infos["candies_eaten"] = self.candies_eaten
            infos["num_candies"] = self.num_candies

        return observation, reward, terminated, infos
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size,self.window_size))

        canvas.fill((255,255,255))

        pix_square_size = self.window_size / self.size

        # Draw target
        pygame.draw_rect(
            canvas,
            (0,255,0),
            pygame.Rect(pix_square_size * self.target_location,
            (pix_square_size,pix_square_size),
        )

        # Draw danger
        pygame.draw_rect(
            canvas,
            (255,0,0),
            pygame.Rect(pix_square_size * self.danger_location,
            (pix_square_size,pix_square_size),
        )

        # Draw agent
        pygame.draw_rect(
            canvas,
            (0,0,255),
            pygame.Rect(pix_square_size * self.agent_location,
            (pix_square_size,pix_square_size),
        )

        # Draw candies
        for x in range(self.size):
            for y in range(self.size):
                if self.candy_map[x,y]:
                    pygame.draw_circle(
                        canvas,
                        (255,255,0),
                        (np.array([x,y]) + 0.5) * pix_square_size,
                        (pix_square_size / 3,
                    )

        # Draw grid
        for x in range(self.size + 1):
            pygame.draw_line(
                canvas,
                0,
                (0,pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

            pygame.draw_line(
                canvas,
                0,
                (pix_square_size * x,0),
                (pix_square_size * x,self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas,canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.meta_data["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas),axes=(1,0,2))
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()