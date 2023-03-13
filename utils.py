import gym
from gym import spaces
import numpy as np  
import pygame
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common_evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from gridworld_sparse_agent import GridWorldSparseAgent
from gridworld_dense_agent import GridWorldDenseAgent
from grid_wrapper import GridWrapper

def custom_eval(model,num_episodes=500):
    env = model.get_env()
    all_episode_rewards = []
    episode_lengths = []
    candies_eaten = 0
    total_candies = 0
    goals_reached = 0
    dangers_reached = 0
    
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        length = 0
        while not done:
            action, __ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            length += 1

        candies_eaten += info[0]['candies_eaten']
        total_candies += info[0]['num_candies']
        goals_reached += info[0]['goal_reached']
        dangers_reached += info[0]['danger_reached']

        all_episode_rewards.append(sum(episode_rewards))
        episode_lengths.append(length)

    mean_episode_reward = np.mean(all_episode_rewards)
    mean_length = np.mean(episode_lengths)

    candy_fraction = candies_eaten / total_candies if total_candies > 0 else 0
    goal_fraction = goals_reached / num_episodes if num_episodes > 0 else 0
    danger_fraction = dangers_reached / num_episodes if num_episodes > 0 else 0 

    statistics = {
        "mean_reward" : mean_episode_reward,
        "mean_length" : mean_length,
        "candy_frac" : candy_fraction,
        "goal_frac" : goal_fraction,
        "danger_frac" : danger_fraction,
    }

    return statistics

def train_and_evaluate(model, train_step=10_000, max_steps=250_000,log_steps=50_000, n_eval_episodes=500):

    current_steps = 0
    step_counter = []
    results = {
        "mean_reward" : [],
        "mean_length" : [],
        "candy_frac" : [],
        "goal_frac" : [],
        "danger_frac" : [],
    }

    while current_steps <= max_steps:

        step_counter.append(current_steps)

        statistics = custom_eval(model,n_eval_episodes)

        for key,value in statistics.items():
            results[key].append(value)

        if current_steps % log_steps == 0:
            print(f"Model after {current_steps} steps:")
            for key,value in statistics.items():
                print(f"{key} : {value}")
            print()
        
        model.learn(train_step)
        current_steps += train_step
        
    return step_counter, results

def plot_results(step_counter, results, outdir="Plots"):

    title_map = {
        "mean_reward" : "Mean Episode Reward",
        "mean_length" : "Mean Episode Length",
        "candy_frac" : "Fraction of Candies Reached",
        "goal_frac" : "Fraction of Episodes Reaching Goal",
        "danger_frac" : "Fraaction of Episodes Reaching Danger",
    }

    agent_names = list(results.keys())

    for key,metric in title_map.items():
        
        for name in agent_names:
            plt.plot(step_counter, results[name][key], label=name)

        plt.title(f"{metric} for Trained Agents")
        plt.xlabel("Number of Training Steps")
        plt.ylabel(title_map[key])
        plt.legend()

        if key in ["candy_frac", "goal_frac", "danger_frac"]:
            plt.ylim((0,1))
        if key == "mean_length":
            plt.ylim(bottom=0)
        
        plt.savefig(f"{outdir}/GridWorld_{key}.png")
        plt.clf()
        
    return

