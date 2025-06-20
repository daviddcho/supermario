import torch
from torch import tensor, optim
import torch.nn as nn
import cv2
import math
import numpy as np
import time
from tqdm import tqdm

from model import DuelingDQN
from wrappers import *
from experience_replay import ExperienceReplay, Experience
from logger import Logger
import hyperparameters as hp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

class DQNAgent:
  def __init__(self):
    self.env = make_mario("SuperMarioBros-1-1-v3", COMPLEX_MOVEMENT)
  
    self.n_episodes = hp.N_EPISODES
    self.alpha = hp.ALPHA 
    self.gamma = hp.GAMMA 
    self.decay = hp.DECAY 
    self.memory_size = hp.MEMORY_SIZE 
    self.batch_size = hp.BATCH_SIZE 
    self.min_exp = hp.MIN_EXP 
    self.target_update_freq = hp.TARGET_UPDATE_FREQ 

    self.input_shape = self.env.observation_space.shape
    self.n_actions = self.env.action_space.n

    self.model = DuelingDQN(self.input_shape, self.n_actions).to(device)
    self.target_model = DuelingDQN(self.input_shape, self.n_actions).to(device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
    self.loss_function = nn.MSELoss() 

    self.replay_memory = ExperienceReplay(self.memory_size)
    self.logger = Logger()

  def load(self, file):
    self.model.load_state_dict(torch.load(file, map_location=torch.device(device)))
    self.target_model.load_state_dict(self.model.state_dict())
    self.model.eval()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

  def update_model(self, minibatch):
    """
    Calculate Mean Square Error (MSE) between actual values and expected values from Deep Q-Network
    Then perform stochasitc gradient descent
    """
    states, actions, rewards, next_states, _ = minibatch

    states = tensor(states).to(device) 
    next_states = tensor(next_states).to(device)
    actions = tensor(actions).to(device) 
    rewards = tensor(rewards).to(device) 

    Q = self.model(states)
    target_Q = self.target_model(next_states)
   
    # Get Q-values of actions taken for each state
    Q = Q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    target_Q = target_Q.amax(1)

    # Perform SGD
    td_target = target_Q * self.gamma + rewards
    loss = self.loss_function(Q, td_target)
    self.optimizer.zero_grad() 
    loss.backward() 
    self.optimizer.step() 

    return Q.mean().item(), loss.item()
    
  def select_action(self, state): 
    """
    Picks action with epsilon-greedy policy
    """
    # Pick best action with DQN 
    state = tensor(np.float32(state)).unsqueeze(0).to(device)
    with torch.no_grad():
      q_value = self.model(state)
      action = q_value.argmax().item()
    return action

  def train(self):
    """
    Deep Double Q-Learning with Experience Replay
    """
    step = 0
    for episode in tqdm(range(self.n_episodes)):
      distance = 0

      current_state = self.env.reset()
      done = False 
      while not done:
        step += 1

        action = self.select_action(current_state)
        next_state, reward, done, info = self.env.step(action)
        
        experience = Experience(current_state, action, reward, next_state, done)
        self.replay_memory.push(experience)
        current_state = next_state 

        if (self.replay_memory.length() < self.min_exp):
          continue

        # Sync target with main 
        if (step % self.target_update_freq) == 0:
          self.target_model.load_state_dict(self.model.state_dict())

        if (step % 4) == 0:
          minibatch = self.replay_memory.sample_batch(self.batch_size)
          q_value, loss = self.update_model(minibatch)
        else:
          q_value, loss = None, None

        distance = max(info['x_pos'], distance)
        self.logger.log_step(reward, distance, q_value, loss)
        
      self.logger.log_episode()
      if (episode % self.target_update_freq) == 0 and episode != 0:
        torch.save(self.model.state_dict(), "pretrained_models/model_%d.pth" % episode)
        self.logger.record(episode, 1, step) 
    
  def play(self):
    """
    When you want to watch Mario play
    """
    def ascii_render(rgb):
      grayscale = rgb.mean(axis=2)  # average over RGB
      chars = " .:-=+*#%@"
      step = 256 // len(chars)
      frame = ""
      for row in grayscale[::8]:  # downsample vertically
        line = "".join(chars[min(len(chars) - 1, int(val) // step)] for val in row[::6])
        frame += line + "\n"
      print("\033c", end="")  # clear screen
      print(frame)

    current_state = self.env.reset()
    done = False
    while not done:    
      time.sleep(0.04)
      frame = self.env.render(mode="rgb_array") 
      ascii_render(frame)
      action = self.select_action(current_state) 
      next_state, reward, done, _ = self.env.step(action)
      current_state = next_state

