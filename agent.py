import torch 
import torch.nn as nn
import cv2
import math
import numpy as np
import time
from tqdm import tqdm

from model import DQN
from wrappers import *
from experience_replay import ExperienceReplay, Experience
import hyperparameters as hp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class DQNAgent:
  def __init__(self):
    self.env = make_mario("SuperMarioBros-1-1-v0", COMPLEX_MOVEMENT)
  
    self.n_episodes = hp.N_EPISODES 
    self.alpha = hp.ALPHA 
    self.epsilon_start = hp.EPSILON_START 
    self.epsilon_final = hp.EPSILON_FINAL 
    self.epsilon = hp.EPSILON_START
    self.gamma = hp.GAMMA 
    self.decay = hp.DECAY 
    self.memory_size = hp.MEMORY_SIZE 
    self.batch_size = hp.BATCH_SIZE 
    self.min_exp = hp.MIN_EXP 
    self.target_update_freq = hp.TARGET_UPDATE_FREQ 

    self.input_shape = self.env.observation_space.shape
    self.n_actions = self.env.action_space.n
    print(self.input_shape, self.n_actions)

    self.model = DQN(self.input_shape, self.n_actions).to(device)
    self.target_model = DQN(self.input_shape, self.n_actions).to(device)

    # if you want to load and watch
    if hp.LOAD:
      self.model.load_state_dict(torch.load("pretrained_models/pretrained_39900_model.pth"))
      self.target_model.load_state_dict(self.model.state_dict())
      self.epsilon_start = self.epsilon_final
      self.model.eval()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
    self.replay_memory = ExperienceReplay(self.memory_size)
  
   
  def compute_loss(self, minibatch, device):
    """
    Calculate Mean Square Error (MSE) between actual values and expected values from Deep Q-Network
    """
    states, actions, rewards, next_states, dones = minibatch
    
    states = torch.tensor(states).to(device) 
    next_states = torch.tensor(next_states).to(device)
    actions = torch.tensor(actions).to(device) 
    rewards = torch.tensor(rewards).to(device) 
    done = torch.ByteTensor(dones).to(device)

    q_values = self.model(states)
    next_q_values = self.target_model(next_states)

    q_value = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0] 

    # Sets the gradients to zero before backprop so gradients dont accumulate?
    self.optimizer.zero_grad() 

    td_target = next_q_value * self.gamma + rewards
    loss = nn.MSELoss()(q_value, td_target)
    loss.backward() # Back Propagation 
    self.optimizer.step() # Gradient Descent? 
    
  def select_action(self, state): 
    """
    Epsilon-greedy policy
    """
    if np.random.random() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      # Pick best action with DQN 
      state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
      q_value = self.model(state)
      #print(q_value.max(1))
      action = q_value.max(1)[1].item()
    return action

  def get_epsilon(self, t):
    """
    Returns value of epsilon. Epsilon declines as we advance in time? or episode? 
    """
    return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1 * ((t + 1) / self.decay))
    
  def write_log(self, episode, n_episodes, total_steps, total_reward, distance, epsilon, game_time, batch_size, memory_size):
    #print(episode, n_episodes, total_reward, distance, epsilon, game_time, memory_size)
    total_frames = 4*total_steps
    with open("marios_run.txt", "a") as file:
      file.write("episode:%d/%d total_steps:%d total_frames:%d total_reward:%d distance:%d epsilon:%.5f game_time:%d batch_size=%d memory_size:%d\n" % 
                (episode, n_episodes, total_steps, total_frames, total_reward, distance, epsilon, game_time, batch_size, memory_size))
    
  def train(self):
    """
    Deep Q-Learning with Experience Replay
    """
    step_index = 0
    for episode in tqdm(range(self.n_episodes)):
      # For logging data
      game_time = 0
      total_reward = 0
      distance = 0
      # Start
      current_state = self.env.reset()
      print(np.array(current_state).shape)

      done = False 
      while not done:
        #time.sleep(0.05)
        step_index += 1
        # Do we reset the epsilon back to start after every episode?
        self.epsilon = self.get_epsilon(step_index)

        # Choose A from S
        action = self.select_action(current_state)
        #self.env.render()
        
        next_state, reward, done, info = self.env.step(action)
        # This is for logging data
        total_reward += reward 
        distance = max(info['x_pos'], distance)
        #print(distance)
        game_time = info['time']

        #print(next_state, next_state.shape)

        # Store step information in replay memory 
        experience = Experience(current_state, action, reward, next_state, done)
        self.replay_memory.push(experience)
        current_state = next_state 

        if (self.replay_memory.length() < self.min_exp):
          continue
        
        if (step_index % self.target_update_freq) == 0:
          self.target_model.load_state_dict(self.model.state_dict())
        
        # Sample random minibatch of transitions from replay memory
        # Calculate the target
        # (Update weights) Perform stochastic gradient descent
        
        # Update every 4 steps
        if (step_index % 4) == 0:
          minibatch = self.replay_memory.sample_batch(self.batch_size)
          self.compute_loss(minibatch, device)

      if (episode % 10000) == 0 and episode != 0:
        torch.save(self.model.state_dict(), "pretrained_%d_model.pth" % episode)
      # Log data
      self.write_log(episode, self.n_episodes, step_index, total_reward, distance, self.epsilon, game_time, self.batch_size, self.memory_size)
  

agent = DQNAgent()
agent.train()
#agent.run()
