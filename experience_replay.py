"""
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import random 
import numpy as np
from collections import namedtuple, deque

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplay:
  def __init__(self, capacity):
    self.memory = deque(maxlen=capacity) 
    self.index = 0
  
  def length(self):
    return len(self.memory) 

  def push(self, experience):
    """
    Push experience into memory
    """
    self.memory.append(experience)
    
  def sample_batch(self, batch_size):
    """
    Sample random experience according to batch size from memory

    Returns a np array of each
    """
    indices = np.random.choice(len(self.memory), batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in indices])
    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32) ,np.array(next_states),  np.array(dones, dtype=np.uint8)
