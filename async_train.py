#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch import tensor, optim
import torch.multiprocessing as mp
import numpy as np
import gym
import random
from collections import namedtuple
from tqdm import trange
import time

from model import DuelingDQN
from wrappers import *
import hyperparameters as hp

Experience = namedtuple('Experience', ['s', 'a', 'r', 's_', 'done'])

def actor_fn(i, shared_model, replay_buffer, distance_log, env_name, device):
  print(f"start actor {i}")
  env = make_mario(env_name, COMPLEX_MOVEMENT)
  local_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to('cpu')
  local_model.load_state_dict(shared_model.state_dict())
  local_model.eval()

  current_state = env.reset()
  while True:
    with torch.no_grad():
      x = tensor(current_state, dtype=torch.float32).unsqueeze(0) / 255.0
      local_model.reset_noise()
      q_value = local_model(x)
      action = q_value.argmax().item()

    next_state, reward, done, info = env.step(action)
    replay_buffer.append(Experience(current_state, action, reward, next_state, done))
    current_state = next_state if not done else env.reset()

    distance_log.append(info['x_pos'])
    if len(distance_log) > 1000:
      distance_log.pop(0)

    if len(replay_buffer) > hp.MEMORY_SIZE:
      replay_buffer.pop(0)

    if random.random() < 0.01:
      local_model.load_state_dict(shared_model.state_dict())

def learner_fn(shared_model, target_model, replay_buffer, distance_log, env_name, device):
  print("start learner")
  env = make_mario(env_name, COMPLEX_MOVEMENT)
  shared_model = shared_model.to(device)
  optimizer = optim.Adam(shared_model.parameters(), lr=hp.ALPHA)
  loss_function = nn.MSELoss()

  while len(replay_buffer) < hp.MIN_EXP:
    print(f"Waiting for replay buffer: {len(replay_buffer)}/{hp.MIN_EXP}")
    time.sleep(1)

  for ep in (t := trange(hp.N_EPISODES)):
    indices = np.random.choice(len(replay_buffer), hp.BATCH_SIZE, replace=False)
    states, actions, rewards, next_states, dones = zip(*[replay_buffer[i] for i in indices])

    states = tensor(np.stack(states)).to(device)
    actions = tensor(actions, dtype=torch.int64).to(device)
    next_states = tensor(np.stack(next_states)).to(device)
    rewards = tensor(np.clip(reward, -1, 1), dtype=torch.float32).to(device)
    dones = tensor(dones, dtype=torch.float32).to(device)

    shared_model.reset_noise()
    target_model.reset_noise()
    # Get Q-values of actions taken for each state
    Q = shared_model(states)
    Q = Q.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
      target_Q = target_model(next_states)
      target_Q = target_Q.amax(1)
      td_target = target_Q * hp.GAMMA + rewards * (1 - dones)

    loss = loss_function(Q, td_target)
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    # Periodically update target
    if random.random() < 0.01:
      target_model.load_state_dict(shared_model.state_dict())

    if (ep % 1000) == 0 and ep != 0:
      torch.save(shared_model.state_dict(), "pretrained_models/model_%d.pth" % ep)
      with open("data/async_distance_log", "a") as f:
        f.write(f"{ep},{np.mean(distance_log[-50:])}\n")
    t.set_description("loss %.2f mean distance %d" % (loss.item(), np.mean(distance_log[-50:])))

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(device)

  mp.set_start_method('forkserver')
  env_name = "SuperMarioBros-1-1-v3"

  env = make_mario(env_name, COMPLEX_MOVEMENT)
  input_shape = env.observation_space.shape
  n_actions = env.action_space.n

  shared_model = DuelingDQN(input_shape, n_actions)
  shared_model.share_memory()

  target_model = DuelingDQN(input_shape, n_actions).to(device)
  target_model.load_state_dict(shared_model.state_dict())

  manager = mp.Manager()
  replay_buffer = manager.list()
  distance_log = manager.list()

  processes = []

  for i in range(16):
    p = mp.Process(target=actor_fn, args=(i, shared_model, replay_buffer, distance_log, env_name, device))
    p.start()
    processes.append(p)

  learner = mp.Process(target=learner_fn, args=(shared_model, target_model, replay_buffer, distance_log, env_name, device))
  learner.start()
  processes.append(learner)

  for p in processes:
    p.join()
