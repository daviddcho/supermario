import pickle
import time

class Logger():
  def __init__(self, n_episodes, epsilon, memory_size, batch_size):
    self.params = [n_episodes, epsilon, memory_size, batch_size]
    self.ep_rewards = []
    self.ep_distances = []
    self.ep_avg_losses = []
    self.ep_avg_qs = []

    self.init_episode()
    self.start_time = time.time()
  
  def init_episode(self):
    self.ep_reward = 0.0
    self.ep_distance = 0.0
    self.ep_loss = 0.0
    self.ep_loss_num = 0.0
    self.ep_q = 0.0
  
  def log_step(self, reward, distance, loss, q):
    self.ep_reward += reward
    self.ep_distance = distance 
    if loss:
      self.ep_loss += loss 
      self.ep_loss_num += 1
      self.ep_q += q 

  def log_episode(self):
    self.ep_rewards.append(self.ep_reward)    
    self.ep_distances.append(self.ep_distance)
    if self.ep_loss_num == 0:
      avg_loss = 0.0
      avg_q = 0.0
    else:
      avg_loss = self.ep_loss/self.ep_loss_num 
      avg_q = self.ep_q/self.ep_loss_num
    self.ep_avg_losses.append(avg_loss)
    self.ep_avg_qs.append(avg_q)

    self.init_episode()
    
  def record(self):
    logs = [self.params, self.ep_rewards, self.ep_distances, self.ep_avg_losses, self.ep_avg_qs]
    filename = "data/log.pkl"
    with open(filename, 'wb') as wfp:
      pickle.dump(log, wfp)

    t = time.time() - self.start_time
    print("Total training time: ", time.strftime("%H:%M:%S", time.gmtime(t)))

"""
TODO:
average reward per episode
win rate
average Q value / value estimates (log scale)
"""
