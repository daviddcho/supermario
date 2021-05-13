import pickle
import matplotlib.pyplot as plt
import time
from hyperparameters import TARGET_UPDATE_FREQ 

class Logger():
  def __init__(self):
    self.ep_rewards = []
    self.ep_distances = []
    self.ep_avg_losses = []
    self.ep_avg_qs = []
    
    self.mean_ep_rewards = []
    self.mean_ep_distances = []
    self.mean_ep_avg_losses = []
    self.mean_ep_avg_qs = []


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
    # Reset
    self.init_episode()
    
  def record(self, episode, epsilon, step):
    n = TARGET_UPDATE_FREQ
    mean_ep_reward = np.round(np.mean(self.ep_rewards[-n:]), 3)
    mean_ep_distance = np.round(np.mean(self.ep_distances[-n:]), 3)
    mean_ep_avg_loss = np.round(np.mean(self.ep_avg_losses[-n:]), 3) 
    mean_ep_avg_q = np.round(np.mean(self.ep_avg_qs[-n:]), 3)
    self.mean_ep_rewards.append(mean_ep_reward)
    self.mean_ep_distance.append(mean_ep_distance)
    self.mean_ep_avg_losses.append(mean_ep_avg_loss)
    self.mean_ep_avg_qs.append(mean_ep_avg_q)

    t = time.time() - self.start_time
    print("Total training time: ", time.strftime("%H:%M:%S", time.gmtime(t)))
    
    # Long term?
    log = [self.ep_rewards, self.ep_distances, self.ep_avg_losses, self.ep_avg_qs]
    filename = "data/log.pkl"
    with open(filename, 'wb') as wfp:
      pickle.dump(log, wfp)

    # And just to see
    with open("data/updatelog", "a") as fp:
      f.write(
        f"{episode:8d}{step:8d}{epsilon:8d}"
        f"{mean_ep_reward:12f}{mean_ep_distance:12f}{mean_ep_avg_loss:12f}{mean_ep_avg_q:12f}"
        f"{t:12}"
      )
    
    self.plot(n)

  def plot(self, n):
    for metric in ["ep_rewards", "ep_distance", "ep_avg_losses", "ep_avg_qs"]:
      plt.plot(getattr(self, f"mean_{metrix}"), [i*n for i in range(1, len(self.mean_ep_rewards))])
      plt.savefig(getattr(self, f"{metric}_plot"))
      plt.clf()

"""
TODO:
average reward per episode
win rate (distance: 3161)
average Q value / value estimates (log scale)
"""
