import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  """
  Deep Q Network from https://arxiv.org/abs/1312.5602
  """
  def __init__(self, input_shape, n_actions):
    super(DQN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU()
    ) 

    conv_out_shape = self.get_conv_out(input_shape)
    self.fc = nn.Sequential( 
      nn.Linear(conv_out_shape, 512),
      nn.ReLU(),
      nn.Linear(512, n_actions)
    )

  def get_conv_out(self, shape):
    out = self.conv(torch.zeros(1, *shape))
    return np.prod(out.shape)

  def forward(self, x):
    x = self.conv(x).view(x.shape[0], -1) 
    return self.fc(x)

class DuelingDQN(nn.Module):
  def __init__(self, input_shape, n_actions):
    super().__init__()
    self.feature = nn.Sequential(
      nn.Conv2d(4, 32, 8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, 4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3, stride=1),
      nn.ReLU(),
      nn.Flatten()
    )

    self.value_stream = nn.Sequential(
      NoisyLinear(3136, 512),
      nn.ReLU(),
      NoisyLinear(512, 1)
    )

    self.advantage_stream = nn.Sequential(
      NoisyLinear(3136, 512),
      nn.ReLU(),
      NoisyLinear(512, n_actions)
    )

  def reset_noise(self):
    for module in self.modules():
      if isinstance(module, NoisyLinear):
        module.reset_noise()

  def forward(self, x):
    x = self.feature(x / 255.0)
    value = self.value_stream(x)
    advantage = self.advantage_stream(x)
    q = value + advantage - advantage.mean(dim=1, keepdim=True)
    return q

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init

    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))


    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer("bias_epsilon", torch.empty(out_features))

    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / self.in_features**0.5
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / self.in_features**0.5)
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / self.out_features**0.5)

  def reset_noise(self):
    self.weight_epsilon.normal_()
    self.bias_epsilon.normal_()

  def forward(self, x):
    if True:
      weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
      bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
    else:
      weight = self.weight_mu
      bias = self.bias_mu
    return F.linear(x, weight, bias)

