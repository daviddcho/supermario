import numpy as np
import torch
import torch.nn as nn

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
