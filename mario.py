#!/usr/bin/env python3
import sys
from agent import *

agent = DQNAgent()  
if sys.argv[1] == "train":
  agent.train() 
elif sys.argv[1] == "play":
  agent.load("pretrained_models/pretrained_39900_model.pth")
  agent.play()
else:
  print("Usage: ./mario.py train | play")
