#!/usr/bin/env python3
import sys
from agent import *

if __name__ == "__main__":
  agent = DQNAgent()  
  if sys.argv[1] == "train":
    agent.train() 
  elif sys.argv[1] == "play":
    agent.load("pretrained_models/model_50000.pth")
    agent.play()
  else:
    print("Usage: ./mario.py train | play")
