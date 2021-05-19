#!/usr/bin/env python3
import pickle
filename = "log.pkl"
log = pickle.load(open(filename, 'rb'))

count = 0
for distance in log[1][-10000:]:
  if distance == 3161:
    count += 1

print(count/10000)
