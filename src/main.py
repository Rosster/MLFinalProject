from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json


class Runner(object):
  def __init__(self):
    self.start = datetime.now()


  def load_data(self):
    #stockerbot_fqp = os.path.expanduser('~/virtual_envs/ml_data/Final/stockerbot-export.csv')
    dojpress_fqp = os.path.expanduser('~/virtual_envs/ml_data/Final/combined.json')
    
    
    return self


  def end(self):
    now = datetime.now()
    print("Time Elapsed: ", now - self.start)


if __name__ == "__main__":
  # Env: /Users/rohanjyoti/virtual_envs/mlenv/bin/python3 
  #   point vscode env to this, settings.json -> workspace settings
  #   "python.pythonPath": "/Users/rohanjyoti/virtual_envs/mlenv/bin/python3"
  # To Run: time ~/virtual_envs/mlenv/bin/python3 main.py
  Runner().load_data()\
          .end()
  