import numpy as np
import matplotlib.pyplot as plt
import torch
import toml
import torch
import torch.autograd as grad
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", default=argparse.SUPPRESS, type=int, 
        help='epochs')
    parser.add_argument("-lr", type=float, default=argparse.SUPPRESS, 
        help=" -log10 of learning rate (ex. 2 -> 0.01)")
    
    args = parser.parse_args()
    if "lr" in args:
        print("lr true")
        
if __name__ == "__main__":
    main()
