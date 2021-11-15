import random
import numpy as np
import torch
from pathlib import Path
def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"