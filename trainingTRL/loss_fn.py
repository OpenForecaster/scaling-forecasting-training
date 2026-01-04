# Create logger, import torch, numpy, sys and logging

import logging
import numpy as np
import sys
import torch

# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Create vector of size 5 of 1s and 0s
y_true = np.array([1, 0, 0])
# y_pred = np.array([0, 0, 1, 1, 0])
y_pred = np.array([0, 0.8, 0.1, 0.2, 0.7])
y_pred = np.array([0.5, 0.5,  0.5])


y_true = [1.0]
y_pred = [0.30]
# Print binary cross entropy 
bce = torch.nn.BCELoss()
bce_loss = bce(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32))
logger.info(f"Binary cross entropy: {-bce_loss.item()}")
