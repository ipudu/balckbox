###############################################################################
# -*- coding: utf-8 -*- 
# Authors: Pu Du
###############################################################################

from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
from torch.autograd import Variable


def angle(A, B, C):
    """calculate the angle"""
    BA = A - B
    BC = C - B
    theta = np.arccos(np.dot(BA, BC)/(norm(BA)*norm(BC)))
    return np.rad2deg(theta)


###############################################################################
#cutoff functions
###############################################################################
def fc(Rij, Rc):
    """cutoff function"""
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)

def fc_prime(Rij, Rc):
    """derivative of cutoff function"""
    if Rij > Rc:
        return 0.
    else:
        return -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)

def fc_tanh(Rij, Rc):
    """ tanh cutoff function"""
    if Rij > Rc:
        return 0.
    else:
        return np.power(np.tanh(1 - Rij / Rc), 3)


###############################################################################
#fingerprint functions
###############################################################################

def G1(Rij, Rc):
    """type 1 of fingerprint"""
    return fc(Rij, Rc)

def G2(Rij, Rc, Rs, etha):
    """type 1 of fingerprint"""
    return np.exp(-eta * (Rij - Rs) ** 2 / (Rc ** 2)) * fc(Rij, Rc)

def G3(Rij, Rc, kappa):
    """type 3 of fingerprint"""
    return np.cos(kappa * Rij) * fc(Rij, Rc)

def G4(Rij, Rik, Rjk, Rc, Rs, zeta, lmb, theta, etha):
    """type 4 of fingerprint"""
    return (1 + lmb * theta) * np.exp(-etha * (Rij ** 2 + Rik ** 2 + Rjk ** 2)) * \
            fc(Rij, Rc) * fc(Rik, Rc) * fc(Rjk, Rc)

def G5(Rij, Rik, Rjk, Rc, Rs, zeta, lmb, theta, etha):
    """type 4 of fingerprint"""
    return (1 + lmb * theta) * np.exp(-etha * (Rij ** 2 + Rik ** 2)) * \
            fc(Rij, Rc) * fc(Rik, Rc)


###############################################################################
#feed forward neural network
###############################################################################

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#Train the Model
