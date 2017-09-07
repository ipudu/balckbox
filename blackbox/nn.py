###############################################################################
# -*- coding: utf-8 -*- 
# Authors: Pu Du
###############################################################################

from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm

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

def G2(Rij, Rc):
    """type 1 of fingerprint"""
    pass

def G4(Rij, Rc, ):
    """type 4 of fingerprint"""
    pass
