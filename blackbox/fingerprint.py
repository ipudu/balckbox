###############################################################################
# -*- coding: utf-8 -*- 
# Authors: Pu Du
###############################################################################

import numpy as np
from numpy.linalg import norm
from loader import XYZLoader

def angle(A, B, C):
    """calculate the cos(theta)"""
    BA = A - B
    BC = C - B
    theta = np.dot(BA, BC)/(norm(BA)*norm(BC))
    return theta

def fc(Rij, Rc):
    """cutoff function"""
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)

def g1(Rij, Rc):
    """type 1 of fingerprint"""
    return fc(Rij, Rc)

def g2(Rij, Rc, Rs, etha):
    """type 1 of fingerprint"""
    return np.exp(-eta * (Rij - Rs) ** 2 / (Rc ** 2)) * fc(Rij, Rc)

def g3(Rij, Rc, kappa):
    """type 3 of fingerprint"""
    return np.cos(kappa * Rij) * fc(Rij, Rc)

def g4(Rij, Rik, Rjk, Rc, Rs, zeta, lmb, theta, etha):
    """type 4 of fingerprint"""
    return (1 + lmb * theta) * np.exp(-etha * (Rij ** 2 + Rik ** 2 + Rjk ** 2)) * \
            fc(Rij, Rc) * fc(Rik, Rc) * fc(Rjk, Rc)

def g5(Rij, Rik, Rjk, Rc, Rs, zeta, lmb, theta, etha):
    """type 4 of fingerprint"""
    return (1 + lmb * theta) * np.exp(-etha * (Rij ** 2 + Rik ** 2)) * \
            fc(Rij, Rc) * fc(Rik, Rc)


class FingerPrint(object):
    """class of calculating fingerprints"""

    def __init__(self, trajectory):
        self.traj  = trajectory

    def neighbors(self, coords):
        """calculate neighbors"""
        n_atoms = self.traj.n_atoms
        dist = np.zeros([n_atoms, n_atoms], dtype=np.float)
        for i in range(n_atoms):
            tmp = np.sqrt((coords - coords[i]) ** 2).sum(axis=1)
            dist[:, i] = tmp
        return dist

    def G1(self, coords, Rc):
        """ calculate g1 fingerprints"""
        dist = self.neighbors(coords)
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(y):
                if i != j:
                    tmp = g1(dist[i][j], Rc)
                    finger[i] += tmp
        return finger
 
    def G2(self, coords, Rc, Rs, theta):
        """ calculate g2 fingerprints"""
        dist = self.neighbors(coords)
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(y):
                if i != j:
                    tmp = g2(dist[i][j], Rc, Rs, theta)
                    finger[i] += tmp
        return finger

    def G3(self, coords, Rc, kappa):
        """ calculate g3 fingerprints"""
        dist = self.neighbors(coords)
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(y):
             if i != j:
                    tmp = g3(dist[i][j], Rc, kappa)
                    finger[i] += tmp
        return finger   

    
    def G4(self, coords, Rc, Rs, zeta, lmb, theta, etha):
        """ calculate g4 fingerprints"""
        dist = self.neighbors(coords)
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(x):
                for k in range(x):
                    if i != j and j != k and i != k:
                        tmp = g4(dist[i][j],
                                 dist[i][k],
                                 dist[j][k],
                                 Rc, Rs, zeta, lmb,
                                 theta, etha)
                        finger[i] += tmp
        return finger

    def G5(self, coords, Rc, Rs, zeta, lmb, etha):
        """ calculate g4 fingerprints"""
        dist = self.neighbors(coords)
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(x):
                for k in range(x):
                    if i != j and j != k and i != k:
                        theta = angle(coords[i], coords[j], coords[k])

                        tmp = g5(dist[i][j], dist[i][k], dist[j][k],
                                 Rc, Rs, zeta, lmb, theta, etha)

                        finger[i] += tmp
        return finger

if __name__ == '__main__':

    a = XYZLoader('hcl_10_wat-traj.xyz')
    b = FingerPrint(a)
    #finger1 = b.G1(b.traj.coords[0], 4)
    finger1 = b.G5(b.traj.coords[0], 4, 0, 1, 0, 0)
    for c in b.traj.coords:
        print(c.shape)