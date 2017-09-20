###############################################################################
# -*- coding: utf-8 -*- 
# Authors: Pu Du
###############################################################################
import numpy as np
from xyz import XYZLoader

class Neighborlist(object):
    """class of calculating neighbor list"""
    def __init__(self, trajectory):
        self.traj  = trajectory

    def neighbors(self, coords):
        """calculate neighbors"""
        n_atoms = self.traj.n_atoms
        dist = np.zeros([n_atoms, n_atoms], dtype=np.float)
        for i in range(n_atoms):
            tmp = np.sqrt((coords - coords[i]) ** 2).sum(axis=1)
            dist[:, i] = tmp
        
        print(dist)

if __name__ == '__main__':

    a = XYZLoader('hcl_10_wat-traj.xyz')
    b = Neighborlist(a)
    b.neighbors(b.traj.coords[0])