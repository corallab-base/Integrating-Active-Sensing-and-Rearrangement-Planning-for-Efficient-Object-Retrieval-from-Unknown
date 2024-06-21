import os
import torch
import numpy as np
import runner


if __name__ == '__main__':
    scene = np.load("test.npy")

    print (runner.feed_forward(scene, [0, 0, -0.255, 0.967, 0.193, 0.247, 0.393]))
