from __future__ import division

import os
import numpy as np
import random

from scipy.misc import imread, imresize

from itertools import islice

import random

def return_data(data_path, split=0.8):

    # Get the data in text file
    data_file = os.path.join(data_path, 'data.txt')

    data_map = []

    # open the file and get information
    with open(data_file) as fp:
        for line in islice(fp, None):
            # Get image path and angle
            image_path, angle = line.strip().split()
            full_image_path = os.path.join(data_path, image_path)
            # Store path information and angle together
            data_map.append([full_image_path, float(angle)])

    # Shuffle the data
    # Note: Shuffle works in place
    random.shuffle(data_map)

    X = [data[0] for data in data_map]
    Y = [data[1] for data in data_map]

    #images = np.array([np.float32(imresize(imread(im), size=(66, 200))) / 255 for im in X])
    images = np.array([np.float32(imresize(imread(im, mode= "RGB"), size=(66, 200))) for im in X])
    split_index = int(split * len(X))

    train_X = images[:split_index]
    train_y = Y[:split_index]
    test_X = images[split_index:]
    test_y = Y[split_index:]

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
