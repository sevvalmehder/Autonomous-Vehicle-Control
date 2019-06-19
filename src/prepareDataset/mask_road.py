import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Define the path of images
mask_path = './_seg/'

# Get the names of images
mask_list = os.listdir(mask_path)
mask_list = [mask_path+i for i in mask_list]

# Read all images
N = len(mask_list)
for i in range(N):
    mask = imageio.imread(mask_list[i])

# This lines can be change depend on your network
height, width = 480,360
#masks = np.zeros((len(image_list), height, width, 3), dtype=np.int16)
masks = np.zeros((len(mask_list), height, width, 1), dtype=np.int8)

# Create directory for masked images
mask_path = './mask'

if not os.path.exists(mask_path):
	os.makedirs(mask_path)

i = 0
for n in tqdm(range(len(mask_list))):
    mask = imageio.imread(mask_list[n])
    
    mask_road = np.zeros((360, 480, 3))

    # 7 represent the road pixels for Carla semantic segmentation images
    # Any value can be assigned for these pixels depend on your network
    mask_road[np.where(mask==7)[0], np.where(mask==7)[1]]=[1,1,1]
    
    name = mask_path + mask_list[n][6:]
    cv2.imwrite(name, mask_road)
    i+=1
