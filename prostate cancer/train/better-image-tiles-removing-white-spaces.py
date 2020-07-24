#!/usr/bin/env python
# coding: utf-8

# ## Importing dependencies
# some parts of this code was used from here https://www.kaggle.com/rftexas/better-image-tiles-removing-white-spaces
# In[1]:


import os
import cv2
import PIL
import random
import openslide
import skimage.io
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, display

train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')#.sample(n=10, random_state=0).reset_index(drop=True)
# train_df = train_df.loc[:20]

images = list(train_df['image_id'])
labels = list(train_df['isup_grade'])


PATH = "../input/prostate-cancer-grade-assessment/"
data_dir = '../input/prostate-cancer-grade-assessment/train_images/'

print(f"Train data shape before reduction: {len(train_df)}")

masks = os.listdir(PATH + 'train_label_masks/')

df_masks = pd.Series(masks).to_frame()

df_masks.columns = ['mask_file_name']

df_masks['image_id'] = df_masks.mask_file_name.apply(lambda x: x.split('_')[0])
df_train = pd.merge(train_df, df_masks, on='image_id', how='outer')

print('Number of masks', len(df_masks))

df_train_red = train_df[~df_train.mask_file_name.isna()]

print(f"Train data shape after reduction: {len(df_train_red)}")

train_df = df_train_red

# ## Compute statistics

# First we need to write a function to compute the proportion of white pixels in the region.
def compute_statistics(image):
    """
    Args:
        image                  numpy.array   multi-dimensional array of the form WxHxC
    
    Returns:
        ratio_white_pixels     float         ratio of white pixels over total pixels in the image 
    """
    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height
    
    num_white_pixels = 0
    
    summed_matrix = np.sum(image, axis=-1)
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    num_white_pixels = np.count_nonzero(summed_matrix > 620)
    ratio_white_pixels = num_white_pixels / num_pixels
    
    green_concentration = np.mean(image[1])
    blue_concentration = np.mean(image[2])
    
    return ratio_white_pixels, green_concentration, blue_concentration


# ## Select k-best regions

# Then we need a function to sort a list of tuples, where one component of the tuple is the proportion of white pixels 
# in the regions. We are sorting in ascending order.
def select_k_best_regions(regions, k=16):
    """
    Args:
        regions -- list           list of 2-component tuples first component the region, 
                                             second component the ratio of white pixels
                                             
        k -- int -- number of regions to select
    """
    regions = [x for x in regions if x[3] > 180 and x[4] > 180]
    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]
    return k_best_regions


# Since we will only store, the coordinates of the top-left pixel, we need a way to retrieve the k best regions, 
# hence the function hereafter...
def get_k_best_regions(coordinates, image, window_size=512):
    regions = {}
    for i, tup in enumerate(coordinates):
        x, y = tup[0], tup[1]
        regions[i] = image[x : x+window_size, y : y+window_size, :]
    
    return regions


# ## Slide over the image

# The main function: the two while loops slide over the image (the first one from top to bottom, the second from left to right). 
# The order does not matter actually.
# Then you select the region, compute the statistics of that region, sort the array and select the k-best regions.
def generate_patches(slide_path, window_size=200, stride=128, k=36):
    
    try:
        image = skimage.io.MultiImage(slide_path)[1]
    except:
        return None, None, None
    
    image = np.array(image)
    
    max_width, max_height = image.shape[0], image.shape[1]
    regions_container = []
    i = 0
    
    while window_size + stride*i <= max_height:
        j = 0
        
        while window_size + stride*j <= max_width:            
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride
            
            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :
            ]
            
            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch)
            
            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)
            regions_container.append(region_tuple)
            
            j += 1
        
        i += 1
    
    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)
    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)
    
    return image, k_best_region_coordinates, k_best_regions

# ## Glue to one picture
def glue_images_one(tiles, image_size=200, n_tiles=32):

        idxes = list(range(n_tiles))

        n_row_tiles = int(np.sqrt(n_tiles))
        image = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]
                else:
                    this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255
                    
                this_img = 255 - this_img
                
                h1 = h * image_size
                w1 = w * image_size
                image[h1:h1+image_size, w1:w1+image_size] = this_img

        image = 255 - image
        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(0, 1, 2)

        return image    

WINDOW_SIZE = 200
STRIDE = 64
K = 36
counter = 0

import matplotlib
from PIL import Image

fig, ax = plt.subplots(3, 2, figsize=(20, 25))

for i, img in enumerate(images):
    url = data_dir + img + '.tiff'
    image, _, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

    if np.sum(image) == None:
        continue

    glued_image = glue_images_one(tiles=best_regions, image_size=WINDOW_SIZE, n_tiles=K)

    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / glued_image.max() * (glued_image - glued_image.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(f'./test/{img}.png')
    print(counter)
    counter += 1
    
