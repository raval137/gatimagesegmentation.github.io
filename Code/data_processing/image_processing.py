import numpy as np


def standardize_img(img_array, mean, std):
    centered = img_array-mean
    standardized = centered/std
    return standardized
