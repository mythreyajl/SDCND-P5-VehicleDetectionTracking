import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


def extract_spatial(img, size=(32,32)):
    features = cv2.resize(img, size).ravel()
    return features


def extract_color_histogram(img, nbins=32, bins_range=(0, 256)):
    bhist = np.histogram(a=img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(a=img[:, :, 1], bins=nbins, range=bins_range)
    rhist = np.histogram(a=img[:, :, 2], bins=nbins, range=bins_range)

    features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return features


def extract_hog(img, orient, pix_per_cell, cell_per_block):
    features = hog(img, orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   visualise=False, feature_vector=False,
                   block_norm="L2-Hys")
    return features


def combine_features():
    pass


def extract_features(path, orient, pix_per_cell, cell_per_block):

    for im_path in glob(path):
        img = cv2.imread(im_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hog_features, _ = hog(img, orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              visualise=False, feature_vector=False,
                              block_norm="L2-Hys")
