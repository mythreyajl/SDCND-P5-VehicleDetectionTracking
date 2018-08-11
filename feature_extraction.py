import numpy as np
from glob import glob
import cv2
import argparse
import os
import pickle

from skimage.feature import hog


def convert_format(img, fmt):

    if fmt != 'BGR':
        switcher = {
            'HSV': cv2.COLOR_BGR2HSV,
            'LUV': cv2.COLOR_BGR2LUV,
            'YUV': cv2.COLOR_BGR2YUV,
            'YCrCb': cv2.COLOR_BGR2YCR_CB,
            'HLS': cv2.COLOR_BGR2HLS,
            'RGB': cv2.COLOR_BGR2RGB
        }
        col = switcher.get(fmt, "Invalid space")
        img = cv2.cvtColor(img, col)
    return img


def extract_spatial(img, size=(32, 32)):

    features = cv2.resize(img, size).ravel()
    return features


def extract_color_histogram(img, nbins=32, bins_range=(0, 256)):

    c1hist = np.histogram(a=img[:, :, 0], bins=nbins, range=bins_range)
    c2hist = np.histogram(a=img[:, :, 1], bins=nbins, range=bins_range)
    c3hist = np.histogram(a=img[:, :, 2], bins=nbins, range=bins_range)

    features = np.concatenate((c1hist[0], c2hist[0], c3hist[0]))
    return features


def extract_hog(img, orient=9, pix_per_cell=8, cell_per_block=2):

    if len(img.shape) < 3 or img.shape[-1] == 1:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=False, feature_vector=True,
                       block_norm="L2-Hys")
    else:
        features = []
        for ch in range(img.shape[-1]):
            feat = hog(img[:, :, ch], orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=False, feature_vector=True,
                       block_norm="L2-Hys")
            features.append(feat)
        features = np.ravel(features)
    return features


def extract_features(path, orient=9, pix_per_cell=8, cell_per_block=2, nbins=32, size=(32, 32)):

    features = []

    for im_path in glob(path+"/*.png"):
        img = cv2.imread(im_path)
        YCrCb = convert_format(img, 'YCrCb')
        Y = YCrCb[:, :, 0]
        HSV = convert_format(img, 'HSV')

        h_feature = extract_hog(img=Y, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
        c_feature = extract_color_histogram(img=YCrCb, nbins=nbins)
        s_feature = extract_spatial(img=HSV[:, :, 0], size=size)
        feature = np.concatenate((h_feature, c_feature, s_feature))

        features.append(feature)

    return features


def save_features(car_feats, non_car_feats):
    data = {"car": car_feats, "non-car": non_car_feats}
    with open('features.p', 'wb') as f:
        pickle.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Parser for feature extraction')
    parser.add_argument('-v', '--vehicles',     dest="vpath", help='Location of vehicle images')
    parser.add_argument('-n', '--non-vehicles', dest="npath", help='Location of non-vehicle images')
    parser.add_argument('-s', '--save',         dest='save',  help='save outputs', action='store_true', default='True')
    args = parser.parse_args()

    if not args.vpath or not args.npath:
        argparse.ArgumentError('No path to vehicles provided')

    return args


if __name__ == '__main__':
    args = parse_args()
    car_features = []
    non_car_features = []
    walk_v = os.walk(args.vpath)
    for dir in list(walk_v)[1:]:
        features = extract_features(dir[0])
        car_features += features

    walk_n = os.walk(args.npath)
    for dir in list(walk_n)[1:]:
        features = extract_features(dir[0])
        non_car_features += features

    if args.save:
        save_features(car_features, non_car_features)
