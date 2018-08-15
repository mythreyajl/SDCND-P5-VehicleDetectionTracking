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


def extract_spatial(img, spatial_size=(32, 32)):

    features = cv2.resize(img, spatial_size).ravel()
    return features


def extract_color_histogram(img, nbins=32, bins_range=(0, 256)):

    c1hist = np.histogram(a=img[:, :, 0], bins=nbins, range=bins_range)
    c2hist = np.histogram(a=img[:, :, 1], bins=nbins, range=bins_range)
    c3hist = np.histogram(a=img[:, :, 2], bins=nbins, range=bins_range)

    features = np.concatenate((c1hist[0], c2hist[0], c3hist[0]))
    return features


def extract_hog(img, orient=9, pix_per_cell=8, cell_per_block=2, feature_vector=True):

    if len(img.shape) < 3 or img.shape[-1] == 1:
        features, im = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=True, feature_vector=feature_vector,
                       block_norm="L2-Hys")
    else:
        features = []
        for ch in range(img.shape[-1]):
            feat, im = hog(img[:, :, ch], orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=True, feature_vector=True,
                       block_norm="L2-Hys")
            features.append(feat)
        features = np.ravel(features)
    return features, im


def extract_features(img, orient=9, pix_per_cell=8, cell_per_block=2, nbins=32, spatial_size=(32, 32)):
    # BGR = img
    YCrCb = convert_format(img, 'YCrCb')
    # HSV = convert_format(img, 'HSV')
    # HLS = convert_format(img, 'HLS')
    # YUV = convert_format(img, 'YUV')
    # LUV = convert_format(img, 'LUV')
    # RGB = convert_format(img, 'RGB')
    """
    h1_feats, _ = extract_hog(img=HSV[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                           cell_per_block=cell_per_block, feature_vector=False)#.ravel()
    h2_feats, _ = extract_hog(img=HSV[:, :, 1], orient=orient, pix_per_cell=pix_per_cell,
                           cell_per_block=cell_per_block, feature_vector=False)#.ravel()
    h3_feats, _ = extract_hog(img=HSV[:, :, 2], orient=orient, pix_per_cell=pix_per_cell,
                           cell_per_block=cell_per_block, feature_vector=False)#.ravel()

    h_features = np.hstack((h1_feats[:, :].ravel(), h2_feats[:, :].ravel(), h3_feats[:, :].ravel()))
    """
    h_features, _ = extract_hog(img=YCrCb, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    c_features = extract_color_histogram(img=YCrCb, nbins=nbins)
    s_features = extract_spatial(img=YCrCb, spatial_size=spatial_size)

    return np.concatenate((h_features, c_features, s_features))
    # return np.hstack((s_features, c_features, h_features)).reshape(1, -1)


def extract_features_folder(path, orient=9, pix_per_cell=8, cell_per_block=2, nbins=32, size=(32, 32)):

    features = []

    for im_path in glob(path+"/*.png"):
        img = cv2.imread(im_path)
        feature = extract_features(img, orient, pix_per_cell, cell_per_block, nbins, size)
        features.append(feature)

    return features


def extract_all_features(car_dir, non_car_dir):
    car_features = []
    non_car_features = []
    walk_v = os.walk(car_dir)
    for dir in list(walk_v)[1:]:
        features = extract_features_folder(dir[0])
        car_features += features
        print("Hi")

    walk_n = os.walk(non_car_dir)
    for dir in list(walk_n)[1:]:
        features = extract_features_folder(dir[0])
        non_car_features += features
        print("Hello")

    return car_features, non_car_features


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
    cf, ncf = extract_all_features(args.vpath, args.npath)

    if args.save:
        save_features(cf, ncf)
