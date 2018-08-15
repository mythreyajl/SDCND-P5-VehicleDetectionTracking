from glob import glob
import cv2
import numpy as np
from classifier import *
from scipy.ndimage.measurements import label


def draw_boxes(img, boxes, color=(255, 0, 0), thickness=2):

    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, thickness)

    return img


def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    if not x_start_stop[0]:
        x_start_stop[0] = 0
    if not x_start_stop[1]:
        x_start_stop[1] = img.shape[0]
    if not y_start_stop[0]:
        y_start_stop[0] = 0
    if not y_start_stop[1]:
        y_start_stop[1] = img.shape[1]

    x_stride = np.int(xy_window[0] * (1 - xy_overlap[0]))
    y_stride = np.int(xy_window[1] * (1 - xy_overlap[1]))

    x_indices = [x_start_stop[0] + x * x_stride
                 for x in range(np.int((x_start_stop[1] - x_start_stop[0]) / x_stride) - 1)]
    y_indices = [y_start_stop[0] + y * y_stride
                 for y in range(np.int((y_start_stop[1] - y_start_stop[0]) / y_stride) - 1)]

    boxes = []
    [boxes.append(((y, x), (y + xy_window[1], x + xy_window[0]))) for x in x_indices for y in y_indices]

    return boxes


def search_in_windows(img, windows, model, scaler, spatial_size=(32, 32), nbins=32, bins_range=(0, 256),
                      orient=9, pix_per_cell=8, cell_per_block=2):

        good_windows = []
        for window in windows:
            window_im = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            feats = extract_features(window_im, orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, nbins=nbins, spatial_size=spatial_size)
            test_feats = scaler.transform(np.array(feats).reshape(1, -1))
            pred = model.predict(test_feats)
            if pred == 1:
                good_windows.append(window)
        return good_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, scale, model, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, xstart:, :]
    ctrans_tosearch = convert_format(img_tosearch, fmt='YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1, im1 = extract_hog(ch1, orient, pix_per_cell, cell_per_block, feature_vector=False)
    hog2, im2 = extract_hog(ch2, orient, pix_per_cell, cell_per_block, feature_vector=False)
    hog3, im3 = extract_hog(ch3, orient, pix_per_cell, cell_per_block, feature_vector=False)

    windows = []
    for xb in range(2*nxsteps-1):
        for yb in range(2*nysteps-1):
            ypos = yb
            xpos = xb

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            hog_features_show = np.hstack((im1[ypos:ypos + 64, xpos:xpos + 64],
                                           im2[ypos:ypos + 64, xpos:xpos + 64],
                                           im3[ypos:ypos + 64, xpos:xpos + 64]))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = extract_spatial(subimg, spatial_size=spatial_size)
            hist_features = extract_color_histogram(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            # test_features = scaler.transform(np.concatenate((hog_features, hist_features, spatial_features)).reshape(1,-1))
            test_features = scaler.transform(np.hstack((hog_features, hist_features, spatial_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = model.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left + xstart, ytop_draw + ystart),
                              (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                windows.append(((xbox_left + xstart, ytop_draw + ystart), (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))

                # cv2.imshow('hog', 255 * (hog_features_show / np.max(hog_features_show)))
                # cv2.imshow('img', img[ytop_draw + ystart:ytop_draw + ystart + win_draw, xbox_left + xstart: xbox_left + win_draw + xstart, :])
                # cv2.waitKey(0)

    return windows


def draw_labeled_bboxes(img, labels):

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 6)

    return img


def heat_map(heatmap, windows, ratio):

    for window in windows:
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    maximum = np.max(heatmap)
    heatmap[heatmap <= maximum*ratio] = 0
    all_labels = label(heatmap)
    return all_labels


def parse_args():

    parser = argparse.ArgumentParser(description='Parser single image pipeline')
    parser.add_argument('-v', '--vehicles', dest="vpath", help='Location of vehicle images')
    parser.add_argument('-n', '--non-vehicles', dest="npath", help='Location of non-vehicle images')
    parser.add_argument('-s', '--save', dest='save', help='save outputs', action='store_true', default='True')
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':

    # Read arguments passed in by the user
    args = parse_args()

    # Read image and obtain windows
    files = glob("./test_images/test*.jpg")

    # Load model
    if not os.path.isfile('train.p'):
        svm, X_test, y_test, X_scaler = build_classifier(args.vpath, args.npath, 'train.p')
    else:
        train = pickle.load(open('train.p', 'rb'))  #
        svm = train["model"]
        X_test = train["X_test"]
        y_test = train["y_test"]
        scaler = train["X_scaler"]

    for filename in files:
        image = cv2.imread(filename)

        windows = find_cars(np.copy(image), ystart=380, ystop=500, xstart=600, scale=1, model=svm, scaler=scaler,
                            orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

        windows += find_cars(np.copy(image), ystart=400, ystop=600, xstart=600, scale=1.5, model=svm, scaler=scaler,
                            orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

        windows += find_cars(np.copy(image), ystart=400, ystop=656, xstart=600, scale=2, model=svm, scaler=scaler,
                            orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

        #windows += find_cars(np.copy(image), ystart=500, ystop=656, scale=3, model=svm, scaler=scaler,
        #                    orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

        # windows = slide_window(image, x_start_stop=[400, 656],
        #                        y_start_stop=[600, None], xy_window=(80, 80), xy_overlap=(0.75, 0.75))

        # Run model on target image
        # windows = slide_window(image,
        #                       x_start_stop=[600, None],
        #                       y_start_stop=[400, 656])
        # windows = search_in_windows(img=image, windows=windows, model=svm, scaler=scaler)

        # windows = find_cars(image, ystart=400, ystop=656, scale=1, model=svm, scaler=scaler,
        #           orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        labels = heat_map(heat, windows, 0.25)

        #overlaid = draw_boxes(image, windows)

        # Display results
        overlaid = draw_labeled_bboxes(np.copy(image), labels)
        cv2.imshow('boxes', overlaid)
        cv2.waitKey(0)
