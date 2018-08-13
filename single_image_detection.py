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


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
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
    # Return the image
    return img


def heat_map(heatmap, windows, threshold):
    for window in windows:
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    heatmap[heatmap <= threshold] = 0
    labels = label(heatmap)
    return labels


# Read image and obtain windows
files = glob("./test_images/test*.jpg")

# Load model
if not os.path.isfile('train.p'):
    svm, X_test, y_test, X_scaler = build_classifier(car, non_car, 'train.p')
else:
    train = pickle.load(open('train.p', 'rb'))  #
    svm = train["model"]
    X_test = train["X_test"]
    y_test = train["y_test"]
    X_scaler = train["X_scaler"]

for filename in files:
    image = cv2.imread(filename)
    windows = slide_window(image, x_start_stop=[np.int(8*image.shape[0]/16), None],
                           y_start_stop=[0, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5))


    # Run model on target image

    # Display results
    good_wins = search_in_windows(img=image, windows=windows, model=svm, scaler=X_scaler)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    labels = heat_map(heat, good_wins, 1)
    # overlaid = draw_boxes(image, good_wins)
    overlaid = draw_labeled_bboxes(np.copy(image), labels)
    cv2.imshow('boxes', overlaid)
    cv2.waitKey(0)
