from glob import glob
import cv2
import numpy as np


def draw_boxes(img, boxes, color=(255, 0, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, thickness)

    return img


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
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

    x_indices = [x * x_stride for x in range(np.int(x_start_stop[1] / x_stride) - 1)]
    y_indices = [y * y_stride for y in range(np.int(y_start_stop[1] / y_stride) - 1)]

    boxes = []
    [boxes.append(((y, x), (y + xy_window[1], x + xy_window[0]))) for x in x_indices for y in y_indices]

    return boxes


image = cv2.imread("./test_images/test1.jpg")
windows = slide_window(image)
overlaid = draw_boxes(image, windows)

cv2.imshow('boxes', overlaid)
cv2.waitKey(0)
