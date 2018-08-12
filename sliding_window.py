from glob import glob
import cv2


def draw_boxes(img, boxes, color=(255, 0, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, thickness)

    return img



img = cv2.imread("./test_images/test1.jpg")
boxes = (((100, 100), (200, 200)), ((300, 500), (400, 700)), ((1000, 100), (1280, 300)))
img = draw_boxes(img, boxes)
cv2.imshow('boxes', img)
cv2.waitKey(0)
