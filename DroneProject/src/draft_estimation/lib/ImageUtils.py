import cv2
import numpy as np


def overlap_imgs(org, rects, new):
    for rect in rects:
        x, y, w, h = rect
        org[y:y+h, x:x+w] = new[y:y+h, x:x+w]


def resize_to_full_screen_factor(img):
    h, w = img.shape[:2]
    return min(600 / h, 1400 / w)


def resize_to_full_screen(img):
    factor = resize_to_full_screen_factor(img)
    return cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def normalize_img_0_1(img):
    return (img - np.min(img)) / np.ptp(img)


def normalize_img_0_255(img):
    return (255*(normalize_img_0_1(img))).astype(np.uint8) 


def read(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Invalid path.")
        exit(1)
    return img
