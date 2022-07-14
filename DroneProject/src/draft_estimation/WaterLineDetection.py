import cv2
import numpy as np

from src.draft_estimation.lib.Regression import fit_and_predict_RANSAC


def is_white(arr):
    return len(arr) == 3 and (arr == 255).all()


class WaterLineDetector:
    def __init__(self, demo=False):
        self.demo = demo
        self.canny_args = (100, 200)
        self.open_kernel = (5, 1)
        self.regressor = None

    def find_water_line(self, img, bottom_mark):
        # TODO: crop the img
        # TODO: remove stationary lines
        canny = cv2.Canny(img, *self.canny_args)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.open_kernel)
        open_canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)
        open_canny = cv2.cvtColor(open_canny, cv2.COLOR_GRAY2RGB)

        x, y, w, h = bottom_mark.rect
        x1, y1, w1, h1 = x-2*h, y+2*h//3, 2*h, 3*h
        x2, y2, w2, h2 = x+w, y+2*h//3, 2*h, 3*h
        img_h, img_w = img.shape[:2]

        xs1 = np.array([x for y in range(max(0, y1), min(y1+h1, img_h)) for x in range(max(0, x1), min(x1+w1, img_w)) if is_white(open_canny[y, x])])
        ys1 = np.array([y for y in range(max(0, y1), min(y1+h1, img_h)) for x in range(max(0, x1), min(x1+w1, img_w)) if is_white(open_canny[y, x])])
        xs2 = np.array([x for y in range(max(0, y2), min(y2+h2, img_h)) for x in range(max(0, x2), min(x2+w2, img_w)) if is_white(open_canny[y, x])])
        ys2 = np.array([y for y in range(max(0, y2), min(y2+h2, img_h)) for x in range(max(0, x2), min(x2+w2, img_w)) if is_white(open_canny[y, x])])
        xs, ys = np.append(xs1, xs2), np.append(ys1, ys2)

        if len(xs) < 2:
            raise ValueError

        X = x + w // 2
        Y = fit_and_predict_RANSAC(xs, ys, np.array([X]))

        if self.demo:
            X0, X1 = x1, x2+w2
            Y0, Y1 = fit_and_predict_RANSAC(xs, ys, np.array([X0, X1]))
            self.canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            self.open_canny = open_canny
            self.rect1 = (x1, y1, w1, h1)
            self.rect2 = (x2, y2, w2, h2)
            self.line = [(X0, int(Y0)), (X1, int(Y1))]

        return X, int(Y)
    
    def run(self, img, sorted_marks):
        # TODO: find curve of marks and return intersection with waterline
        return self.find_water_line(img, sorted_marks[0])
    
    
