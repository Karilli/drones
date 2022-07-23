from tkinter import W
import cv2
import numpy as np

from src.draft_estimation.lib.Regression import fit_and_predict_RANSAC


class WaterLineDetector:
    def __init__(self):
        self.canny_args = (100, 200)
        self.open_kernel = (10, 1)
        self.regressor = None

        # for debugging and demos
        self.canny = None
        self.open_canny = None
        self.rect1 = None
        self.rect2 = None
        self.line = None

    def find_water_line(self, img, bottom_mark):
        # TODO: crop the img
        # TODO: remove stationary lines
        self.canny = cv2.Canny(img, *self.canny_args)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.open_kernel)
        self.open_canny = cv2.morphologyEx(self.canny, cv2.MORPH_OPEN, kernel)
        self.open_canny = cv2.cvtColor(self.open_canny, cv2.COLOR_GRAY2RGB)

        x, y, w, h = bottom_mark.rect
        x1, y1, w1, h1 = x-2*h, y+2*h//3, 2*h, 3*h
        x2, y2, w2, h2 = x+w, y+2*h//3, 2*h, 3*h
        self.rect1 = (x1, y1, w1, h1)
        self.rect2 = (x2, y2, w2, h2)

        ys1, xs1, _ = np.where(self.open_canny[y1:y1+h1, x1:x1+w1] == 255)
        ys2, xs2, _ = np.where(self.open_canny[y2:y2+h2, x2:x2+w2] == 255)
        xs1, ys1 = xs1[::3] + x1, ys1[::3] + y1
        xs2, ys2 = xs2[::3] + x2, ys2[::3] + y2
        xs, ys = np.append(xs1, xs2), np.append(ys1, ys2)

        if len(xs) < 2:
            raise ValueError("Water line detection fault")

        X = x + w // 2
        Y = fit_and_predict_RANSAC(xs, ys, np.array([X]))

        # for debugging and demos
        X0, X1 = x1, x2+w2
        Y0, Y1 = fit_and_predict_RANSAC(xs, ys, np.array([X0, X1]))
        self.line = [(X0, int(Y0)), (X1, int(Y1))]

        return X, int(Y)
    
    def run(self, img, sorted_marks):
        # TODO: search for curve of marks and return intersection with waterline
        return self.find_water_line(img, sorted_marks[0])
    
    
