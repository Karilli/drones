import cv2
import numpy as np
from Regression import fit_and_predict_RANSAC, fit_and_predict_LMS


def is_white(arr):
    return (arr == 255).all() and len(arr) == 3


class WaterLineDetector:
    def find(self, img_path, bottom_mark, demo=False):
        img = cv2.imread(img_path)
        canny = cv2.Canny(img, 100, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        open_canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)
        open_canny = cv2.cvtColor(open_canny, cv2.COLOR_GRAY2RGB)
        
        x, y, w, h = bottom_mark.rect
        x1, y1, w1, h1 = x-2*h, y+2*h//3, 2*h, 3*h
        x2, y2, w2, h2 = x+w, y+2*h//3, 2*h, 3*h
        img_h, img_w, _ = img.shape

        xs1 = np.array([x for y in range(y1, min(y1+h1, img_h)) for x in range(x1, min(x1+w1, img_w)) if is_white(open_canny[y, x])])
        ys1 = np.array([y for y in range(y1, min(y1+h1, img_h)) for x in range(x1, min(x1+w1, img_w)) if is_white(open_canny[y, x])])
        xs2 = np.array([x for y in range(y2, min(y2+h2, img_h)) for x in range(x2, min(x2+w2, img_w)) if is_white(open_canny[y, x])])
        ys2 = np.array([y for y in range(y2, min(y2+h2, img_h)) for x in range(x2, min(x2+w2, img_w)) if is_white(open_canny[y, x])])
        xs, ys = np.append(xs1, xs2), np.append(ys1, ys2)

        X = x + w // 2
        Y = fit_and_predict_RANSAC(xs, ys, np.array([X]))

        if demo:
            X0, X1 = x1, x2+w2
            Y0, Y1 = fit_and_predict_RANSAC(xs, ys, np.array([X0, X1]))
            self.canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            self.open_canny = open_canny
            self.rect1 = (x1, y1, w1, h1)
            self.rect2 = (x2, y2, w2, h2)
            self.line = [(X0, int(Y0)), (X1, int(Y1))]

        return X, int(Y)
