import cv2
import numpy as np

from src.draft_estimation.lib.Regression import predict_RANSAC, fit_RANSAC, fit_polynom, predict_polynom
from src.draft_estimation.Constants import OPEN_KERNEL


class WaterLineDetector:
    def __init__(self):
        self.water_line = None
        self.rect1 = None
        self.rect2 = None
        self.mark_curve = None

        # for debugging and demos
        self.canny = None
        self.open_canny = None

    def fit_water_line(self, img, bottom_mark):
        # TODO: crop the img
        # TODO: remove stationary lines
        self.canny = cv2.Canny(img, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KERNEL)
        self.open_canny = cv2.morphologyEx(self.canny, cv2.MORPH_OPEN, kernel)

        x, y, w, h = bottom_mark.rect
        x1, y1, w1, h1 = x-2*h, y+2*h//3, 2*h, 3*h
        x2, y2, w2, h2 = x+w, y+2*h//3, 2*h, 3*h
        self.rect1 = (x1, y1, w1, h1)
        self.rect2 = (x2, y2, w2, h2)

        ys1, xs1 = np.where(self.open_canny[y1:y1+h1, x1:x1+w1] == 255)
        ys2, xs2 = np.where(self.open_canny[y2:y2+h2, x2:x2+w2] == 255)
        xs1, ys1 = xs1[::3] + x1, ys1[::3] + y1
        xs2, ys2 = xs2[::3] + x2, ys2[::3] + y2
        xs, ys = np.append(xs1, xs2), np.append(ys1, ys2)

        if len(xs) < 2:
            raise ValueError("Water line detection fault")

        r = fit_RANSAC(xs, ys)
        self.water_line = lambda x: predict_RANSAC(r, x)
        return self

    def fit_mark_curve(self, sorted_marks):
        # TODO: implement another approach: find median x-coord of center of marks (y = median x-coord)
        pts = list(map(lambda x: x.bottom(), sorted_marks))
        if len(sorted_marks) == 1:
            # y = q
            q = self.water_line(np.array([pts[0][0]]))[0]
            self.mark_curve = lambda _: q
            return self

        # y = kx + q
        xs = np.array([x for x, y in pts])
        ys = np.array([y for x, y in pts])
        self.mark_curve = fit_polynom(lambda x, k, q: k*x+q, xs, ys)
        return self

    @staticmethod
    def get_line_params(fnc):
        x1, x2 = 0, 10
        y1, y2 = fnc(np.array([x1]))[0], fnc(np.array([x2]))[0]
        k = (y2 - y1) / (x2 - x1)
        q = y1 - k*x1
        return k, q

    def line_intersection(self):
        k2, q2 = self.get_line_params(self.water_line)
        k1, q1 = self.get_line_params(self.mark_curve)
        x = (q1 - q2) / (k2 - k1)
        y = (q1*k2 - q2*k1) / (k2 - k1)
        return int(x), int(y)

    def run(self, img, sorted_marks):
        self.fit_water_line(img, sorted_marks[0])
        self.fit_mark_curve(sorted_marks)
        return self.line_intersection()
