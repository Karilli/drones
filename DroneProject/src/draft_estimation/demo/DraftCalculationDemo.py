#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################


import cv2
import numpy as np

from src.draft_estimation.DraftCalculation import DraftCalculator
from src.draft_estimation.demo.DraftMarkSegmentationDemo import choose_kernel_radius
from src.draft_estimation.lib.Board import Board
from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.lib.ImageUtils import read


def main(img_path):
    img = read(img_path)
    board = Board(img, 1, 2)
    board.draw_img(img, 0, 0)

    kernel_radius = choose_kernel_radius(img)
    d_calc = DraftCalculator(kernel_radius)

    value = round(d_calc.run([img]), 2)
    for mark in d_calc.dm_rec.marks:
        mark.draw(img)

    if d_calc.pts:
        if len(d_calc.pts) >= 4:
            cv2.line(img, d_calc.pts[2], d_calc.pts[3], Color.YELLOW.value, 1, cv2.LINE_AA)
        cv2.line(img, d_calc.pts[1], d_calc.pts[2], Color.YELLOW.value, 1, cv2.LINE_AA)
        cv2.line(img, d_calc.pts[0], d_calc.pts[1], Color.YELLOW.value, 1, cv2.LINE_AA)

        x = d_calc.wl_det.rect1[0] + d_calc.wl_det.rect1[2] + (d_calc.wl_det.rect2[0] - d_calc.wl_det.rect1[0] - d_calc.wl_det.rect1[2]) // 2
        y = d_calc.wl_det.rect1[1] + d_calc.wl_det.rect1[3] // 2
        cv2.putText(img, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Color.YELLOW.value, 1, cv2.LINE_AA)
        
        h, w = img.shape[:2]
        x = np.arange(0, w-1, 0.001)
        f = d_calc.wl_det.water_line(x).astype(int)
        g = d_calc.wl_det.mark_curve(x).astype(int)
        for x, y1, y2 in zip(x.astype(int), g, f):
            if 0 <= x < w:
                if 0 <= y1 < h:
                    img[y1, x] = Color.YELLOW.value
                if 0 <= y2 < h:
                    img[y2, x] = Color.YELLOW.value
    board.draw_img(img, 0, 1)
    board.show()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\01.png")
