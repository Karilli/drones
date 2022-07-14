#########################################################
import os                                              ##
import sys                                             ##
sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#########################################################


import cv2

from src.draft_estimation.lib.Board import Board
from DraftMarkSegmentationDemo import choose_kernel_radius

from src.draft_estimation.DraftCalculation import DraftCalculator
from src.draft_estimation.lib.Colors import Color


def main(img_path):
    img = cv2.imread(img_path)
    board = Board(img, 1, 2)
    board.draw_img(img, 0, 0)

    kernel_radius = choose_kernel_radius(img)
    d_calc = DraftCalculator(kernel_radius, True)

    value = round(d_calc.run([img]), 2)
    for mark in d_calc.dm_rec.marks:
        mark.draw(img)

    if len(d_calc.pts) >= 4:
        cv2.line(img, d_calc.pts[2], d_calc.pts[3], Color.YELLOW.value, 1, cv2.LINE_AA)
    cv2.line(img, d_calc.pts[1], d_calc.pts[2], Color.YELLOW.value, 1, cv2.LINE_AA)
    cv2.line(img, d_calc.pts[0], d_calc.pts[1], Color.YELLOW.value, 1, cv2.LINE_AA)

    cv2.line(img, d_calc.wl_det.line[0], d_calc.wl_det.line[1], Color.BLUE.value, 1, cv2.LINE_AA)
    cv2.putText(img, str(value), d_calc.wl_det.line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, Color.BLUE.value, 1, cv2.LINE_AA)
    board.draw_img(img, 0, 1)
    board.show()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\05.png")
