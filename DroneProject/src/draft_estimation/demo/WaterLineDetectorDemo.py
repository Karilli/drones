#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################

import cv2
import numpy as np

from src.draft_estimation.lib.Board import Board
from src.draft_estimation.lib.DraftMarks import DraftMark
from src.draft_estimation.WaterLineDetection import WaterLineDetector
from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.lib.ImageUtils import overlap_imgs, read


def get_rect(img):
    rect = None
    x1, y1 = -1, -1
    drawing = False

    def draw_rect(event, x, y, flags, param):
        nonlocal rect, x1, y1, drawing, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif drawing and event == cv2.EVENT_MOUSEMOVE:
            cv2.rectangle(img, (x1, y1), (x, y), Color.GREEN.value, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (x1, y1), (x, y), Color.GREEN.value, -1)
            rect = x1, y1, x - x1, y - y1

    cv2.namedWindow("Select lowest mark.")
    cv2.setMouseCallback("Select lowest mark.", draw_rect)
    while True:
        cv2.imshow("Select lowest mark.", img)
        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            return rect


def main(img_path):
    img = read(img_path)
    board = Board(img, 2, 2)
    board.draw_img(img, 0, 0)

    x, y, w, h = get_rect(img.copy())
    det = WaterLineDetector()
    marks = [DraftMark((x, y, w, h), img, True)]
    X, Y = det.run(img, marks)

    board.draw_img(det.canny, 0, 1)
    board.draw_img(det.open_canny, 1, 0)

    overlap_imgs(img, [det.rect1, det.rect2], cv2.cvtColor(det.open_canny, cv2.COLOR_GRAY2BGR))
    cv2.line(img, (X, Y), (x + w//2, y+h), Color.YELLOW.value, 2, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), Color.GREEN.value, 1)
    h, w = img.shape[:2]
    x = np.arange(0, w-1, 0.001)
    f = det.water_line(x).astype(int)
    g = det.mark_curve(x).astype(int)
    for x, y1, y2 in zip(x.astype(int), g, f):
        if 0 <= x < w:
            if 0 <= y1 < h:
                img[y1, x] = Color.YELLOW.value
            if 0 <= y2 < h:
                img[y2, x] = Color.YELLOW.value
    board.draw_img(img, 1, 1)

    board.show()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\02.png")
