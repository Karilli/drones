#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################

import cv2
import numpy as np

from src.draft_estimation.lib.Board import Board
from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator


KEY_ARROW_UP = 2490368
KEY_ARROW_DOWN = 2621440


def choose_kernel_radius(img):
    radius = 15

    cv2.namedWindow("Choose kernel radius.")
    board = Board(img, 3, 3)
    board.draw_img(img, 0, 0)
    h, w = img.shape[:2]

    def draw():
        nonlocal board, radius, w, h

        dm_seg = DraftMarkSegmentator(radius)
        try:
            dm_seg.run(img)
        except ValueError:
            pass

        kernel_img = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(kernel_img, (w//2, h//2), dm_seg.kernel_radius//2, Color.WHITE.value, -1)
        cv2.putText(kernel_img, "radius = " + str(radius), (w//2, h//2 - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Color.WHITE.value, 1, cv2.LINE_AA)

        board.draw_img(dm_seg.color_corrected, 0, 1)
        board.draw_img(kernel_img, 0, 2)
        board.draw_img(dm_seg.tophat_img, 1, 0)
        board.draw_img(dm_seg.blackhat_img, 2, 0)
        board.draw_img(dm_seg.tophat_bin_img, 1, 1)
        board.draw_img(dm_seg.blackhat_bin_img, 2, 1)
        board.draw_marks(list(filter(lambda x: x.tophat_flag, dm_seg.marks)), 1, 2)
        board.draw_marks(list(filter(lambda x: not x.tophat_flag, dm_seg.marks)), 2, 2)

        cv2.imshow("Choose kernel radius.", board.board)

    draw()
    while True:
        code = cv2.waitKeyEx(5)
        if code == KEY_ARROW_UP:
            radius = min(radius + 1, h//4, w//4)
            draw()
        elif code == KEY_ARROW_DOWN:
            radius = max(1, radius - 1)
            draw()
        elif code != -1:
            cv2.destroyAllWindows()
            return radius

        if not cv2.getWindowProperty("Choose kernel radius.", cv2.WND_PROP_VISIBLE):
            return radius


def main(img_path):
    org = cv2.imread(img_path)
    choose_kernel_radius(org)


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\01.png")
