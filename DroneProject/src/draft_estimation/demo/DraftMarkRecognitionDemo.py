#########################################################
import os                                              ##
import sys                                             ##
sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#########################################################


import cv2
import numpy as np

from src.draft_estimation.lib.Board import Board
from src.draft_estimation.lib.Colors import Color
from DraftMarkSegmentationDemo import choose_kernel_radius

from src.draft_estimation.DraftMarkRecognition import DraftMarkRecognizer
from src.draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator


def main(img_path):
    org = cv2.imread(img_path)
    board = Board(org, 3, 3)
    board.draw_img(org, 0, 0)

    kernel_radius = choose_kernel_radius(org)
    marks = DraftMarkSegmentator(kernel_radius).run(org)
    dm_rec = DraftMarkRecognizer()

    board.draw_marks(list(filter(lambda x: x.tophat_flag, marks)), 0, 1)
    board.draw_marks(list(filter(lambda x: not x.tophat_flag, marks)), 0, 2)

    dm_rec.eval_marks(marks).conf_filter()
    tophat_marks = list(filter(lambda x: x.tophat_flag, dm_rec.marks))
    blackhat_marks = list(filter(lambda x: not x.tophat_flag, dm_rec.marks))
    board.draw_marks(tophat_marks, 1, 1)
    board.draw_marks(blackhat_marks, 1, 2)

    dm_rec.join_tophat_and_blackhat().join_mark_strings().resolve_y_overlaps()
    board.draw_marks(dm_rec.marks, 2, 1)

    # med_x = np.median(list(map(lambda x: x.rect[0] + x.rect[2] // 2, dm_rec.marks)))
    # med_w = np.median(list(map(lambda x: x.rect[3], dm_rec.marks)))
    # if not np.isnan(med_x):
    #     board.draw_line((int(med_x - 2 * med_w), 0), (int(med_x - 2 * med_w), board.h), Color.PINK.value, 1, 2, 1)
    #     board.draw_line((int(med_x + 2 * med_w), 0), (int(med_x + 2 * med_w), board.h), Color.PINK.value, 1, 2, 1)
    #     board.draw_line((int(med_x), 0), (int(med_x), board.h), Color.YELLOW.value, 1, 2, 1)

    dm_rec.check_marks()
    board.draw_marks(dm_rec.marks, 2, 2)

    board.show()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\07.png")
