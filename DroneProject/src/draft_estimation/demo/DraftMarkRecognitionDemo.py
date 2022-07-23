#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################


import cv2

from src.draft_estimation.lib.Board import Board
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

    dm_rec.join_tophat_and_blackhat()
    board.draw_marks(dm_rec.marks, 1, 0)
    
    dm_rec.join_strings()
    board.draw_marks(dm_rec.marks, 2, 0)

    dm_rec.resolve_y_overlaps()
    board.draw_marks(dm_rec.marks, 2, 1)
    
    dm_rec.match_marks()
    board.draw_marks(dm_rec.marks, 2, 2)

    board.show()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\01.png")
