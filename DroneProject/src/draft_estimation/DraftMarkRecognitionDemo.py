import cv2
from DraftMarkRecognition import DraftMarkRecognizer, HEIGHT_OF_MARK_CM
from Board import Board


def main(img_path):
    rec = DraftMarkRecognizer()

    org = cv2.imread(img_path)
    board = Board(org, 3, 3)
    board.draw_img(org, 0, 0)

    num_of_marks = 2
    img_h_cm = HEIGHT_OF_MARK_CM * (2 * num_of_marks - 1)
    rec.search_marks(img_path, img_h_cm).w_to_h_ratio_filter(0.1, 2).area_filter(50, 2000)
    board.draw_marks(list(filter(lambda x: x.tophat_flag, rec.marks)), 1, 0)
    board.draw_marks(list(filter(lambda x: not x.tophat_flag, rec.marks)), 1, 1)

    rec.eval_marks().conf_filter(0)
    board.draw_marks(list(filter(lambda x: x.tophat_flag, rec.marks)), 2, 0)
    board.draw_marks(list(filter(lambda x: not x.tophat_flag, rec.marks)), 2, 1)

    rec.join_tophat_and_blackhat().join_mark_strings(0.25, 2, 0.9)
    board.draw_marks(rec.marks, 1, 2)

    rec.resolve_y_overlaps().resurrect_missing_marks()
    board.draw_marks(rec.marks, 2, 2)

    board.show()


if __name__ == '__main__':
    main("..\\data\\images\\image_01.png")
