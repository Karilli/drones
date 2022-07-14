#########################################################
import os                                              ##
import sys                                             ##
sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#########################################################

import cv2
import numpy as np

from src.draft_estimation.lib.DraftMarks import join_rects
from src.draft_estimation.DraftCalculation import distance
from src.draft_estimation.DraftMarkRecognition import are_overlapping, area_of_overlap


def test_distance():
    assert round(distance((0, 0), (23, 45)), 2) == 50.54
    assert distance((2, 4), (5, 4)) == 3
    assert round(distance((9, 4), (2, 9)), 2) == 8.6


def show(rects):
    img = np.zeros((100, 100), dtype=np.uint8)
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, 1)
    cv2.imshow("", img)
    cv2.waitKey(0)

PAIR_1 = ((5, 10, 40, 60), (15, 20, 15, 30)), (5, 10, 40, 60), 450
PAIR_2 = ((30, 20, 30, 10), (40, 25, 10, 15)), (30, 20, 30, 20), 50
PAIR_3 = ((10, 20, 30, 30), (25, 30, 35, 40)), (10, 20, 50, 50), 300
PAIR_4 = ((20, 30, 20, 20), (35, 40, 35, 5)), (20, 30, 50, 20), 25
PAIR_5 = ((20, 10, 40, 40), (5, 30, 25, 30)), (5, 10, 55, 50), 200

PAIR_6 = ((10, 20, 60, 50), (20, 10, 10, 20)), (10, 10, 60, 60), 100
PAIR_7 = ((10, 15, 20, 35), (20, 30, 30, 40)), (10, 15, 40, 55), 200
PAIR_8 = ((5, 20, 25, 10), (15, 10, 25, 30)), (5, 10, 35, 30), 150
PAIR_9 = ((5, 25, 45, 15), (30, 10, 40, 20)), (5, 10, 65, 30), 100
PAIR_10 = ((30, 30, 10, 50), (20, 20, 30, 50)), (20, 20, 30, 60), 400
PAIR_11 = ((30, 40, 50, 10), (10, 30, 50, 50)), (10, 30, 70, 50), 300
PAIR_12 = ((30, 30, 30, 40), (10, 60, 70, 30)), (10, 30, 70, 60), 300
PAIR_13 = ((10, 20, 40, 30), (30, 10, 50, 60)), (10, 10, 70, 60), 600


OVERLAPPING_PAIRS = [PAIR_1, PAIR_2, PAIR_3, PAIR_4, PAIR_5, PAIR_6, PAIR_7, PAIR_8, PAIR_9, PAIR_10, PAIR_11, PAIR_12, PAIR_13]


PAIR_21 = ((20, 10, 40, 20), (30, 50, 20, 30)), (20, 10, 40, 70), 0
PAIR_22 = ((5, 15, 45, 35), (70, 70, 25, 10)), (5, 15, 90, 65), 0
PAIR_23 = ((25, 30, 35, 40), (70, 45, 20, 15)), (25, 30, 65, 40), 0
PAIR_24 = ((55, 30, 35, 10), (15, 50, 25, 40)), (15, 30, 75, 60), 0


NONOVERLAPPING_PAIRS = [PAIR_21, PAIR_22, PAIR_23, PAIR_24]


def test_join_rects():
    for pair, joined, _ in OVERLAPPING_PAIRS + NONOVERLAPPING_PAIRS:
        assert join_rects(*pair) == joined, f"joined {pair} != {joined} instead got {join_rects(*pair)}"


def test_are_overlapping():
    for pair, _, _ in OVERLAPPING_PAIRS:
        assert are_overlapping(*pair), f"{pair} evaluated as non-overlapping instead of overlapping"
    for pair, _, _ in NONOVERLAPPING_PAIRS:
        assert not are_overlapping(*pair), f"{pair} evaluated as overlapping instead of non-overlapping"


def test_area_of_overlap():
    for pair, _, area in OVERLAPPING_PAIRS + NONOVERLAPPING_PAIRS:
        assert area_of_overlap(*pair) == area, f"area of overlap of {pair} != {area} instead got {area_of_overlap(*pair)}"


def main():
    test_join_rects()
    test_distance()
    test_are_overlapping()
    test_area_of_overlap()
    print("Test succesfully completed")


if __name__ == "__main__":
    main()
   