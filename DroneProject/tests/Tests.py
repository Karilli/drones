#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################

from src.draft_estimation.lib.Grid2DMath import are_overlapping, overlap_area, join, distance
from tests.Pairs import OVERLAPPING_PAIRS, NONOVERLAPPING_PAIRS


def test_distance():
    assert round(distance((0, 0), (23, 45)), 2) == 50.54
    assert distance((2, 4), (5, 4)) == 3
    assert round(distance((9, 4), (2, 9)), 2) == 8.6


def test_join():
    for pair, joined, _ in OVERLAPPING_PAIRS + NONOVERLAPPING_PAIRS:
        assert join(*pair) == joined, f"joined {pair} != {joined} instead got {join(*pair)}"


def test_are_overlapping():
    for pair, _, _ in OVERLAPPING_PAIRS:
        assert are_overlapping(*pair), f"{pair} evaluated as non-overlapping instead of overlapping"
    for pair, _, _ in NONOVERLAPPING_PAIRS:
        assert not are_overlapping(*pair), f"{pair} evaluated as overlapping instead of non-overlapping"


def test_overlap_area():
    for pair, _, area in OVERLAPPING_PAIRS + NONOVERLAPPING_PAIRS:
        assert overlap_area(*pair) == area, f"area of overlap of {pair} != {area} instead got {overlap_area(*pair)}"


def main():
    test_join()
    test_distance()
    test_are_overlapping()
    test_overlap_area()
    print("Test succesfully completed")


if __name__ == "__main__":
    main()
