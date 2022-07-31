#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################

from src.draft_estimation.DraftMarkRecognition import DraftMarkRecognizer
from src.draft_estimation.WaterLineDetection import WaterLineDetector
from src.draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator
# TODO: maybe use distance instead of distance_y?
from src.draft_estimation.lib.Grid2DMath import distance_y
from src.draft_estimation.lib.SequenceMatcher import mark_dist
from src.draft_estimation.lib.ImageUtils import read

from scipy.signal import medfilt


class DraftCalculator:
    def __init__(self, kernel_radius):
        self.dm_seg = DraftMarkSegmentator(kernel_radius)
        self.dm_rec = DraftMarkRecognizer()
        self.wl_det = WaterLineDetector()

        self.pts = []

    def calc_draft(self, p0, marks, value):
        # TODO: this sucks when there are irregularities as in 02.png
        if len(marks) >= 3:
            self.pts = [p0] + list(map(lambda x: x.bottom(), marks[:3]))
            d2 = 2 * distance_y(self.pts[2], self.pts[3]) / (mark_dist(marks[1], marks[2]))
            d1 = 2 * distance_y(self.pts[1], self.pts[2]) / (mark_dist(marks[0], marks[1]))
            d0 = distance_y(self.pts[0], self.pts[1])
            dist = 2 * (d0 * d2) / (d1**2)

        elif len(marks) == 2:
            self.pts = [p0, marks[0].bottom(), marks[0].top(), marks[1].bottom()]
            d2 = distance_y(self.pts[2], self.pts[3]) * 2 / (mark_dist(marks[1], marks[0]) - 1)
            d1 = distance_y(self.pts[1], self.pts[2]) * 2
            d0 = distance_y(self.pts[0], self.pts[1])
            dist = 2 * (d0 * d2) / (d1**2)

        else:
            self.pts = [p0, marks[0].bottom(), marks[0].top()]
            d1 = distance_y(self.pts[1], self.pts[2]) * 2
            d0 = distance_y(self.pts[0], self.pts[1])
            dist = 2 * (d0) / (d1)

        if self.pts[0][1] < self.pts[1][1]:
            return value + dist
        return value - dist


    def run(self, frames):
        measures = []
        for frame in frames:
            try:
                marks = self.dm_seg.run(frame)
                marks, value = self.dm_rec.run(marks)
                p0 = self.wl_det.run(frame, marks)
                draft = self.calc_draft(p0, marks, value)
                measures.append(draft)
            except ValueError as e:
                print(e)
                if str(e) == "Segmentation fault":
                    pass # TODO: Couldn't find any draft mark segments
                elif str(e) == "Recognition fault":
                    pass # TODO: Couldn't find mark string, might need to loosen params
                elif str(e) == "Water line detection fault":
                    pass # TODO: Couldn't find waterline, might need to loosen params

        if len(measures) >= 10:
            measures = medfilt(measures)
        if len(measures) == 0:
            return 0
        return sum(measures) / len(measures)


def main(img_path):
    import cProfile
    import pstats
    import cv2

    from src.draft_estimation.demo.DraftMarkSegmentationDemo import choose_kernel_radius

    img = read(img_path)
    print("h x w:", img.shape[:2])
    kernel_radius = choose_kernel_radius(img)
    d_calc = DraftCalculator(kernel_radius)

    with cProfile.Profile() as pr:
        d_calc.run([img])

    stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\01_UHD.png")
