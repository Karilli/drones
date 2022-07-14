from src.draft_estimation.DraftMarkRecognition import DraftMarkRecognizer
from src.draft_estimation.WaterLineDetection import WaterLineDetector
from src.draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator

from scipy.signal import medfilt
from math import sqrt


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class DraftCalculator:
    def __init__(self, kernel_radius, demo=False):
        self.dm_seg = DraftMarkSegmentator(kernel_radius, demo)
        self.dm_rec = DraftMarkRecognizer()
        self.wl_det = WaterLineDetector(demo)
        self.demo = demo

    @staticmethod
    def get_bottom_pt(mark):
        x, y, w, h = mark.rect
        return (x+w//2, y+h)
    
    @staticmethod
    def get_top_pt(mark):
        x, y, w, h = mark.rect
        return (x+w//2, y)

    def calc_draft(self, p0, marks, value):
        if len(marks) >= 3:
            self.pts = [p0] + list(map(self.get_bottom_pt, marks[:3]))
            d2 = distance(self.pts[2], self.pts[3])
            d1 = distance(self.pts[1], self.pts[2])
            d0 = distance(self.pts[0], self.pts[1])
            dist = (2 * d0 * d2) / d1**2

        elif len(marks) == 2:
            self.pts = [p0, self.get_bottom_pt(marks[0]), self.get_top_pt(marks[0]), self.get_bottom_pt(marks[1])]
            d2 = distance(self.pts[2], self.pts[3])
            d1 = distance(self.pts[1], self.pts[2])
            d0 = distance(self.pts[0], self.pts[1])
            dist = (d0 * d2) / d1**2

        else:
            self.pts = [p0, self.get_bottom_pt(marks[0]), self.get_top_pt(marks[0])]
            d1 = distance(self.pts[1], self.pts[2])
            d0 = distance(self.pts[0], self.pts[1])
            dist = d0 / d1

        if self.pts[0][1] < self.pts[1][1]:
            return value + dist
        return value - dist


    def run(self, frames):
        measures = []
        for frame in frames:
            marks = self.dm_seg.run(frame)
            try:
                marks, value = self.dm_rec.run(marks)
            except ValueError:
                continue # TODO: Couldn't find mark string, might need to loosen params
            try:
                p0 = self.wl_det.run(frame, marks)
            except ValueError:
                continue # TODO: Couldn't find waterline, might need to loosen params
            draft = self.calc_draft(p0, marks, value)
            measures.append(draft)

        if len(measures) >= 10:
            measures = medfilt(measures)
        return sum(measures) / len(measures)
