import numpy as np

from math import sqrt

from src.draft_estimation.lib.DraftMarks import DraftMarkString
from src.draft_estimation.lib.OCR import TemplateOCR


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def are_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3] 
    x3, y3, x4, y4 = rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]
    return (x1 < x4) and (x2 > x3) and (y2 > y3) and (y1 < y4)


def area_of_overlap(rect1, rect2):
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3] 
    x3, y3, x4, y4 = rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]
    x_dist = (min(x2, x4) - max(x1, x3))
    y_dist = (min(y2, y4) - max(y1, y3))
    area = 0
    if x_dist > 0 and y_dist > 0:
        area = x_dist * y_dist
    return max(area, 0)


class DraftMarkRecognizer:
    def __init__(self):
        self.marks = None
        self.templates = TemplateOCR("..\\DroneProject\\data\\mark_templates")
        self.max_overlap_percentage = 0.2
        self.min_conf = 0.5
        self.mark_string_h_ratio = 0.85
        self.mark_string_dist = (0.5, 2)

    def get_mark_pairs(self):
        marks = list(self.marks)
        for i in range(len(marks)):
            for j in range(i+1, len(marks)):
                yield marks[i], marks[j]

    def join_mark_strings(self):
        # TODO: fine-tune
        (vert_dist, horz_dist), h_ratio = self.mark_string_dist, self.mark_string_h_ratio
        for m1, m2 in self.get_mark_pairs():
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            cx1, cy1 = m1.center()
            cx2, cy2 = m2.center()
            mean_h = (h1 + h2) / 2
            if (abs(cy1 - cy2) < mean_h * vert_dist) and (abs(cx1 - cx2) < mean_h * horz_dist) and (h_ratio <= h1/h2 <= 1 / h_ratio):
                m1.join_with(m2)
                self.marks.add(m1.mark_string)
                self.marks.remove(m1)
                self.marks.remove(m2)
        return self

    def eval_marks(self, marks):
        # TODO: fine-tune
        self.marks = marks
        for mark in self.marks:
            x, y, w, h = mark.rect
            if 0.1 <= w/h <= 0.5:
                mark.label, mark.conf = self.templates.eval_char(mark.materialize(), "17")
            elif 0.5 <= w/h <= 1.0:
                mark.label, mark.conf = self.templates.eval_char(mark.materialize(), "0123456789M")
            elif 1.0 <= w/h <= 2.0:
                mark.label, mark.conf = self.templates.eval_char(mark.materialize(), "023456789M")
        return self

    def conf_filter(self):
        # TODO: fine-tune
        mn = self.min_conf
        for mark in self.marks.copy():
            if mark.conf < mn:
                self.marks.remove(mark)
        return self

    def join_tophat_and_blackhat(self):
        for m1, m2 in self.get_mark_pairs():
            if not are_overlapping(m1.rect, m2.rect):
                continue
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            if area_of_overlap(m1.rect, m2.rect) / min(w1*h1, w2*h2) <= self.max_overlap_percentage:
                continue
            self.marks.remove(min(m1, m2, key=lambda x: x.conf))
        return self

    # def resolve_x(self):
    #     # filter by distance from med_x +- 2 * med_w
    #     med_x = np.median(list(map(lambda x: x.rect[0] + x.rect[2] // 2, self.marks)))
    #     med_w = np.median(list(map(lambda x: x.rect[3], self.marks)))
    #     area = (med_x - 2*med_w), 0, 4*med_w, 2000
    #     for mark in self.marks.copy():
    #         if not are_overlapping(mark.rect, area) and area_of_overlap(mark.rect, area) / min(mark.rect[2]*mark.rect[3], area[2]*area[3]) <= 1 - self.max_overlap_percentage:
    #             self.marks.remove(mark)
    #     return self

    def resolve_y_overlaps(self):
        for m1, m2 in self.get_mark_pairs():
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            if are_overlapping((0, y1, 3, h1), (1, y2, 3, h2)):
                self.marks.remove(min(m1, m2, key=lambda x: x.conf))
        return self

    def check_marks(self):
        # med_x = np.median(list(map(lambda x: x.rect[0], self.marks)))
        # sorted_marks = sorted(self.marks, key=lambda mark: distance(mark.rect[:2], (med_x, mark.rect[1])))

        # print(list(map(lambda x: x.label, sorted_marks)))
        # TODO: assert that marks are in correct order and distance between them
        # assert distances
        # assert h ratio
        # assert positional relation (M/0 -> 8 -> 6 -> 4 -> 2 -> M/0)
        # TODO: try to guess missing marks and look for them, also try to recover misssing mark strings
        # TODO: maybe assert that only two marks can be in a mark string?
        pass 

    def read_marks(self):
        marks = list(sorted(self.marks, key=lambda x: x.rect[1], reverse=True))

        found_count = 0
        for mark in marks:
            if isinstance(mark, DraftMarkString):
                value = int(mark.label.replace("M", "0")) - 2*found_count
                return marks, value
            found_count += 1

        raise ValueError

    def run(self, marks):
        self.eval_marks(marks).conf_filter()
        self.join_mark_strings().join_tophat_and_blackhat()
        self.resolve_y_overlaps().check_marks()
        self.templates.train(self.marks).save()
        return self.read_marks()


def main(img_path):
    #########################################################
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
    #########################################################

    import cProfile
    import pstats
    import cv2

    from src.draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator
    from src.draft_estimation.demo.DraftMarkSegmentationDemo import choose_kernel_radius

    img = cv2.imread(img_path)
    kernel_radius = choose_kernel_radius(img)
    marks = DraftMarkSegmentator(kernel_radius).run(img).marks

    with cProfile.Profile() as pr:
        DraftMarkRecognizer().run(marks)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


if __name__ == '__main__':
    main("..\\DroneProject\\data\\images\\01.png")
