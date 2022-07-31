from src.draft_estimation.lib.DraftMarks import DraftMarkString, DraftMark
from src.draft_estimation.lib.OCR import TemplateOCR
from src.draft_estimation.lib.SequenceMatcher import match_sequences
from src.draft_estimation.lib.Grid2DMath import are_overlapping, overlap_area_percentage, distance_y, distance_x
from src.draft_estimation.Constants import TEMPLATES_PATH, STRING_MAX_VERT_DIST, STRING_MAX_HORZ_DIST, STRING_MIN_H_RATIO, STRING_MAX_H_RATIO, MARK_W_TO_H_RATIOS, MARK_MIN_CONF, MARK_MAX_OVERLAP_PERCENTAGE


class DraftMarkRecognizer:
    def __init__(self):
        self.marks = None
        self.templates = TemplateOCR(TEMPLATES_PATH)

    def join_strings(self):
        for m1, m2 in self.marks.pairs(False):
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            mean_h = (h1 + h2) / 2
            if (distance_y(m1.center(), m2.center()) < mean_h * STRING_MAX_VERT_DIST) and (distance_x(m1.center(), m2.center()) < mean_h * STRING_MAX_HORZ_DIST) and (STRING_MIN_H_RATIO <= h1/h2 <= STRING_MAX_H_RATIO):
                m1.join_with(m2)
                self.marks.add(m1.string)
        return self

    def eval_marks(self, marks):
        self.marks = marks
        for mark in self.marks.marks:
            _, _, w, h = mark.rect
            alphabet = [c for c, (min_, max_) in MARK_W_TO_H_RATIOS if min_ <= w/h <= max_]
            mark.label, mark.conf = self.templates.eval_char(mark.materialize(), alphabet)
        return self

    def conf_filter(self):
        for mark in self.marks.marks.copy():
            if mark.conf < MARK_MIN_CONF:
                self.marks.remove(mark)
        return self

    def join_tophat_and_blackhat(self):
        for m1, m2 in self.marks.pairs(True):
            if are_overlapping(m1.rect, m2.rect):
                x1, y1, w1, h1 = m1.rect
                x2, y2, w2, h2 = m2.rect
                if overlap_area_percentage(m1.rect, m2.rect) >= MARK_MAX_OVERLAP_PERCENTAGE:
                    self.marks.remove(min(m1, m2, key=lambda x: x.conf))
        return self

    def resolve_y_overlaps(self):
        for m1, m2 in self.marks.pairs(True):
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            if are_overlapping((0, y1, 3, h1), (1, y2, 3, h2)):
                self.marks.remove(min(m1, m2, key=lambda x: x.conf))
        return self

    def match_marks(self):
        # TODO: look for removed marks and revive them
        matched_marks = set(match_sequences(self.marks))
        if not matched_marks:
            raise ValueError("Recognition fault")
        for m in self.marks.marks_and_strings():
            if m not in matched_marks:
                self.marks.remove(m)

    def read_marks(self):
        marks = self.marks.y_sorted(reverse=True, strings=True)
        # TODO: the above string mark is missing: e.g. [6, 8, X, 2, 4, 6, 8 3M]
        for mark in marks:
            if isinstance(mark, DraftMarkString):
                if mark is marks[0]:
                    value = int(mark.label.replace("M", "0"))
                else:
                    value = int(mark.label[0] + marks[0].label) - 10
                return marks, value
        raise ValueError("Recognition fault")

    def run(self, marks):
        self.eval_marks(marks).conf_filter()
        self.join_tophat_and_blackhat().join_strings()
        # TODO: fine-tune is resolve_y_overlaps neccesary ?
        self.resolve_y_overlaps().match_marks()
        self.templates.train(self.marks).save()
        return self.read_marks()
