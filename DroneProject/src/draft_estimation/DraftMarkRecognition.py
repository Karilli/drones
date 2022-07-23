from src.draft_estimation.lib.DraftMarks import DraftMarkString, DraftMark
from src.draft_estimation.lib.OCR import TemplateOCR
from src.draft_estimation.lib.SequenceMatcher import match_sequences
from src.draft_estimation.lib.Grid2DMath import are_overlapping, overlap_area_percentage


class DraftMarkRecognizer:
    def __init__(self):
        self.marks = None
        self.templates = TemplateOCR("..\\DroneProject\\data\\templates\\xor")
        self.max_overlap_percentage = 0.2
        self.min_conf = 0.5
        self.string_h_ratio = 0.85
        self.string_dist = (0.5, 2)

    def join_strings(self):
        # TODO: fine-tune
        (vert_dist, horz_dist), h_ratio = self.string_dist, self.string_h_ratio
        for m1, m2 in self.marks.pairs(False):
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            cx1, cy1 = m1.center()
            cx2, cy2 = m2.center()
            mean_h = (h1 + h2) / 2
            if (abs(cy1 - cy2) < mean_h * vert_dist) and (abs(cx1 - cx2) < mean_h * horz_dist) and (h_ratio <= h1/h2 <= 1 / h_ratio):
                m1.join_with(m2)
                self.marks.add(m1.string)
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
        for mark in self.marks.copy():
            if mark.conf < self.min_conf:
                self.marks.remove(mark)
        return self

    def join_tophat_and_blackhat(self):
        for m1, m2 in self.marks.pairs(True):
            if are_overlapping(m1.rect, m2.rect):
                x1, y1, w1, h1 = m1.rect
                x2, y2, w2, h2 = m2.rect
                if overlap_area_percentage(m1.rect, m2.rect) >= self.max_overlap_percentage:
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
        # TODO: fine-tune
        # TODO: look for removed marks
        matched_marks = set(match_sequences(self.marks, 2))
        if not matched_marks:
            raise ValueError("Recognition fault")
        for m in self.marks.copy():
            if m not in matched_marks:
                self.marks.remove(m)

    def read_marks(self):
        marks = self.marks.y_sorted(reverse=True, strings=True)
        for mark in marks:
            if isinstance(mark, DraftMarkString):
                value = int(mark.label[0] + marks[0].label)
                return marks, value
        raise ValueError("Recognition fault")

    def run(self, marks):
        self.eval_marks(marks).conf_filter()
        self.join_tophat_and_blackhat().join_strings()
        self.resolve_y_overlaps().match_marks()
        self.templates.train(self.marks).save()
        return self.read_marks()
