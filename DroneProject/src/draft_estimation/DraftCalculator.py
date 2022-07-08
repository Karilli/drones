from DraftMarkRecognition import DraftMarkRecognizer
from WaterLineDetection import WaterLineDetector


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    

class DraftCalculator:
    def get_points(self, sorted_marks):
        p0, p1, p2, p3 = None, None, None, None
        value, found_string = None, False
        for i, mark in enumerate(sorted_marks):
            if i == 0:
                p1 = mark.rect[:2]
                p0 = WaterLineDetector().find(frame, mark)
            elif i == 1:
                p2 = mark.rect[:2]
            elif i == 2:
                p3 = mark.rect[:2]
            if mark.mark_string and not found_string:
                found_string = True
                value = int(mark.mark_string.label.replace("M", "0")) - 2*i
            if found_string and i > 2:
                break

        if not found_string:
            raise ValueError
        
        return p0, p1, p2, p3, value

    def get_distances(p0, p1, p2, p3):
        d0, d1, d2 = None, None, None
        if p3:
            d2 = distance(p2, p3)
        if p2:
            d1 = distance(p1, p2)
        if p1:
            d0 = distance(p0, p1)
        return d0, d1, d2

    def calc_draft(self, d0, d1, d2, value):
        if d2:
            return value - 0.2 * d0 / (d1**2 / d2)
        if d1:
            return value - 0.2 * d0 / d1
        return d0

    def filter_measures(self, measures):
        pass # TODO

    def get_draft(self, frames):
        measures = []
        for frame in frames:
            p0, p1, p2, p3, value = self.get_points(DraftMarkRecognizer().run(frame).get_y_sorted())
            d0, d1, d2 = self.get_distances(p0, p1, p2, p3)
            measures.append(self.calc_draft(d0, d1, d2, value))
            self.filter(measures)
        return sum(measures) / len(measures)
