import cv2
import numpy as np

from src.draft_estimation.lib.Colors import Color


def join_rects(rect_1, rect_2):
    x1, y1, w1, h1 = rect_1
    x2, y2, w2, h2 = rect_2
    x3, y3 = x1 + w1, y1 + h1
    x4, y4 = x2 + w2, y2 + h2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x3, x4) - x
    h = max(y3, y4) - y
    return x, y, w, h


class DraftMark:
    def __init__(self, rect, img, tophat_flag):
        self.rect = rect
        self.img = img
        self.tophat_flag = tophat_flag
        self.label = None
        self.conf = None
        self.materialized = None
        self.mark_string = None

    def materialize(self):
        if self.materialized is None:
            x, y, w, h = self.rect
            self.materialized = self.img[y:y+h, x:x+w]
        return self.materialized

    def center(self):
        x, y, w, h = self.rect
        return x + w // 2, y - h // 2

    def join_with(self, other):
        if self.mark_string and other.mark_string:
            for mark in other.mark_string.marks:
                self.mark_string.add(mark)
        elif self.mark_string:
            self.mark_string.add(other)
        elif other.mark_string:
            other.mark_string.add(self)
            self.mark_string = other.mark_string
        else:
            DraftMarkString(self, other)
        return self

    def draw(self, img):
        if self.mark_string:
            self.mark_string.draw(img)
            return
        
        x, y, w, h = self.rect
        col = Color.GREEN.value if self.tophat_flag else Color.RED.value
        cv2.rectangle(img, (x, y), (x+w, y+h), col, 1)
        if self.label and self.conf:
            cv2.putText(img, str((self.label, round(self.conf, 2))), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return self


class DraftMarkString:
    def __init__(self, m1, m2):
        self.marks = [m1, m2]
        m1.mark_string = self
        m2.mark_string = self
        self.materialized = None
        self.label = self.comp_label()
        self.conf = self.comp_conf()
        self.rect = self.comp_rect()

    def comp_label(self):
        if any(mark.label is None for mark in self.marks):
            return None
        return "".join(mark.label for mark in sorted(self.marks, key=lambda x: x.rect[0]))

    def comp_conf(self):
        if any(mark.conf is None for mark in self.marks):
            return None
        # TODO: fine-tune, mean or max?
        return max(mark.conf for mark in self.marks)

    def comp_rect(self):
        rect = self.marks[0].rect
        for i in range(1, len(self.marks)):
            rect = join_rects(rect, self.marks[i].rect)
        return rect

    def draw(self, img):
        x, y, w, h = self.rect
        cv2.rectangle(img, (x, y), (x+w, y+h), Color.BLUE.value, 1)
        if self.label and self.conf:
            cv2.putText(img, str((self.label, round(self.conf, 2))), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Color.BLUE.value, 1, cv2.LINE_AA)
        return self
    
    def add(self, other):
        self.marks.append(other)
        other.mark_string = self
        self.label = self.comp_label()
        self.conf = self.comp_conf()
        self.rect = self.comp_rect()

    def materialize(self):
        x1, y1, w1, h1 = self.rect
        self.materialized = np.zeros((h1, w1), dtype=np.uint8)
        for mark in self.marks:
            x2, y2, w2, h2 = mark.rect
            x, y, w, h = x2-x1, y2-y1, w2, h2
            self.materialized[y:y+h, x:x+w] = mark.materialize() 
        return self.materialized
