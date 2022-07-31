import cv2
import numpy as np

from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.lib.Grid2DMath import join


class DraftMark:
    def __init__(self, rect, img, tophat_flag):
        self.rect = rect
        self.img = img
        self.tophat_flag = tophat_flag
        self.label = None
        self.conf = None
        self.materialized = None
        self.string = None

    def materialize(self):
        if self.materialized is None:
            x, y, w, h = self.rect
            self.materialized = self.img[y:y+h, x:x+w]
        return self.materialized

    def center(self):
        x, y, w, h = self.rect
        return x + w // 2, y - h // 2

    def join_with(self, other):
        if self.string and other.string:
            if self.string is other.string:
                return self
            for mark in other.string.marks:
                self.string.add(mark)
        elif self.string:
            self.string.add(other)
        elif other.string:
            other.string.add(self)
            self.string = other.string
        else:
            DraftMarkString(self, other)
        return self

    def draw(self, img):
        if self.string:
            self.string.draw(img)
            return

        x, y, w, h = self.rect
        col = Color.GREEN.value if self.tophat_flag else Color.RED.value
        cv2.rectangle(img, (x, y), (x+w, y+h), col, 1)
        if self.label and self.conf:
            cv2.putText(img, str((self.label, round(self.conf, 2))), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return self

    def __repr__(self):
        return f"{self.label}"

    def __hash__(self):
        return hash(self.rect)

    def __eq__(self, other):
        return self.rect == other.rect
    
    def bottom(self):
        x, y, w, h = self.rect
        return (x+w//2, y+h-1)
    
    def top(self):
        x, y, w, h = self.rect
        return (x+w//2, y+1)


class DraftMarkString:
    def __init__(self, m1, m2):
        self.marks = [m1, m2]
        m1.string = self
        m2.string = self
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
        # TODO: fine-tune mean or max?
        return max(mark.conf for mark in self.marks)

    def comp_rect(self):
        rect = self.marks[0].rect
        for i in range(1, len(self.marks)):
            rect = join(rect, self.marks[i].rect)
        return rect

    def draw(self, img):
        x, y, w, h = self.rect
        cv2.rectangle(img, (x, y), (x+w, y+h), Color.BLUE.value, 1)
        if self.label and self.conf:
            cv2.putText(img, str((self.label, round(self.conf, 2))), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Color.BLUE.value, 1, cv2.LINE_AA)
        return self

    def add(self, other):
        self.marks.append(other)
        other.string = self
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

    def center(self):
        x, y, w, h = self.rect
        return x + w // 2, y - h // 2

    def __repr__(self):
        return f"{self.label}"
    
    def __hash__(self):
        return hash(self.rect)

    def __eq__(self, other):
        return self.rect == other.rect
    
    def bottom(self):
        x, y, w, h = self.rect
        return (x+w//2, y+h-1)
    
    def top(self):
        x, y, w, h = self.rect
        return (x+w//2, y+1)
