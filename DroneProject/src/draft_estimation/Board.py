import numpy as np
import cv2
from Colors import Color


class Board:
    def __init__(self, img, rows, cols):
        self.rows = rows
        self.cols = cols
        h, w, _ = img.shape
        self.board = 255 * np.zeros((rows*h+rows-1, cols*w+cols-1, 3), dtype=np.uint8)
        for y in range(1, cols+1):
            y = y*h+y-1
            cv2.line(self.board, (0, y), (cols*w+cols-1, y), Color.WHITE.value, thickness=1)
        for x in range(1, rows+1):
            x = x*w+x-1
            cv2.line(self.board, (x, 0), (x, rows*h+rows-1), Color.WHITE.value, thickness=1)

    def draw_marks(self, marks, x1, y1):
        if not marks:
            return

        w, h = next(iter(marks)).img.shape
        img = np.zeros((w, h), np.uint8)
        for mark in marks:
            x, y, w, h = mark.rect
            img[y:y+h, x:x+w] = mark.materialize_mark()

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for mark in marks:
            mark.draw(img)
        self.draw_img(img, x1, y1)

    def draw_img(self, img, x1, y1):
        if not (0 <= x1 <= self.rows) or not (0 <= y1 <= self.cols):
            raise ValueError

        h, w, _ = img.shape
        x, y = y1*w+y1, x1*h+x1
        self.board[y:y+h, x:x+w] = img
    

    def show(self):
        cv2.imshow("", self.board)
        cv2.waitKey(0)
