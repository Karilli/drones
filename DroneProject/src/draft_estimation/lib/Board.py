import numpy as np
import cv2

from src.draft_estimation.lib.Colors import Color


def resize_to_full_screen(img):
    max_h, max_w = 600, 1400
    h, w = img.shape[:2]
    factor = min(max_h / h, max_w / w)
    return cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


class Board:
    def __init__(self, img, rows, cols):
        self.rows = rows
        self.cols = cols
        self.h, self.w, _ = img.shape
        self.board = 255 * np.zeros((rows*(self.h+1)-1, cols*(self.w+1)-1, 3), dtype=np.uint8)

        for y in range(1, rows+1):
            y = y*(self.h+1)-1
            cv2.line(self.board, (0, y), (cols*(self.w+1)-1, y), Color.WHITE.value, thickness=1)
        for x in range(1, cols+1):
            x = x*(self.w+1)-1
            cv2.line(self.board, (x, 0), (x, rows*(self.h+1)-1), Color.WHITE.value, thickness=1)

    def move_point(self, pt, row, col):
        x, y = pt
        return col * (self.w + 1) + x, row * (self.h + 1) + y

    def draw_marks(self, marks, row, col):
        if not marks:
            return

        img = np.zeros((self.h, self.w), np.uint8)
        for mark in marks:
            x, y, w, h = mark.rect
            img[y:y+h, x:x+w] = mark.materialize()

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for mark in marks:
            mark.draw(img)
        return self.draw_img(img, row, col)

    def draw_line(self, pt1, pt2, color, thickness, row, col):
        cv2.line(self.board, self.move_point(pt1, row, col), self.move_point(pt2, row, col), color, thickness=1)
    
    def draw_img(self, img, row, col):
        if not (0 <= row <= self.rows) or not (0 <= col <= self.cols):
            raise ValueError

        img = self.check_dims(img)
        x, y = self.move_point((0, 0), row, col)
        self.board[y:y+self.h, x:x+self.w] = img
        return img

    def check_dims(self, img):
        if len(img.shape) == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def show(self):
        cv2.imshow("", resize_to_full_screen(self.board))
        cv2.waitKey(0)
        return self
