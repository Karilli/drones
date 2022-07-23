#############################################################
if __name__ == "__main__":                                 ##
    import os                                              ##
    import sys                                             ##
    sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#############################################################


import numpy as np
import cv2

from os.path import exists
from pickle import loads, dumps
from enum import Enum
from collections import defaultdict
from time import perf_counter

from src.draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator
from src.draft_estimation.demo.DraftMarkSegmentationDemo import choose_kernel_radius
from src.draft_estimation.lib.ImageUtils import resize_to_full_screen
from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.lib.Grid2DMath import overlap_area_percentage, join


PATH = str
ID = int  # 0 for image, 1..n for sequence
IMG_ID = tuple[PATH, ID]

RECT = tuple[int, int, int, int]
LABEL = str
TOPHAT_FLAG = bool
MARK = tuple[RECT, LABEL, TOPHAT_FLAG]
STRING = tuple[MARK, MARK]
P = int


ENTER = 13
ESC = 27
SPACE = 32
ERROR = -1


class DataBase:
    def __init__(self, filename):
        self.filename = filename
        self.db: dict[IMG_ID, tuple[set[MARK], set[STRING], P]] = defaultdict(lambda: (set(), set()))
        self.load()

    def load(self):
        if not exists(self.filename):
            return self

        with open(self.filename, 'rb') as file:
            self.db = defaultdict(lambda: (set(), set()), loads(file.read()))
        return self

    def load_imgs(self):
        for i in range(1, 10):
            img_path = "..\\DroneProject\\data\\images\\" + str(i).rjust(2, "0") + ".png"
            self.db[(img_path, 0)]
        return self

    def store(self):
        with open(self.filename, 'wb') as file:
            file.write(dumps(dict(self.db)))
        return self

    @staticmethod
    def get_label(img, rect, curr_label):
        img = img.copy()
        x, y, w, h, = rect
        win_name = f"Label the mark, current label: {curr_label}"
        cv2.rectangle(img, (x, y), (x+w, y+h), Color.GREEN.value, 1)
        cv2.imshow(win_name, resize_to_full_screen(img))
        label = cv2.waitKey(0)
        cv2.destroyWindow(win_name)
        if label == SPACE or label == ENTER or label == ERROR:
            return curr_label
        if curr_label and curr_label in "0123456789M" and chr(label).upper() not in "0123456789M":
            return curr_label
        return chr(label).upper()

    @staticmethod
    def show_img(img, marks, strings, win_name):
        img = img.copy()
        for mark1, mark2 in strings:
            x, y, w, h = join(mark1[0], mark2[0])
            offset = 2
            cv2.rectangle(img, (x-offset, y-offset), (x+w+offset, y+h+offset), Color.BLUE.value, 1)
        for (x, y, w, h), label, tophat_flag in marks:
            col = Color.GREEN.value if tophat_flag else Color.RED.value
            cv2.rectangle(img, (x, y), (x+w, y+h), col, 1)
            if label is not None:
                cv2.putText(img, str(label), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        cv2.imshow(win_name, img)

    # TODO: make window with image paths
    def show(self):
        for (img_path, id), (marks, strings, p) in self.db.items():
            img = cv2.imread(img_path)
            if self.update_img(img_path) == ESC:
                return
        self.store()
        return self

    def update_seq(self, seq_path):
        pass # TODO: not implemented

    # TODO: resize to full screen
    def update_img(self, img_path):
        marks, strings = self.db[(img_path, 0)]
        button_down_delta, x1, y1 = 0, None, None
        org = cv2.imread(img_path)

        def marks_at_pt(x, y):
            return marks_in_area((x, y, 0, 0))

        def marks_in_area(area):
            nonlocal marks
            return list(filter(lambda mark: overlap_area_percentage(area, mark[0]) >= 0.95, marks))

        def label_marks(marks_to_label):
            nonlocal marks, strings, org
            for mark in marks_to_label:
                label = self.get_label(org, mark[0], str(mark[1]))
                new_mark = (mark[0], label, mark[2])
                marks.remove(mark)
                marks.add(new_mark)
                for string in strings:
                    if mark == string[0]:
                        strings.remove(string)
                        strings.add((new_mark, string[1]))
                    elif mark == string[1]:
                        strings.remove(string)
                        strings.add((string[0], new_mark))

        def draw_rect_at(x, y, tophat_flag):
            nonlocal dm_seg, marks
            mark = dm_seg.marks.mark_at(x, y, tophat_flag)
            if mark:
                add_mark((mark.rect, None, tophat_flag))

        def remove_marks_in_area(marks_in_area):
            for mark in marks_in_area:
                remove_mark(mark)

        def remove_mark(mark):
            nonlocal marks, strings
            marks.remove(mark)
            for string in strings.copy():
                if mark in string:
                    strings.remove(string)

        def add_mark(mark):
            nonlocal marks
            marks.add(mark)

        def add_string(string):
            nonlocal strings
            strings.add(string)

        def mouse_callback_fnc(event, x, y, flags, param):
            nonlocal x1, y1, button_down_delta
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                button_down_delta = perf_counter()
                x1, y1 = x, y
            elif button_down_delta and (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP):
                tophat_flag = event == cv2.EVENT_LBUTTONUP
                # only clicked
                if perf_counter() - button_down_delta < 0.10 or abs((x-x1)*(y-y1)) < 20:
                    marks_to_label = marks_at_pt(x, y)
                    if len(marks_to_label) != 0:
                        label_marks(marks_to_label)
                    else:
                        draw_rect_at(x, y, tophat_flag)
                # mouse moved
                else:
                    rect = (x1, y1, x-x1, y-y1)
                    marks_in = marks_in_area(rect)
                    if flags & cv2.EVENT_FLAG_SHIFTKEY:
                        remove_marks_in_area(marks_in)
                    elif len(marks_in) == 2:
                        add_string((marks_in[0], marks_in[1]))
                    else:
                        add_mark((rect, None, tophat_flag))
                button_down_delta = 0

        kernel_radius = 15
        dm_seg = DraftMarkSegmentator(kernel_radius).search_marks(org)
        win_name = "Select marks."
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_callback_fnc)

        while True:
            self.show_img(org, marks, strings, win_name)
            code = cv2.waitKey(1)
            if code == ESC:
                return ESC
            elif code == ENTER:
                self.db[(img_path, 0)] = (marks, strings)
                self.store()
                return ENTER
            elif code != ERROR and chr(code).lower() == "p":
                kernel_radius = choose_kernel_radius(org)
                dm_seg = DraftMarkSegmentator(kernel_radius).search_marks(org)
                cv2.destroyAllWindows()
                cv2.namedWindow(win_name)
                cv2.setMouseCallback(win_name, mouse_callback_fnc)


def main(filename):
    db = DataBase(filename).load_imgs()
    db.show()


if __name__ == "__main__":
    main("..\\DroneProject\\data\\db\\new_db")
