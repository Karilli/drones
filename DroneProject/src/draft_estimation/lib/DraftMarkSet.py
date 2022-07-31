from src.draft_estimation.lib.DraftMarks import DraftMarkString
from src.draft_estimation.lib.Grid2DMath import overlap_area_percentage, are_overlapping


class DraftMarkSet():
    def __init__(self):
        self.marks = set()
        self.strings = set()
        self.removed_marks = set()
        self.removed_strings = set()

    def marks_and_strings(self, removed=False):
        used_in_strings = set()
        strings = self.removed_strings if removed else self.strings
        marks = self.removed_marks if removed else self.marks
        for string in strings:
            for mark in string.marks:
                used_in_strings.add(mark)
        return strings | marks - used_in_strings

    def __remove_mark(self, mark):
        if mark in self.marks:
            self.marks.remove(mark)
        self.removed_marks.add(mark)

    def __remove_string(self, string):
        if string in self.strings:
            self.strings.remove(string)
        self.removed_strings.add(string)
        for mark in string.marks:
            self.__remove_mark(mark)            

    def remove(self, key):
        if isinstance(key, DraftMarkString):
            self.__remove_string(key)
        elif key.string:
            self.__remove_string(key.string)
        else:
            self.__remove_mark(key)

    def __add_string(self, string):
        if string in self.removed_strings:
            self.removed_strings.remove(string)
        self.strings.add(string)
        for mark in string.marks:
            self.__add_mark(mark) 

    def __add_mark(self, mark):
        if mark in self.removed_marks:
            self.removed_marks.remove(mark)
        self.marks.add(mark)

    def add(self, key):
        if isinstance(key, DraftMarkString):
            self.__add_string(key)
        elif key.string:
            self.__add_string(key.string)
        else:
            self.__add_mark(key)
    
    def add_from(self, iterator):
        for key in iterator:
            self.add(key)

    def y_sorted(self, reverse, strings):
        iterator = self.marks if not strings else self.marks_and_strings()
        return list(sorted(iterator, key=lambda x: x.rect[1], reverse=reverse))

    def pairs(self, strings):
        marks = list(self.marks if not strings else self.marks_and_strings())
        for i in range(len(marks)):
            for j in range(i+1, len(marks)):
                yield marks[i], marks[j]

    def in_area(self, area, min_overlap_percentage, strings, removed=False):
        marks = self.removed_marks if removed else self.marks
        iterator = marks if not strings else self.marks_and_strings(removed) 
        return list(filter(lambda mark: overlap_area_percentage(mark.rect, area) >= min_overlap_percentage, iterator))

    def mark_at(self, x, y, tophat_flag):
        for mark in self.marks:
            if mark.tophat_flag == tophat_flag and are_overlapping(mark.rect, (x, y, 0, 0)):
                return mark

    def __contains__(self, key):
        return key in self.marks or key in self.strings

    def __iter__(self):
        return iter(self.marks_and_strings())
