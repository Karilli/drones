from itertools import combinations
from src.draft_estimation.lib.Grid2DMath import top_left_pt, bottom_right_pt, distance_y
from src.draft_estimation.Constants import MARK_MAX_GAP, MIN_MARKS, MARKS_MIN_HORZ_DIST, MARKS_MAX_HORZ_DIST, MARKS_MIN_H_RATIO, MARKS_MAX_H_RATIO


def mark_dist(mark1, mark2):
    a = int(mark1.label.replace("M", "0"))
    b = int(mark2.label.replace("M", "0"))
    if mark2.center()[1] < mark1.center()[1]:
        a, b = b, a

    if a >= 10 and b >= 10:
        return a - b
    if a >= 10:
        b += 10 * (a // 10 - 1)
        return a - b
    if b >= 10:
        a += 10 * (b // 10)
        return a - b
    return 10*(a<=b)+a-b


def check_strings(comb):
    strings = list(filter(lambda x: (x.label) == 2, comb))
    for mark1, mark2 in zip(strings[:-1], strings[1:]):
        diff = mark_dist(mark1, mark2)
        if diff != 10:
            return False
    return True


def check_comb(comb):
    if any(map(lambda x: len(x.label) == 1, comb)) and not check_strings(comb):
        return False

    for m1, m2 in zip(comb[:-1], comb[1:]):
        diff = mark_dist(m1, m2)
        gap = diff // 2 - 1

        if diff % 2 != 0 or diff < 0:
            return False
        if gap > MARK_MAX_GAP:
            return False
        if not (MARKS_MIN_HORZ_DIST*(gap+1)*m1.rect[3] <= distance_y(m1.center(), m2.center()) <= MARKS_MAX_HORZ_DIST*(gap+1)*m1.rect[3]):
            return False
        if not (MARKS_MIN_H_RATIO <= m1.rect[3] / m2.rect[3] <= MARKS_MAX_H_RATIO):
            return False
    return True


def match_sequence(seq):
    for k in range(len(seq), MIN_MARKS-1, -1):
        res_comb, res_conf = None, None
        for comb in combinations(seq, k):
            conf = sum(map(lambda x: x.conf, comb)) / k
            if check_comb(comb) and (res_comb is None or res_conf < conf):
                res_comb, res_conf = comb, conf
        if res_comb is not None:
            return res_comb
    return []


def match_sequences(marks):
    even = list(filter(lambda x: (len(x.label) == 2 and (x.label[1] == "M" or x.label[1] == "0")) or (len(x.label) == 1 and x.label != "M" and int(x.label) % 2 == 0), marks.y_sorted(reverse=False, strings=True)))
    even_strings = list(filter(lambda x: len(x.label) == 2 and x.label[0] != "M" and (x.label[1] == "M" or int(x.label) % 2 == 0), marks.y_sorted(reverse=False, strings=True)))
    odd_strings = list(filter(lambda x: len(x.label) == 2 and "M" not in x.label and int(x.label) % 2 == 1, marks.y_sorted(reverse=False, strings=True)))
    return max(map(match_sequence, [even, even_strings, odd_strings]), key=lambda x: len(x))
