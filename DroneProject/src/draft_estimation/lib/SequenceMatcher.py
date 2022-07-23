from itertools import combinations
from src.draft_estimation.lib.Grid2DMath import top_left_pt, bottom_right_pt


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


def check_comb(comb, max_gap):
    if any(map(lambda x: len(x.label) == 1, comb)) and not check_strings(comb):
        return False

    for mark1, mark2 in zip(comb[:-1], comb[1:]):
        diff = mark_dist(mark1, mark2)
        gap = diff // 2 - 1
        cx1, cy1 = mark1.center()
        cx2, cy2 = mark2.center()

        if diff % 2 != 0 or diff < 0:
            return False
        if gap > max_gap:
            return False
        if not (1.5*(gap+1)*mark1.rect[3] <= abs(cy1 - cy2) <= 2.5*(gap+1)*mark1.rect[3]):
            return False
        if min(mark1.rect[3], mark2.rect[3]) / max(mark1.rect[3], mark2.rect[3]) < 0.7:
            return False
    return True


def match_sequence(seq, max_gap):
    for k in range(len(seq), 1, -1):
        for comb in combinations(seq, k):
            if check_comb(comb, max_gap):
                return comb
    return []


def match_sequences(marks, max_gap=2):
    even = list(filter(lambda x: (len(x.label) == 2 and (x.label[1] == "M" or x.label[1] == "0")) or (len(x.label) == 1 and x.label != "M" and int(x.label) % 2 == 0), marks.y_sorted(reverse=False, strings=True)))
    even_strings = list(filter(lambda x: len(x.label) == 2 and x.label[0] != "M" and (x.label[1] == "M" or int(x.label) % 2 == 0), marks.y_sorted(reverse=False, strings=True)))
    odd_strings = list(filter(lambda x: len(x.label) == 2 and "M" not in x.label and int(x.label) % 2 == 1, marks.y_sorted(reverse=False, strings=True)))
    return max(map(lambda x: match_sequence(x, max_gap), [even, even_strings, odd_strings]), key=lambda x: len(x))
