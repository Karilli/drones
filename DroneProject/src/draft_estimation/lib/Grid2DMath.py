from math import sqrt


def distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def top_left_pt(rect):
    x, y, w, h = rect
    if w < 0 and h < 0:
        return x+w, y+h
    elif w < 0:
        return x+w, y
    elif h < 0:
        return x, y+h
    return x, y


def bottom_right_pt(rect):
    x, y, w, h = rect
    if w < 0 and h < 0:
        return x, y
    elif w < 0:
        return x, y+h
    elif h < 0:
        return x+w, y
    return x+w, y+h


def join(rect1, rect2):
    (x1, y1), (x2, y2) = top_left_pt(rect1), bottom_right_pt(rect1)
    (x3, y3), (x4, y4) = top_left_pt(rect2), bottom_right_pt(rect2)
    x = min(x1, x3)
    y = min(y1, y3)
    w = max(x2, x4) - x
    h = max(y2, y4) - y
    return x, y, w, h


def are_overlapping(rect1, rect2):
    (x1, y1), (x2, y2) = top_left_pt(rect1), bottom_right_pt(rect1)
    (x3, y3), (x4, y4) = top_left_pt(rect2), bottom_right_pt(rect2)
    return (x1 < x4) and (x2 > x3) and (y2 > y3) and (y1 < y4)


def overlap_area(rect1, rect2):
    (x1, y1), (x2, y2) = top_left_pt(rect1), bottom_right_pt(rect1)
    (x3, y3), (x4, y4) = top_left_pt(rect2), bottom_right_pt(rect2)
    x_dist = (min(x2, x4) - max(x1, x3))
    y_dist = (min(y2, y4) - max(y1, y3))
    area = 0
    if x_dist > 0 and y_dist > 0:
        area = x_dist * y_dist
    return max(area, 0)


def overlap_area_percentage(rect1, rect2):
    _, _, w1, h1 = rect1
    _, _, w2, h2 = rect2
    if min(abs(w1*h1), abs(w2*h2)) == 0:
        return are_overlapping(rect1, rect2)
    return overlap_area(rect1, rect2) / min(abs(w1*h1), abs(w2*h2))
