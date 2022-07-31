# TODO: fine-tune

TEMPLATES_PATH = "..\\DroneProject\\data\\templates\\adaptive"
TEMPLATE_ADAPTIVITY = 100
OPEN_KERNEL = (10, 1)
DEFAULT_P = 15

# recognition requiremnets
MARK_MAX_GAP = 2  # recognition can skip maximally MAX_GAP marks
MIN_MARKS = 1  # minimum required marks for succesfull recognition

# requirements for single mark
MARK_MIN_CONF = 0.5
MARK_MAX_AREA = 2000
MARK_MIN_AREA = 50
MARK_W_TO_H_RATIOS = [
    ("0", (0.5, 2)),
    ("1", (0.1, 1)),
    ("2", (0.5, 2)),
    ("3", (0.5, 2)), 
    ("4", (0.5, 2)),
    ("5", (0.5, 2)),
    ("6", (0.5, 2)),
    ("7", (0.1, 2)),
    ("8", (0.5, 2)),
    #("9", (0.5, 2)),
    ("M", (0.5, 2))
]
MARK_MIN_W_TO_H_RATIO = min(map(lambda x: x[1][0], MARK_W_TO_H_RATIOS))
MARK_MAX_W_TO_H_RATIO = max(map(lambda x: x[1][1], MARK_W_TO_H_RATIOS))

# positional requiremnets for overlapping marks
MARK_MAX_OVERLAP_PERCENTAGE = 0.2

# positional requiremnets for marks above each other
MARKS_MIN_H_RATIO = 0.7
MARKS_MAX_H_RATIO = 1 / MARKS_MIN_H_RATIO
MARKS_MIN_HORZ_DIST = 1.5  # times height of above mark
MARKS_MAX_HORZ_DIST = 2.5  # times height of above mark

# positional requiremnets for marks in string
STRING_MIN_H_RATIO = 0.85
STRING_MAX_H_RATIO = 1 / STRING_MIN_H_RATIO
STRING_MAX_VERT_DIST = 0.5  # times mean height
STRING_MAX_HORZ_DIST = 2  # times mean height
