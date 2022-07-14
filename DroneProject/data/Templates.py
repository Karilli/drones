#########################################################
import os                                              ##
import sys                                             ##
sys.path.insert(1, os.path.abspath("..\\DroneProject"))##
#########################################################


import cv2
import numpy as np
from pickle import loads, dumps
from data.DBManager import DataBase, DBType


def resize_to_full_screen(img):
    max_h, max_w = 200, 400
    h, w = img.shape[:2]
    factor = min(max_h / h, max_w / w)
    return cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def make_templates(template_path):
    res = {}
    alphabet = "0123456789M"
    db = DataBase(DBType.MARK)
    for char in alphabet:
        h = np.median(list(map(lambda x: x[1].shape[0], filter(lambda x: char == x[0].upper(), db.db.values()))))
        w = np.median(list(map(lambda x: x[1].shape[1], filter(lambda x: char == x[0].upper(), db.db.values()))))
        imgs = [img for label, img in db.db.values() if char == label.upper()]
        if not imgs:
            h = np.median(list(map(lambda x: x[1].shape[0], filter(lambda x: x[0].upper() in alphabet, db.db.values()))))
            w = np.median(list(map(lambda x: x[1].shape[1], filter(lambda x: x[0].upper() in alphabet, db.db.values()))))
            template = np.zeros((int(h), int(w)), dtype=np.uint8)
        else:
            template = sum(img for img in map(lambda x: cv2.resize(x, (int(w), int(h)), cv2.INTER_AREA) == 255, imgs)) / len(imgs)
            template = np.array([np.array([int(255 * x) for x in l], dtype=np.uint8) for l in template], dtype=np.uint8)
        res[char] = template

        cv2.imshow(char, resize_to_full_screen(template))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with open(template_path, 'wb') as file:
        file.write(dumps(res))


def edit(img):
    color = 255
    radius = 5
    drawing = False
    h, w = img.shape[:2]
    img = resize_to_full_screen(img)
    new_h, new_w = img.shape[:2]

    def draw_circle(event, x, y, flags, param):
        nonlocal color, radius, img, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(img, (x, y), radius, color, -1)
        elif drawing and event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(img, (x, y), radius, color, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img, (x, y), radius, color, -1)

    cv2.namedWindow("Editor.")
    cv2.setMouseCallback("Editor.", draw_circle)
    while True:
        cv2.imshow("Editor.", img)
        code = cv2.waitKey(1)
        if code == -1:
            continue
        elif chr(code) == "c":
            color = 0 if color == 255 else 255
        elif chr(code) == "w":
            radius = min(new_w // 10, new_h // 10, radius + 1)
        elif chr(code) == "s":
            radius = max(1, radius - 1)
        elif code != -1:
            cv2.destroyAllWindows()
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def edit_templates(template_path):
    with open(template_path, 'rb') as file:
        templates = loads(file.read())
    
    for char, img in templates.items():
        templates[char] = edit(img)
    
    with open(template_path, 'wb') as file:
        file.write(dumps(templates))


def show_templates(template_path):
    with open(template_path, 'rb') as file:
        templates = loads(file.read())

    for char, img in templates.items():
        print(img.shape)
        cv2.imshow(char, resize_to_full_screen(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(template_path):
    #make_templates(template_path)
    #edit_templates(template_path)
    show_templates(template_path)


if __name__ == '__main__':
    main("..\\DroneProject\\data\\mark_templates")
