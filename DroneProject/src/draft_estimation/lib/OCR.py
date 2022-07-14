import cv2
import pytesseract
import numpy as np

from src.draft_estimation.lib.DraftMarks import DraftMarkString
from pickle import loads, dumps


def eval_char_tess(char_img, alphabet):
    h, w = char_img.shape
    w_offset, h_offset = 1, 1
    img = 255 * np.ones((h+2*h_offset, w+2*w_offset), dtype=np.uint8)
    img[h_offset:h+h_offset, w_offset:w+w_offset] = cv2.bitwise_not(char_img)

    custom_config = f"-c tessedit_char_whitelist={alphabet} --oem 3 --psm 10"
    pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    return max((label, float(conf)) for conf, label in zip(data["conf"], data["text"]))


def eval_char_human(char_img, alphabet):
    cv2.imshow("What char is in the picture?", cv2.resize(char_img, (0, 0), fx=25, fy=25, interpolation=cv2.INTER_AREA))
    value = chr(cv2.waitKey(0))
    cv2.destroyAllWindows()
    if value in alphabet:
        return value, 1
    return "", -1


def normalize_img_0_1(img):
    return (img - np.min(img)) / np.ptp(img)


def normalize_img_0_255(img):
    return (255*(normalize_img_0_1(img))).astype(np.uint8)   


class TemplateOCR:
    def __init__(self, filename):
        # TODO: fine-tune
        self.filename = filename
        self.adaptivity = 100
        self.templates = None
        self.load()

    def load(self):
        with open(self.filename, 'rb') as file:
            self.templates = loads(file.read())
        return self

    def save(self):
        with open(self.filename, 'wb') as file:
            file.write(dumps(self.templates))
        return self

    def eval_char(self, char_img, alphabet):
        res_label, res_conf = "", -10
        for c in alphabet:
            template = self.templates[c]
            h, w = template.shape
            char_img = cv2.resize(char_img, (w, h), interpolation=cv2.INTER_AREA)
            conf = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)[0][0]
            if not res_conf or conf > res_conf:
                res_label, res_conf = c, conf
        return res_label, res_conf

    def train_help(self, mark):
        template = self.templates[mark.label]
        h, w = template.shape[:2]
        char_img = cv2.resize(mark.materialize(), (w, h), interpolation=cv2.INTER_AREA)
        self.templates[mark.label] = normalize_img_0_255(normalize_img_0_1(template) + (normalize_img_0_1(char_img) / self.adaptivity))

    def train(self, marks):
        for mark in marks:
            if isinstance(mark, DraftMarkString):
                for m in mark.marks:
                    self.train_help(m)
            else:
                self.train_help(mark)
        return self
