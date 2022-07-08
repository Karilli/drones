import cv2
import pytesseract
import numpy as np

def eval_mark(mark):
    h, w = mark.shape
    w_offset, h_offset = 1, 1
    img = 255 * np.ones((h+2*h_offset, w+2*w_offset), dtype=np.uint8)
    img[h_offset:h+h_offset, w_offset:w+w_offset] = cv2.bitwise_not(mark)

    custom_config = r"-c tessedit_char_whitelist=0123456789M --oem 3 --psm 10"
    pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    return max((label, float(conf)) for conf, label in zip(data["conf"], data["text"]))
