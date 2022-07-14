################################################
import os                                     ##
import sys                                    ##
sys.path.insert(1, os.path.abspath("..\\DroneProject\\src"))##
################################################


import numpy as np
import cv2

from os.path import exists
from pickle import loads, dumps
from enum import Enum
from draft_estimation.DraftMarkSegmentation import DraftMarkSegmentator
from draft_estimation.demo.DraftMarkSegmentationDemo import choose_kernel_radius


class DBType(str, Enum):
    MARK = "..\\DroneProject\\data\\draft_marks"
    IMAGE = "..\\DroneProject\\data\\draft_images_db"


IMAGE = np.ndarray
IMAGE_LABEL = str
IMAGE_HASH = bytes


def resize_to_full_screen(img):
    max_h, max_w = 600, 1400
    h, w = img.shape[:2]
    factor = min(max_h / h, max_w / w)
    return cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


class DataBase:
    def __init__(self, db_type: DBType):
        self.path = db_type
        self.db: dict[IMAGE_HASH, tuple[IMAGE_LABEL, IMAGE]] = {}
        self.load()
    
    def load(self):
        if not exists(self.path):
            return self
        
        with open(self.path, 'rb') as file:
            self.db = loads(file.read())
        return self
    
    def store(self):
        with open(self.path, 'wb') as file:
            file.write(dumps(self.db))
        return self
    
    def __getitem__(self, image):
        return self.db[image.tobytes()][0]
    
    def __setitem__(self, image, label):
        self.db[image.tobytes()] = label, image
    
    def get_label_and_set(self, image):
        self[image] = self.get_label_from_user(image)

    @staticmethod
    def get_label_from_user(image, curr_label=None):
        cv2.imshow(curr_label, resize_to_full_screen(image))
        label = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if label == 32 or label == 13 or label == -1:
            return curr_label
        if curr_label and curr_label in "0123456789M" and chr(label) not in "0123456789M":
            return curr_label
        return chr(label)

    def show(self, filter_fnc):
        for label, image in self.db.values():
            if filter_fnc(label):
                self[image] = self.get_label_from_user(image, label)
        self.store()
    
    def update_from_img(self, img_path):
        org = cv2.imread(img_path)

        kernel_radius = choose_kernel_radius(org)
        dm_seg = DraftMarkSegmentator(kernel_radius).run(org)

        for mark in dm_seg.marks:
            img = org.copy()
            mark.draw(img)
            cv2.imshow("mark", resize_to_full_screen(img))
            self.get_label_and_set(mark.materialize_mark())
        self.store()


# def convert_images_to_png():
#     from PIL import Image
#     for i in range(0, 9):
#         img_path = f"..\\data\\images\\image_0{i}"
#         im = Image.open(img_path + ".jfif" )
#         im.save(img_path + ".png")


def main():
    db = DataBase(DBType.MARK)
    #db.show(lambda x: x in "0123456789M")
    db.update_from_img(f"..\\DroneProject\\data\\images\\image_01.png")


if __name__ == "__main__":
    main()
