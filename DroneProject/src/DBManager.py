import numpy as np
import cv2

from os.path import exists
from pickle import loads, dumps
from enum import Enum
from draft_estimation.lib.DraftMarkRecognition import DraftMarkRecognizer


class DBType(str, Enum):
    MARK = "..\\data\\draft_marks"


IMAGE = np.ndarray
IMAGE_LABEL = str
IMAGE_HASH = bytes


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
    
    def get_label_and_set(self, image, resize_flag=True):
        self[image] = self.get_label_from_user(image, resize_flag=resize_flag)

    @staticmethod
    def get_label_from_user(image, curr_label=None, resize_flag=True):
        if resize_flag:
            h, w = image.shape
            factor = 10000 / (h*w) / 2
            image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cv2.imshow(curr_label, image)
        label = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if label == 32 or label == -1:
            return curr_label
        return chr(label)

    def show(self, filter_fnc):
        for label, image in self.db.values():
            if filter_fnc(label):
                self[image] = self.get_label_from_user(image, label)
        self.store()
    
    def update_from_img(self, img_path, resize_flag=True):
        org = cv2.imread(img_path)
        cv2.imshow("How many marks?", org)
        num_of_marks = int(chr(cv2.waitKey(0)))
        cv2.destroyAllWindows()

        rec = DraftMarkRecognizer().search_marks(img_path, 20 * num_of_marks - 10).w_to_h_ratio_filter(0.5, 2).area_filter(50, 2000)
        for mark in rec.marks:
            mark.draw(org)

        cv2.imshow("Found marks.", org)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for mark in rec.marks:
            mark.materialize_mark()
            self.get_label_and_set(mark.materialized_mark, resize_flag)
        self.store()


def convert_images_to_png():
    from PIL import Image
    for i in range(0, 9):
        img_path = f"..\\data\\images\\image_0{i}"
        im = Image.open(img_path + ".jfif" )
        im.save(img_path + ".png")


def make_templates():
    res = {}
    a = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "M", "m"])
    db = DataBase(DBType.MARK)
    h = np.median(list(map(lambda x: x[1].shape[0], filter(lambda x: x[0] in a, db.db.values()))))
    w = np.median(list(map(lambda x: x[1].shape[1], filter(lambda x: x[0] in a, db.db.values()))))
    for label1 in a:
        imgs = []
        for label2, img in db.db.values():
            if label1.lower() == label2:
                imgs.append(img)
        if not imgs:
            continue
        template = sum(img for img in map(lambda x: cv2.resize(x, (int(w), int(h)), cv2.INTER_AREA) == 255, imgs)) / len(imgs)
        template = np.array([np.array([255 * round(x) for x in l], dtype=np.uint8) for l in template], dtype=np.uint8)
        res[label1] = template
        cv2.imshow(label1, template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    with open("..\\data\\mark_templates", 'wb') as file:
            file.write(dumps(res))


def main():
    #db = DataBase(DBType.MARK)
    make_templates()
    #db.show(lambda x: True)
    #db.update_from_img(f"..\\data\\images\\image_01.png", True)


if __name__ == "__main__":
    main()
