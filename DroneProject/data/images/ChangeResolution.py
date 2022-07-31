import cv2
import shutil
from enum import Enum


class Resolution(Enum):
    SD = (720, 576)
    HD = (1280, 720)
    FullHD = (1920, 1080)
    UHD = (3840, 2160)


def main(img_path, resolution):
    img = cv2.imread(img_path)
    new_img = cv2.resize(img, resolution.value, interpolation=cv2.INTER_LANCZOS4)
    x = img_path.rsplit('.', 1)
    new_name = x[0] + "_" + resolution.name + "." + x[1]
    x = shutil.copyfile(img_path, new_name)
    cv2.imwrite(new_name, new_img)
    # try:
    #     img = cv2.imread(img_path)
    #     new_img = cv2.resize(img, resolution.value, interpolation = cv2.INTER_LANCZOS4)
    #     x = img_path.rsplit('.', 1)[0]
    #     new_name = x[0] + "_" + resolution.name + x[1]
    #     cv2.imwrite(new_name, new_img)
    # except:
    #     print(f"'{img_path}' couldn't be resized.")


if __name__ == "__main__":
    main("..\\DroneProject\\data\\images\\01.png", Resolution.UHD)
