from DraftCalculator import DraftCalculator
import cv2


def main(img_path):
    img = cv2.imread(img_path)
    print(DraftCalculator().get_draft([img]))


if __name__ == '__main__':
    main("..\\data\\images\\image_01.png")
