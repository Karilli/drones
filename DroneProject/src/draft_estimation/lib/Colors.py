from enum import Enum


class Color(Enum):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    LIGHT_BLUE = (255, 255, 0)
    GRAY = (127, 127, 127)
    WHITE = (255, 255, 255)
    PINK = (255, 0, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)


def main():
    import cv2
    import numpy as np

    for color in Color:
        w, h = 200, 200
        img = np.zeros((w, h, 3), dtype=np.uint8)
        img[:h, :w] = np.array(list(color.value))
        cv2.imshow(color.name, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
