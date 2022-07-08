import cv2
from DraftMarkRecognition import DraftMark
from WaterLineDetection import WaterLineDetector
from Colors import Color
from Board import Board


def draw_into_img(org, rect, new):
    x, y, w, h = rect
    org[y:y+h, x:x+w] = new[y:y+h, x:x+w]


drawing = False # true if mouse is pressed
ix,iy = -1,-1
irect = None
iimg = None
# mouse callback function
def draw_rect(event,x,y,flags,param):
    global ix,iy,drawing,iimg,irect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(iimg,(ix,iy),(x,y),Color.GREEN.value,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(iimg,(ix,iy),(x,y),Color.GREEN.value,-1)
        irect = ix, iy, x-ix, y-iy


def get_rect(img):
    global iimg, irect
    iimg = img
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rect)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            r = irect
            irect = None
            cv2.destroyAllWindows()
            return r


def main(img_path):
    img = cv2.imread(img_path)
    board = Board(img, 2, 2)
    board.draw_img(img, 0, 0)
    x, y, w, h = get_rect(img.copy())
    det = WaterLineDetector()
    mark = DraftMark((x, y, w, h), img, True)

    X, Y = det.find(img_path, mark, True)
    draw_into_img(img, det.rect1, det.canny)
    draw_into_img(img, det.rect2, det.canny)
    cv2.line(img, (X, Y), (x + w//2, y+h), Color.YELLOW.value, 2, cv2.LINE_AA)
    cv2.line(img, det.line[0], det.line[1], Color.BLUE.value, 1, cv2.LINE_AA)
    cv2.rectangle(img,(x,y),(x+w, y+h),Color.GREEN.value,1)

    board.draw_img(det.canny, 0, 1)
    board.draw_img(det.open_canny, 1, 0)
    board.draw_img(img, 1, 1)
    board.show()


if __name__ == "__main__":
    main("..\\data\\images\\image_01.png")
