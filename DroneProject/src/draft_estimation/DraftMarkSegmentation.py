import cv2

from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.lib.DraftMarks import DraftMark
from src.draft_estimation.lib.DraftMarkSet import DraftMarkSet


class DraftMarkSegmentator:  # :-D
    def __init__(self, kernel_radius):
        self.marks = DraftMarkSet()
        self.kernel_radius = kernel_radius
        self.w_to_h_ratio_range = (0.1, 2)
        self.area_range = (50, 2000)

        # for debbuging and demos
        self.color_corrected = None
        self.tophat_img = None
        self.blackhat_img = None
        self.tophat_bin_img = None
        self.blackhat_bin_img = None

    def search_marks(self, img):
        # TODO: crop the img
        # TODO: black on red marks problem
        # mask = cv2.inRange(img, (0, 0, 50), (15, 15, 255))
        # img_diff = img.max(axis=2) - img.min(axis=2)
        self.color_corrected = img.copy()
        # self.color_corrected[img_diff > 75] = Color.BLUE.value

        grey_img = cv2.cvtColor(self.color_corrected, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_radius, self.kernel_radius))
        self.tophat_img = cv2.morphologyEx(grey_img, cv2.MORPH_TOPHAT, kernel)
        self.blackhat_img = cv2.morphologyEx(grey_img, cv2.MORPH_BLACKHAT, kernel)

        _, self.tophat_bin_img = cv2.threshold(self.tophat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, self.blackhat_bin_img = cv2.threshold(self.blackhat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tophat_cntrs, _ = cv2.findContours(self.tophat_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blackhat_cntrs, _ = cv2.findContours(self.blackhat_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.marks.add_from({DraftMark(cv2.boundingRect(cntr), self.tophat_bin_img, True) for cntr in tophat_cntrs})
        self.marks.add_from({DraftMark(cv2.boundingRect(cntr), self.blackhat_bin_img, False) for cntr in blackhat_cntrs})

        return self

    def w_to_h_ratio_filter(self):
        # TODO: fine-tune params and make them adaptive
        mn, mx = self.w_to_h_ratio_range
        for mark in self.marks.copy():
            _, _, w, h = mark.rect
            if not (mn < w/h < mx):
                self.marks.remove(mark)
        return self

    def area_filter(self):
        # TODO: fine-tune params and make them adaptive
        mn, mx = self.area_range
        for mark in self.marks.copy():
            _, _, w, h = mark.rect
            if not (mn < w*h < mx):
                self.marks.remove(mark)
        return self

    def run(self, org_img):
        self.search_marks(org_img).w_to_h_ratio_filter().area_filter()
        if not self.marks:
            raise ValueError("Segmentation fault")
        return self.marks
