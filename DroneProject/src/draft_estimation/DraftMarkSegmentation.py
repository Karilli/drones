import cv2

from src.draft_estimation.lib.Colors import Color
from src.draft_estimation.lib.DraftMarks import DraftMark
from src.draft_estimation.lib.MarkSet import MarkSet


class DraftMarkSegmentator:  # :-D
    def __init__(self, kernel_radius, demo=False):
        self.marks = MarkSet()
        self.kernel_radius = kernel_radius
        self.w_to_h_ratio_range = (0.1, 2)
        self.area_range = (50, 2000)
        self.demo = demo

    def search_marks(self, img):
        # TODO: crop the img
        # TODO: fine-tune red masking
        mask = cv2.inRange(img, (0, 0, 50), (15, 15, 255))
        img[mask>0] = Color.PINK.value
        color_corrected = img
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_radius, self.kernel_radius))
        tophat_img = cv2.morphologyEx(grey_img, cv2.MORPH_TOPHAT, kernel)
        blackhat_img = cv2.morphologyEx(grey_img, cv2.MORPH_BLACKHAT, kernel)

        _, tophat_bin_img = cv2.threshold(tophat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, blackhat_bin_img = cv2.threshold(blackhat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tophat_cntrs, _ = cv2.findContours(tophat_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blackhat_cntrs, _ = cv2.findContours(blackhat_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.marks |= {DraftMark(cv2.boundingRect(cntr), tophat_bin_img, True) for cntr in tophat_cntrs}
        self.marks |= {DraftMark(cv2.boundingRect(cntr), blackhat_bin_img, False) for cntr in blackhat_cntrs}

        if self.demo:
            self.color_corrected = color_corrected
            self.tophat_img = tophat_img
            self.blackhat_img = blackhat_img
            self.tophat_bin_img = tophat_bin_img
            self.blackhat_bin_img = blackhat_bin_img

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
        return self.marks
