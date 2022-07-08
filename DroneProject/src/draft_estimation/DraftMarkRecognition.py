import cv2
from OCR import eval_mark
from Colors import Color


HEIGHT_OF_MARK_CM = 10


class MySet(set):
    def __init__(self, *args, **kwargs):
        self.removed = set()
        super().__init__(*args, **kwargs)
    
    def remove(self, key):
        self.removed.add(key)
        if super().__contains__(key):
            super().remove(key)
        return self
    
    def get_removed(self):
        return self.removed
    
    def add(self, key):
        super().add(key)
        if key in self.removed:
            self.removed.remove(key)


class DraftMarkRecognizer:
    def __init__(self):
        self.marks = MySet()

    def search_marks(self, img_path, p):
        org_img = cv2.imread(img_path)
        grey_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

        img_h = grey_img.shape[0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*img_h//p, 2*img_h//p))
        tophat_img = cv2.morphologyEx(grey_img, cv2.MORPH_TOPHAT, kernel)
        blackhat_img = cv2.morphologyEx(grey_img, cv2.MORPH_BLACKHAT, kernel)

        _, tophat_bin_img = cv2.threshold(tophat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, blackhat_bin_img = cv2.threshold(blackhat_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tophat_cntrs, _ = cv2.findContours(tophat_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blackhat_cntrs, _ = cv2.findContours(blackhat_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.marks |= {DraftMark(cv2.boundingRect(cntr), tophat_bin_img, True) for cntr in tophat_cntrs}
        self.marks |= {DraftMark(cv2.boundingRect(cntr), blackhat_bin_img, False) for cntr in blackhat_cntrs}

        return self

    def w_to_h_ratio_filter(self, mn, mx):
        for mark in self.marks.copy():
            _, _, w, h = mark.rect
            if not (mn < w/h < mx):
                self.marks.remove(mark)
        return self
    
    def area_filter(self, mn, mx):
        for mark in self.marks.copy():
            _, _, w, h = mark.rect
            if not (mn < w*h < mx):
                self.marks.remove(mark)
        return self
    
    def join_mark_strings(self, vert_dist, horz_dist, h_ratio):
        sorted_marks = sorted(self.marks, key=lambda mark: (mark.rect[1], mark.rect[0]))
        for m1, m2 in zip(sorted_marks[:-1], sorted_marks[1:]):
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            cx1, cy1 = m1.center()
            cx2, cy2 = m2.center()
            mean_h = (h1 + h2) / 2
            if abs(cy1 - cy2) < mean_h * vert_dist and abs(cx1 - cx2) < mean_h * horz_dist and h_ratio <= h1/h2 <= 1 / h_ratio:
                m1.join_with(m2)
        return self

    def eval_marks(self):
        for mark in self.marks:
            mark.eval()
        return self
    
    def conf_filter(self, mn):
        for mark in self.marks.copy():
            if mark.conf < mn:
                self.marks.remove(mark)
        return self

    def __get_y_sorted(self):
        sorted_marks = sorted(self.marks, key=lambda x: x.rect[1])
        for i in range(len(sorted_marks)):
            for j in range(i+1, len(sorted_marks)):
                yield sorted_marks[i], sorted_marks[j]

    def join_tophat_and_blackhat(self):
        for m1, m2 in self.__get_y_sorted():
            if m1.tophat_flag == m2.tophat_flag or not are_overlapping(m1.rect, m2.rect):
                continue
            self.marks.remove(min(m1, m2, key=lambda x: x.conf))
        return self

    def resolve_y_overlaps(self):
        for m1, m2 in self.__get_y_sorted():
            x1, y1, w1, h1 = m1.rect
            x2, y2, w2, h2 = m2.rect
            if not are_overlapping((0, y1, 3, h1), (1, y2, 3, h2)):
                continue
            if m1.mark_string and m2.mark_string:
                if m1.mark_string is m2.mark_string:
                    continue
                self.marks -= {min(m1.mark_string, m2.mark_string, key=lambda x: x.conf()).marks}
            if m1.mark_string:
                if m1.mark_string.conf() >= m2.conf:
                    self.marks -= set(m1.mark_string.marks)
                else:
                    self.marks.remove(m2)
            elif m2.mark_string:
                if m2.mark_string.conf() >= m1.conf:
                    self.marks -= set(m2.mark_string.marks)
                else:
                    self.marks.remove(m1)
            else:
                self.marks.remove(min(m1, m2, key=lambda x: x.conf))
        return self
    
    def resurrect_missing_marks(self):
        pass # TODO

    def run(self, img_path):
        num_of_marks = 2
        img_h_cm = HEIGHT_OF_MARK_CM * (2 * num_of_marks - 1)
        self.search_marks(img_path, img_h_cm).area_filter(50, 2000).w_to_h_ratio_filter(0.1, 2)
        self.eval_marks().conf_filter(0.2)
        self.join_tophat_and_blackhat().join_mark_strings(0.25, 2, 0.9)
        self.resolve_y_overlaps().resurrect_missing_marks()
    
    def get_y_sorted(self):
        return sorted(self.marks, key=lambda x: x.rect[1], reverse=True)


def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        DraftMarkRecognizer().run("..\\data\\images\\image_01.png")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


if __name__ == '__main__':
    main()
