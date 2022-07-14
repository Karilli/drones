from src.draft_estimation.lib.DraftMarks import DraftMarkString


class MarkSet(set):
    def __init__(self, *args, **kwargs):
        self.removed = set()
        super().__init__(*args, **kwargs)
    
    def remove_one(self, key):
        self.removed.add(key)
        if super().__contains__(key):
            super().remove(key)
        return self

    def remove(self, key):
        if isinstance(key, DraftMarkString):
            for mark in key.marks:
                self.remove_one(mark)
        else:
            self.remove_one(key)
    
    def get_removed(self):
        return self.removed

    def add(self, key):
        super().add(key)
        if key in self.removed:
            self.removed.remove(key)
