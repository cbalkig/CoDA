from typing import List


class Metrics:
    def __init__(self, acc: float, f1: float, precision: float, recall: float, cr: dict,
                 cm: List[List[int]]) -> None:
        self.acc = acc
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.cr = cr  # Classification report
        self.cm = cm  # Confusion matrix
