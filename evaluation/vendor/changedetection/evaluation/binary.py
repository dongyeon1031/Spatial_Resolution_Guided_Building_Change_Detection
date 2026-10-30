from dataclasses import dataclass

import numpy as np

from .common import ConfusionMatrixEvaluator, safe_divide


@dataclass(frozen=True)
class BinaryChangeMetrics:
    recall: float
    precision: float
    oa: float
    f1: float
    iou: float
    miou: float
    kappa: float


class BinaryChangeEvaluator(ConfusionMatrixEvaluator):
    def __init__(self):
        super().__init__(num_classes=2)

    def compute(self):
        matrix = self.confusion_matrix
        tp = matrix[1, 1]
        tn = matrix[0, 0]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        total = matrix.sum()

        recall = safe_divide(tp, tp + fn)
        precision = safe_divide(tp, tp + fp)
        oa = safe_divide(np.trace(matrix), total)
        f1 = safe_divide(2 * recall * precision, recall + precision)
        iou = safe_divide(tp, tp + fp + fn)
        background_iou = safe_divide(tn, tn + fp + fn)
        miou = 0.5 * (background_iou + iou)

        observed_accuracy = safe_divide(np.trace(matrix), total)
        expected_accuracy = np.sum(
            safe_divide(np.sum(matrix, axis=0), total) * safe_divide(np.sum(matrix, axis=1), total)
        )
        kappa = safe_divide(observed_accuracy - expected_accuracy, 1 - expected_accuracy)

        return BinaryChangeMetrics(
            recall=float(recall),
            precision=float(precision),
            oa=float(oa),
            f1=float(f1),
            iou=float(iou),
            miou=float(miou),
            kappa=float(kappa),
        )
