import numpy as np
from typing import NamedTuple


def roc(predictions, labels, target_class_id, resolution=20):
    """Get ROC

    :param np.ndarray[float] predictions: float[,]
    :param np.ndarray[int] labels: int[]
    :param int target_class_id:
    :param int resolution:
    :return:
    """
    prediction_list = sorted(
        (Prediction(p, c) for p, c in zip(predictions, labels)),
        key=lambda prediction: prediction.probability[target_class_id]
    )

    tp_count = 0
    fp_count = 0
    result = []
    for step in range(resolution):
        start_i = int(round(step * len(prediction_list) / resolution))
        stop_i = int(round((1 + step) * len(prediction_list) / resolution))

        for p in prediction_list[start_i:stop_i]:
            if p.right_class_id == target_class_id:
                tp_count += 1
            else:
                fp_count += 1

        result.append(RocIntermediateItem(tp_count, fp_count))

    return [RocItem(r.tp_count / tp_count, r.fp_count / fp_count) for r in result]


class Prediction(NamedTuple):
    probability: np.ndarray[float]
    right_class_id: int


class RocIntermediateItem(NamedTuple):
    tp_count: int
    fp_count: int


class RocItem(NamedTuple):
    tp_rate: float
    fp_rate: float
