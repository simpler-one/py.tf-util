import numpy as np
from typing import Iterable NamedTuple


def roc(predictions, labels, target_class_id, resolution=20):
    """Get ROC

    :param np.ndarray[float] predictions: float[,]
    :param np.ndarray[int] or Iterable[int] labels: int[]
    :param int target_class_id:
    :param int resolution:
    :return:
    """
    prediction_list = sorted(
        (_Prediction(p, c) for p, c in zip(predictions, labels)),
        key=lambda prediction: prediction.probability[target_class_id],
        reverse=True
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

        result.append(_RocIntermediateItem(tp_count, fp_count))

    return [RocItem(r.tp_count / tp_count, r.fp_count / fp_count) for r in result]


class RocItem(NamedTuple):
    tp: float
    fp: float


class _Prediction(NamedTuple):
    probability: "np.ndarray[float]"
    right_class_id: int


class _RocIntermediateItem(NamedTuple):
    tp_count: int
    fp_count: int
