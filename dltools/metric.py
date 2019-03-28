import numpy as np


def iou_bitmap(y_true, y_pred, verbose=False):
    """
    Compute the IoU between two arrays

    If the arrays are probabilities (floats) instead of predictions (integers
    or booleans) they are automatically rounded to the nearest integer and
    converted to bool before the IoU is computed.

    Parameters
    ----------
    y_true : ndarray
        array of true labels
    y_pred : ndarray
        array of predicted labels
    verbose : bool (optional)
        print the intersection and union separately

    Returns
    -------
    float :
        the intersection over union (IoU) value scaled between 0.0 and 1.0

    """
    EPS = np.finfo(float).eps

    # Make sure each pixel was predicted e.g. turn probability into prediction
    if y_true.dtype in [np.float32, np.float64]:
        y_true = y_true.round().astype(bool)

    if y_pred.dtype in [np.float32, np.float64]:
        y_pred = y_pred.round().astype(bool)

    # Reshape to 1d
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Compute intersection and union
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = (intersection + EPS) / (sum_ - intersection + EPS)

    if verbose:
        print('Intersection:', intersection)
        print('Union:', sum_ - intersection)

    return jac


def iou(y_true, y_pred):
    iou_list = [iou_bitmap(yt, yp)
                for (yt, yp) in zip(y_true, y_pred)]
    return np.mean(iou_list)
