from util import get_alpr, get_bbox, convertAsNumpy
import cv2
from mxnet import nd
import numpy as np

# Test functions in the util.py file


def open_alpr_fetch_test():
    """
    Check if get_alpr function is able to return alpr object and throw approperiate errors
    """

    error = False
    try:
        a = get_alpr("eu", "/srv/openalpr")
    except Exception as exc:
        error = True
        print("Cannot create ALPR object")
    assert error == False, "Test Failed"
    try:
        a = get_alpr("th", "/hello")
        print("Did not raise error when config is invalid")
        error = True
    except Exception as exc:
        print("[1/3] Test alpr Pass")
    assert error == False, "Test Failed"


def get_bbox_test():
    """
    Check that get_bbox function is able to correcly return the approperiate coordinates
    """

    bbox = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]

    assert tuple(get_bbox(bbox, 0)) == (1, 2, 3, 4), "Test Fail"
    assert tuple(get_bbox(bbox, 1)) == (5, 6, 7, 8), "Test Fail"
    assert tuple(get_bbox(bbox, 2)) == (9, 10, 11, 12), "Test Fail"
    assert tuple(get_bbox(bbox, 3)) == (13, 14, 15, 16), "Test Fail"

    print("[2/3] Test get_boxx Pass")


def numpy_convert_test():
    """
    Check that convertAsNumpy is able to convert nd.array to np.array
    """

    a = nd.array([1, 2, 3])
    b = nd.array([1, 2, 3])
    c = nd.array([1, 2, 3])
    a_new, b_new, c_new = convertAsNumpy(a, b, c)

    assert isinstance(a_new, np.ndarray)
    assert isinstance(b_new, np.ndarray)
    assert isinstance(c_new, np.ndarray)
    print("[3/3] Test numpy_convert Pass")


if __name__ == "__main__":
    open_alpr_fetch_test()
    get_bbox_test()
    numpy_convert_test()
