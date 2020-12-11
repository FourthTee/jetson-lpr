import tvm
from tvm.contrib import graph_runtime
from gluoncv import data, utils
import cv2
import mxnet as mx
from openalpr import Alpr
import time
import numpy as np
from imutils.video import FPS
import argparse
from util import get_alpr, evaluate, get_bbox, convertAsNumpy, build, run, draw_plates
import sys
from multithreading import (
    VideoGet,
    VideoShow,
    VideoCaptureThreading,
    infRunner,
    VidShow,
)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = (
        "v4l2src device=/dev/video{} ! "
        "video/x-raw, width=(int){}, height=(int){} ! "
        "videoconvert ! appsink"
    ).format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def detect(target: str, language: str, dir: str, camera: str):
    """
    Take feed from camera and detect vehicle (using TVM opt model),
    draw bounding box, and read license plate (based on the language)

    Parameters
    ----------
    target: str
        The device (CPU or GPU) that TVM will inference on
    language: str
        Region of the license plate that OpenALPR will detect
    dir: str
        Directory that has shared library, graphs, and params
    camera: str
        Specified camera input to use
    """

    alpr = get_alpr(language)
    ctx = tvm.context(target, 0)
    if ctx.exist:
        graph, lib, params = build(dir)
    else:
        print("Target does not exist")
        sys.exit(1)

    print("Starting video stream...")

    cap = VideoCaptureThreading("/dev/video" + camera)
    cap.start()
    module = graph_runtime.create(graph, lib, ctx)
    module.load_params(params)
    fps = FPS()
    # runner = infRunner(module, ctx, alpr, cap.x, cap.img, cap.oframe).start()
    fps = fps.start()
    while True:
        s = time.time()
        _, frame, oframe, x, img = cap.read()

        class_IDs, scores, bounding_boxs = run(x, module, ctx)

        class_IDs, bounding_boxs, scores = convertAsNumpy(
            class_IDs, bounding_boxs, scores
        )
        """
        if not _ or runner.stopped:
            runner.stop()
            cap.stop()
            break	
        runner.x = cap.x
        runner.oframe = cap.oframe
        runner.img = cap.img
        """

        oframe = draw_plates(class_IDs, scores, bounding_boxs, oframe, img, alpr)
        cv2.imshow("frame", oframe)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.stop()
            break

        fps.update()
    fps.stop()
    cv2.destroyAllWindows()
