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
    cap = cv2.VideoCapture("/dev/video" + camera)
    if not cap.isOpened():
        print("Could not open video device (change video_camera)")
        sys.exit(1)
    module = graph_runtime.create(graph, lib, ctx)
    module.load_params(params)
    fps = FPS().start()
    while True:

        ret, frame = cap.read()
        oframe = frame
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype("uint8")
        x, img = data.transforms.presets.ssd.transform_test(frame, short=480)

        class_IDs, scores, bounding_boxs = run(x, module, ctx)

        class_IDs, bounding_boxs, scores = convertAsNumpy(
            class_IDs, bounding_boxs, scores
        )

        oframe = draw_plates(class_IDs, scores, bounding_boxs, oframe, img, alpr)

        cv2.imshow("frame", oframe)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()
    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # clean up capture window
    cap.release()
    cv2.destroyAllWindows()
