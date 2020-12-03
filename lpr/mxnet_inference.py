import gluoncv as gcv
from gluoncv import model_zoo, data, utils
import mxnet as mx
from openalpr import Alpr
import time
import os
import imutils
from imutils.video import FPS
import cv2
import numpy as np
from util import get_alpr, draw_plates, convertAsNumpy
import sys


def detect(target: str, language: str, camera: str):
    """
    Take feed from camera and detect vehicle (using mxnet),
    draw bounding box, and read license plate (based on the language)

    Parameters
    ----------
    target: str
        The device (CPU or GPU) that mxnet will inference on
    language: str
        Region of the license plate that OpenALPR will detect
    camera: str
        Specified camera input to use
    """

    if target == "llvm":
        ctx = mx.cpu()
    elif target == "cuda":
        ctx = mx.gpu()
    else:
        print("Target does not exist")
        sys.exit(1)

    alpr = get_alpr(language)
    model_name = "ssd_512_mobilenet1.0_voc"
    net = model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()

    print("Starting video stream...")
    cap = cv2.VideoCapture("/dev/video" + camera)
    if not cap.isOpened():
        print("Could not open video device (change video_camera)")
        sys.exit(1)

    fps = FPS().start()
    while True:
        ret, frame = cap.read()
        oframe = frame

        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype("uint8")
        x, img = data.transforms.presets.ssd.transform_test(frame, short=480)
        x = x.as_in_context(ctx)

        class_IDs, scores, bounding_boxs = net(x)

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
