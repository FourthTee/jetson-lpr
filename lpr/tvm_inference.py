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
from util import get_alpr, evaluate, get_bbox, convertAsNumpy, build, run


def detect(target, language, dir, camera):
    """
    Take feed from camera and detect vehicle (using TVM opt model),
    draw bounding box, and read license plate (based on the language)
    """
    alpr = get_alpr(language)
    ctx = tvm.context(target, 0)
    if ctx.exist:
        graph, lib, params = build(dir)
    else:
        raise Exception("Target does not exist")

    print("Starting video stream...")
    cap = cv2.VideoCapture("/dev/video" + camera)
    if not cap.isOpened():
        raise Exception("Could not open video device")

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

        for i, obj in enumerate(class_IDs[0]):
            if scores[0][i][0] > 0.6:
                if obj[0] in [5, 6]:
                    x1, y1, x2, y2 = get_bbox(bounding_boxs, i)
                    oframe = cv2.rectangle(oframe, (x1, y1), (x2, y2), (36, 255, 12), 2)
                    cropped = img[int(y1) : int(y2), int(x1) : int(x2)]
                    results = alpr.recognize_ndarray(cropped)

                    if len(results["results"]) == 0:
                        continue
                    else:
                        plate = results["results"][0]["plate"]
                        confidence = results["results"][0]["confidence"]
                        cv2.putText(
                            oframe,
                            plate + ": " + str(confidence),
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (36, 255, 12),
                            2,
                        )

            else:
                break

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
