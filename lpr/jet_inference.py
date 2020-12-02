from openalpr import Alpr
import time
import os
import imutils
from imutils.video import FPS
import cv2
import numpy as np
import argparse
import jetson.inference
import jetson.utils
from util import get_alpr


def detect(language, camera):
    """
    Take feed from camera dn detect vehicle (using jetson-inference package),
    draw bounding box, and read license plate (based on the language)
    """

    alpr = get_alpr(language)
    net = jetson.inference.detectNet("ssd-mobilenet-v1", threshold=0.5)

    print("Starting video stream...")
    cap = cv2.VideoCapture("/dev/video" + camera)
    if not cap.isOpened():
        raise Exception("Could not open video device")

    fps = FPS().start()
    while True:
        ret, frame = cap.read()
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)
        img = jetson.utils.cudaFromNumpy(img)
        detections = net.Detect(img, 1280, 720)
        img = jetson.utils.cudaToNumpy(img, 1280, 720, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR).astype(np.uint8)

        for obj in detections:
            classid = obj.ClassID
            x1, y1, x2, y2 = [int(i) for i in obj.ROI]
            if classid in [3, 4, 6, 8]:
                cropped = frame[y1:y2, x1:x2]
                results = alpr.recognize_ndarray(cropped)
                frame = cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2
                )
                if len(results["results"]) == 0:
                    continue
                else:
                    plate = results["results"][0]["plate"]
                    confidence = results["results"][0]["confidence"]
                    cv2.putText(
                        frame,
                        plate + ": " + str(confidence),
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()
    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # clean up capture window
    cap.release()
    cv2.destroyAllWindows()