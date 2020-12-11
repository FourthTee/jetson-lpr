import threading
from threading import Thread
import cv2
import mxnet as mx
from gluoncv import data, utils
from util import get_alpr, evaluate, get_bbox, convertAsNumpy, build, run, draw_plates
from queue import *
from imutils.video import FPS


class infRunner:
    """ Class used to perform inference on a different thread """

    def __init__(self, module, ctx, alpr, x=None, img=None, oframe=None):
        self.x = x
        self.img = img
        self.alpr = alpr
        self.oframe = oframe
        self.stopped = False
        self.module = module
        self.ctx = ctx
        class_IDs, scores, bounding_boxs = run(x, module, ctx)
        self.class_IDs, self.scores, self.bounding_boxs = convertAsNumpy(
            class_IDs, bounding_boxs, scores
        )

    def start(self):
        Thread(target=self.inf, args=()).start()
        return self

    def inf(self):
        while not self.stopped:

            class_IDs, scores, bounding_boxs = run(self.x, self.module, self.ctx)

            class_IDs, bounding_boxs, scores = convertAsNumpy(
                class_IDs, bounding_boxs, scores
            )

            oframe = draw_plates(
                class_IDs, scores, bounding_boxs, self.oframe, self.img, self.alpr
            )

            cv2.imshow("frame", oframe)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stopped = True

    def get(self):
        return self.class_IDs, self.bounding_boxs, self.scores

    def stop(self):
        self.stopped = True


class VideoCaptureThreading:
    """ Class use to perform videocapture in parallel on a different thread """

    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.oframe = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.frame = mx.nd.array(cv2.cvtColor(self.oframe, cv2.COLOR_BGR2RGB)).astype(
            "uint8"
        )
        self.x, self.img = data.transforms.presets.ssd.transform_test(
            self.frame, short=480
        )

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print("[!] Threaded video capturing has already been started.")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.oframe = frame
                self.frame = mx.nd.array(
                    cv2.cvtColor(self.oframe, cv2.COLOR_BGR2RGB)
                ).astype("uint8")
                self.x, self.img = data.transforms.presets.ssd.transform_test(
                    self.frame, short=480
                )

    def read(self):
        with self.read_lock:
            # frame = self.frame.copy()
            grabbed = self.grabbed
            oframe = self.oframe.copy()
            frame = self.frame
            x = self.x
            img = self.img
        return grabbed, frame, oframe, x, img

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


class VidShow:
    """ Class used to display frames in parallel on a different thread """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True