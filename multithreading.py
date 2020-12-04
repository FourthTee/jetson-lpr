from threading import Thread
import cv2


class VideoGet:
    def __init__(self, camera_id):
        self.stream = cv2.VideoCapture("/dev/video" + str(camera_id))
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def stop(self):
        self.stopped = True
        # self.stream.release()


class VideoShow:
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True