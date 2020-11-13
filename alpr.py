import os
import time

import jetson.inference
import jetson.utils

import cv2
import requests
import threading
import numpy as np
from imutils.video import FPS
from openalpr import Alpr



cap_width=1920
cap_height=1080

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
alpr = Alpr("us", "/etc/openalpr/openalpr.conf","/usr/share/openalpr/runtime_data")
os.system('sudo service nvargus-daemon restart')

def gstreamer_pipeline(
    capture_width=cap_width,
    capture_height=cap_height,
    display_width=960,
    display_height=616,
    framerate=30/1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)
    img = jetson.utils.cudaFromNumpy(img)
    detections = net.Detect(img, 1280, 720)
    img = jetson.utils.cudaToNumpy(img, 1280, 720, 4)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR).astype(np.uint8)
    return img, detections


def get_bbox(detection):
    num=detection.ClassID
    x1,y1,x2,y2=[int(i) for i in detection.ROI]
    return  num,x1,y1,x2,y2

def detect_plate(crop_frame):
    return alpr.recognize_ndarray(crop_frame)


def send_picture(frame):
    cv2.imwrite('temp.jpg',frame)
    url = 'http://172.28.253.160:8000'
    files = {'media': open('temp.jpg', 'rb')}
    return requests.post(url, files=files)

def calculate_vote(arr):
    results=None
    if arr:
        stats={i:arr.count(i) for i in arr}
        results=max(stats, key=lambda key: stats[key])

    return results,[]

def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture('../video-test2.mp4')
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        fps_i=None
        detection=0
        empty_frame=0
        arr=[]
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            fps=FPS().start()
            ret_val, img = cap.read()
            frame = img.copy()
            img,detections=detect(img)


            for i in range(len(detections)):
                num,x1,y1,x2,y2=get_bbox(detections[i])

                if num in [3,4,6,8]:
                    detection=detection+1
                    crop = frame[y1:y2,x1:x2]
                    results=detect_plate(crop)

                    if results['results']:
                        for i,plate in enumerate(results['results'][0]['candidates']):
                            if plate['confidence']>0.4:
                                arr.append(plate['plate'])
                                #print(arr)
                                if i>4:
                                    break


                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)

            if detection>=10:
                results,arr=calculate_vote(arr)
                send_picture(frame)
                print(results)
                detection=0

            cv2.imshow("CSI Camera", img)
            if fps_i is None:
                    fps_i=0

            fps.update()
            fps.stop()
            fps_i=fps.fps()
            #print("FPS: {:.2f}".format(fps.fps()))

            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    main()
