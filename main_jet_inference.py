import gluoncv as gcv
from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx
from openalpr import Alpr
import time
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import numpy as np
import argparse
import jetson.inference
import jetson.utils
import random as rng
#os.system('sudo service nvargus-daemon restart')
cap_width=640
cap_height=480
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
def get_alpr():
    # load alpr model
    alpr = Alpr("eu", "/etc/openalpr/openalpr.conf","/home/nvidia/lpr/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    alpr.set_top_n(20)
    alpr.set_default_region("md")
    return alpr

def detect():
    alpr = get_alpr()
    net = jetson.inference.detectNet("ssd-mobilenet-v1", threshold=0.5)
    if (args.stream):
        print("Starting video stream...")
        #cap = cv2.VideoCapture('/dev/video1')
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        #camera = jetson.utils.videoSource("csi://0")
        #display = jetson.utils.videoOutput("display://0")
        fps = FPS().start()
        while True:
            #start = time.time()
            ret, frame = cap.read()
            img = frame.copy()
            #img = camera.Capture()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)
            img = jetson.utils.cudaFromNumpy(img)
            detections = net.Detect(img, 1280, 720)
            img = jetson.utils.cudaToNumpy(img, 1280, 720, 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR).astype(np.uint8)
	    
            #plates = []
            #confidence = []
    
            for obj in detections:
                classid = obj.ClassID
                x1,y1,x2,y2 = [int(i) for i in obj.ROI]
                if classid in [3,4,6,8]:
                    cropped = frame[y1:y2,x1:x2]
                    results = alpr.recognize_ndarray(cropped)
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (36,255,12), 2)
                    if len(results['results']) == 0:
                        continue
                    else:
                        #plates.append(results['results'][0]['plate'])
                        #confidence.append(results['results'][0]['confidence'])
                        plate = results['results'][0]['plate']
                        confidence = results['results'][0]['confidence']
                        cv2.putText(frame, plate+': '+confidence, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            #end = time.time()
            #print("Time: "+str(end-start))
            #if len(plates) > 0:
            #    print(plates)
            #    print(confidence)
            cv2.imshow('frame',frame)
            #display.Render(img)
            #display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        fps.stop()
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("Approx. FPS: {:.2f}".format(fps.fps()))
        # do a bit of cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--stream', help='Specify video stream')
    parser.add_argument('--visualize', default=False, help='Visualize bounding box image')
    args = parser.parse_args()
    detect()
