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

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ("v4l2src device=/dev/video{} ! "
               "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
               "videoconvert ! appsink").format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def get_alpr():
    # load alpr model
    alpr = Alpr("eu", "/etc/openalpr/openalpr.conf","/home/fourth/Desktop/repo/openalpr/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    alpr.set_top_n(20)
    alpr.set_default_region("md")
    return alpr

def detect():
    ctx = mx.gpu()
    visualize = args.visualize
    from_image = args.image
    alpr = get_alpr()
    COLORS = np.random.uniform(0, 255, size=(91, 3))
    
    # load model
    model_name = "ssd_512_mobilenet1.0_voc"
    #model_name = "yolo3_darknet53_voc"
    net = model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()
    if (args.image):
        for filename in os.listdir(os.getcwd()+"/"+args.image):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                start = time.time()
                print(filename+"--")
                x, img = data.transforms.presets.ssd.load_test(args.image+"/"+filename, short=512)
                x = x.as_in_context(ctx)

                # call forward and show plot
                class_IDs, scores, bounding_boxs = net(x)
                #print(scores)
                if visualize:
                    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                            class_IDs[0], class_names=net.classes)
                    plt.show()
                plates = []
                confidence = []
                #mx.nd.waitall()
                
                #print(time.time() - start)

                class_IDs = class_IDs.asnumpy()
                bounding_boxs = bounding_boxs.asnumpy()
                scores = scores.asnumpy()

                for i, obj in enumerate(class_IDs[0]):
                    if scores[0][i][0] > 0.6:
                        if obj[0] in [5, 6]:
                    
                            x1 = bounding_boxs[0][i][0]
                            y1 = bounding_boxs[0][i][1]
                            x2 = bounding_boxs[0][i][2]
                            y2 = bounding_boxs[0][i][3]
                            
                            cropped = img[int(y1):int(y2), int(x1):int(x2)]
                            results = alpr.recognize_ndarray(cropped)
                            
                            if len(results['results']) == 0:
                                continue
                            else:
                                plates.append(results['results'][0]['plate'])
                                confidence.append(results['results'][0]['confidence'])
                    else:
                        break
                end = time.time()
                print("Inference time: "+str((end - start)*1000) + " ms")
                print("Plates: "+ str(plates))
                print("Confidence: "+ str(confidence))
    elif (args.stream):
        print("Starting video stream...")
        #cap = cv2.VideoCapture('/dev/video1')
        #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=1), cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(0)
        fps = FPS().start()
        while True:
            start = time.time()
            ret, frame = cap.read()
            #print("frame read")
            oframe = frame
            #cv2.imshow('frame', frame)
            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            #print("frame collected")
            

            x, img = data.transforms.presets.ssd.transform_test(mx.nd.array(frame), short=512)
            x = x.as_in_context(ctx)
            #print("Loading predictions")
            class_IDs, scores, bounding_boxs = net(x)
            #print(scores)
            if visualize:
                ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                        class_IDs[0], class_names=net.classes)
                plt.show()
            #print("Prediction complete")
            plates = []
            confidence = []
            #mx.nd.waitall()
            class_IDs = class_IDs.asnumpy()
            bounding_boxs = bounding_boxs.asnumpy()
            scores = scores.asnumpy()
            #img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
            #gcv.utils.viz.cv_plot_image(img)
            #print("Checking Plates")
            #img = oframe
            for i, obj in enumerate(class_IDs[0]):
                if scores[0][i][0] > 0.6:
                    if obj[0] in [5, 6]:
                        #print("Found")
                        x1 = bounding_boxs[0][i][0]
                        y1 = bounding_boxs[0][i][1]
                        x2 = bounding_boxs[0][i][2]
                        y2 = bounding_boxs[0][i][3]
                        oframe = cv2.rectangle(oframe, (x1, y1), (x2, y2), (36,255,12), 2)
                        cropped = img[int(y1):int(y2), int(x1):int(x2)]
                        results = alpr.recognize_ndarray(cropped)
                        
                        if len(results['results']) == 0:
                            continue
                        else:
                            plate = results['results'][0]['plate']
                            plates.append(results['results'][0]['plate'])
                            confidence.append(results['results'][0]['confidence'])
                            cv2.putText(oframe, plate, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        
                else:
                    break
            
            end = time.time()
            #print("Time: "+str(end-start))
            if len(plates) > 0:
                print(plates)
                print(confidence)
            cv2.imshow('frame',oframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        fps.stop()
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("Approx. FPS: {:.2f}".format(fps.fps()))
        # do a bit of cleanup
        cap.release()
        cv2.destroyAllWindows()

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ("v4l2src device=/dev/video{} ! "
               "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
               "videoconvert ! appsink").format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--stream', help='Specify video stream')
    parser.add_argument('--visualize', default=False, help='Visualize bounding box image')
    args = parser.parse_args()
    detect()
