import gluoncv as gcv
from gluoncv import model_zoo, data, utils
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
from util import get_alpr, get_bbox, convertAsNumpy

def detect(language):
    ctx = mx.gpu()
    visualize = args.visualize
    alpr = get_alpr(language)
    
    # load model
    model_name = "ssd_512_mobilenet1.0_voc"
    net = model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()

    if (args.image):
        for filename in os.listdir(os.getcwd()+"/"+args.image):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                start = time.time()
                print(filename+"--")
                x, img = data.transforms.presets.ssd.load_test(args.image+"/"+filename, short=512)
                x = x.as_in_context(ctx)

                class_IDs, scores, bounding_boxs = net(x)

                if visualize:
                    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                            class_IDs[0], class_names=net.classes)
                    plt.show()

                plates = []
                confidence = []
                class_IDs, bounding_boxs, scores = convertAsNumpy(class_IDs, bounding_boxs, scores)
                for i, obj in enumerate(class_IDs[0]):
                    if scores[0][i][0] > 0.6:
                        if obj[0] in [5, 6]:
                    
                            x1, y1, x2, y2 = get_bbox(bounding_boxs, i)
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
        #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(int(args.stream))
        if not cap.isOpened():
            raise Exception("Could not open video device")

        fps = FPS().start()
        while True:
            ret, frame = cap.read()
            oframe = frame

            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            x, img = data.transforms.presets.ssd.transform_test(frame, short=480)
            x = x.as_in_context(ctx)
       
            class_IDs, scores, bounding_boxs = net(x)
            
            class_IDs, bounding_boxs, scores = convertAsNumpy(class_IDs, bounding_boxs, scores)
            
            for i, obj in enumerate(class_IDs[0]):
                if scores[0][i][0] > 0.6:
                    if obj[0] in [5, 6]:
                        x1, y1, x2, y2 = get_bbox(bounding_boxs, i)
                        oframe = cv2.rectangle(oframe, (x1, y1), (x2, y2), (36,255,12), 2)
                        cropped = img[int(y1):int(y2), int(x1):int(x2)]
                        results = alpr.recognize_ndarray(cropped)
                        if len(results['results']) == 0:
                            continue
                        else:
                            plate = results['results'][0]['plate']
                            confidence = results['results'][0]['confidence']
            
                            cv2.putText(oframe, plate + ": " + str(confidence), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                else:
                    break

            cv2.imshow('frame',oframe)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        fps.stop()
        print("Approx. FPS: {:.2f}".format(fps.fps()))
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--stream', help='Specify video stream')
    parser.add_argument('--visualize', default=False, help='Visualize bounding box image')
    args = parser.parse_args()
    detect("eu")
