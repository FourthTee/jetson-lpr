from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx
from openalpr import Alpr
import time
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import numpy as np
# set context
ctx = mx.gpu()
visualize = False
from_image = False
# load alpr model
alpr = Alpr("eu", "/etc/openalpr/openalpr.conf","/home/fourth/Desktop/repo/openalpr/runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
alpr.set_top_n(20)
alpr.set_default_region("md")
# load model
net = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True, ctx=ctx)
COLORS = np.random.uniform(0, 255, size=(91, 3))
if from_image:
    for filename in os.listdir('/home/fourth/Desktop/expr/lpr-Openalpr/input'):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            start = time.time()
            print(filename+"--")
            x, img = data.transforms.presets.ssd.load_test("input/"+filename, short=512)
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
            
            class_IDs = class_IDs.asnumpy()
            bounding_boxs = bounding_boxs.asnumpy()
            scores = scores.asnumpy()


            for i, obj in enumerate(class_IDs[0]):
                if obj[0] in [2, 3, 5, 7]:
                    if scores[0][i][0] > 0.6:
                        
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
            print((end - start) *1000)
            print("Inference time: "+str(end - start))
            print(plates)
            print(confidence)
else:
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()    

    while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        start = time.time()

        x, img = data.transforms.presets.ssd.transform_test(mx.nd.array(frame), short=512)
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
        class_IDs = class_IDs.asnumpy()
        bounding_boxs = bounding_boxs.asnumpy()
        scores = scores.asnumpy()


        for i, obj in enumerate(class_IDs[0]):
            if scores[0][i][0] > 0.6:
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[i], 2)
            else:
                break
        end = time.time()
        print("Time: "+str(end-start))
        if len(plates) > 0:
            print(plates)
            print(confidence)
        cv2.imshow("Frame", frame)
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # update the FPS counter
        fps.update()
        
    
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()